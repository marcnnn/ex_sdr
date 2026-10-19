defmodule ExSdr.Digital.Kernels do
  @moduledoc """
  Stream kernels of the digital family: `step(item, state, opts) ->
  {output, next_state}` (see `NxFixed.run_scan/5`), written against
  `NxFixed.Num` where they handle samples, so that `:format` gives the float
  reference or a fixed-point instance that passes the width check.

  Symbols, bits and bytes are `u8`. Constants (tables, taps, polynomials)
  come in as options and are folded into the graph.
  """
  import Nx.Defn
  import NxFixed.Num

  alias NxFixed.Fixed
  alias NxFixed.Num.Format
  alias NxFixed.Primitives, as: P

  # -- table lookups -------------------------------------------------------------------

  @doc """
  The entry of `values` (floats) at `idx` (a `u8` symbol, clipped to the
  table) in `format`: an `f32`, or a `Fixed` for fixed point.
  """
  deftransform lookup(idx, values, %Format{} = f) do
    idx = Nx.clip(idx, 0, length(values) - 1)

    case f.kind do
      :float ->
        Nx.take(Nx.tensor(values, type: :f32, backend: Nx.BinaryBackend), idx)

      :fixed ->
        ints = quantize_constants(values, f)
        table = Nx.tensor(ints, type: container(f.bits), backend: Nx.BinaryBackend)
        %Fixed{t: Nx.take(table, idx), frac: f.frac}
    end
  end

  @doc "`chunks_to_symbols_bf`: symbol to the real value `:table` gives it."
  defn chunks_to_symbols_real(sym, state, opts \\ []) do
    opts = keyword!(opts, [:format, :table])
    {lookup(sym, opts[:table], opts[:format]), state}
  end

  @doc "`chunks_to_symbols_bc`: symbol to the complex value of `:table` (`{re, im}` floats)."
  defn chunks_to_symbols_complex(sym, state, opts \\ []) do
    opts = keyword!(opts, [:format, :table])
    {complex_lookup(sym, opts[:table], opts[:format]), state}
  end

  deftransformp complex_lookup(sym, table, f) do
    {lookup(sym, Enum.map(table, &elem(&1, 0)), f), lookup(sym, Enum.map(table, &elem(&1, 1)), f)}
  end

  @doc "`map_bb`: `out = map[in]` (`:map`, a list of bytes)."
  defn map_bytes(x, state, opts \\ []) do
    opts = keyword!(opts, [:map])
    {map_lookup(x, opts[:map]), state}
  end

  deftransformp map_lookup(x, map) do
    table = Nx.tensor(map, type: :u8, backend: Nx.BinaryBackend)
    Nx.take(table, Nx.clip(x, 0, length(map) - 1))
  end

  # -- decisions -------------------------------------------------------------------------

  @doc """
  `constellation_decoder_cb`: index of the nearest of `:points` (a list of
  `{re, im}`), ties to the smaller index.
  """
  defn decide(x, state, opts \\ []) do
    opts = keyword!(opts, [:format, :points])
    {nearest(x, opts[:points], opts[:format]), state}
  end

  deftransformp nearest({re, im}, points, f) do
    dist =
      for {pr, pi} <- points do
        dr = sub(re, constant(pr, f))
        di = sub(im, constant(pi, f))
        add(mul(dr, dr), mul(di, di)) |> raw()
      end

    # running minimum, strict comparison: ties keep the smaller index
    [first | rest] = dist

    {_, index} =
      rest
      |> Enum.with_index(1)
      |> Enum.reduce({first, Nx.tensor(0, type: :u8)}, fn {d, i}, {best, index} ->
        better = Nx.less(d, best)
        {Nx.select(better, d, best), Nx.select(better, Nx.tensor(i, type: :u8), index)}
      end)

    index
  end

  defp raw(%Fixed{t: t}), do: t
  defp raw(t), do: t

  @doc "`binary_slicer_fb`: 1 where the sample is not negative."
  defn slice(x, state, opts \\ []) do
    _ = keyword!(opts, [:format])
    {non_negative(x), state}
  end

  deftransformp non_negative(%Fixed{t: t}), do: t |> Nx.greater_equal(0) |> Nx.as_type(:u8)
  deftransformp non_negative(x), do: x |> Nx.greater_equal(0) |> Nx.as_type(:u8)

  @doc """
  `constellation_soft_decoder_cf` on a whole frame: `items` is `{re, im}`
  (`f32[n]`), the result `f32[n, bits_per_symbol]`, the log-likelihood
  ratio of each bit of the nearest-point index (most significant first),
  positive for 1: `ln(sum_{bit=1} exp(-d^2/npwr) / sum_{bit=0} exp(-d^2/npwr))`.
  Float only (it needs exp and log).
  """
  def soft_decide({re, im}, _state, opts) do
    points = Keyword.fetch!(opts, :points)
    npwr = Keyword.fetch!(opts, :npwr)
    labels = Keyword.fetch!(opts, :labels)
    m = length(points)
    k = round(:math.log2(m))
    {pr, pi} = points |> Enum.unzip() |> then(fn {a, b} -> {Nx.tensor(a), Nx.tensor(b)} end)

    dr = Nx.subtract(Nx.new_axis(Nx.as_type(re, :f32), 1), pr)
    di = Nx.subtract(Nx.new_axis(Nx.as_type(im, :f32), 1), pi)
    metric = dr |> Nx.pow(2) |> Nx.add(Nx.pow(di, 2)) |> Nx.divide(-npwr)

    cols =
      for j <- 0..(k - 1) do
        ones = Nx.tensor(for(l <- labels, do: Bitwise.band(Bitwise.bsr(l, k - 1 - j), 1)))
        Nx.subtract(masked_lse(metric, ones), masked_lse(metric, Nx.subtract(1, ones)))
      end

    {Nx.stack(cols, axis: 1), nil}
  end

  defp masked_lse(metric, mask) do
    big = Nx.tensor(-3.0e38, type: :f32)
    pred = Nx.broadcast(Nx.equal(mask, 1), Nx.shape(metric))
    mx = Nx.select(pred, metric, big) |> Nx.reduce_max(axes: [1], keep_axes: true)
    e = metric |> Nx.subtract(mx) |> Nx.exp() |> Nx.multiply(mask)
    mx |> Nx.squeeze(axes: [1]) |> Nx.add(Nx.log(Nx.sum(e, axes: [1])))
  end

  @doc "Complex gain: `y = g * x` (`multiply_const_cc` with a real constant)."
  defn scale_complex(x, state, opts \\ []) do
    opts = keyword!(opts, [:format, :gain])
    f = opts[:format]
    {quantize(cscale(x, constant(opts[:gain], f)), f), state}
  end

  # -- differential coding -----------------------------------------------------------------

  @doc "`diff_encoder_bb`: `out = (in + previous out) mod modulus`; the state is the previous output."
  defn diff_encode(x, prev, opts \\ []) do
    opts = keyword!(opts, [:modulus])
    sum = Nx.as_type(x, :u16) + Nx.as_type(prev, :u16)
    out = sum |> Nx.remainder(opts[:modulus]) |> Nx.as_type(:u8)
    {out, out}
  end

  @doc "`diff_decoder_bb`: `out = (in - previous in) mod modulus`; the state is the previous input."
  defn diff_decode(x, prev, opts \\ []) do
    opts = keyword!(opts, [:modulus])
    m = opts[:modulus]
    d = Nx.as_type(x, :s16) - Nx.as_type(prev, :s16)
    out = d |> Nx.remainder(m) |> plus(m) |> Nx.remainder(m) |> Nx.as_type(:u8)
    {out, x}
  end

  deftransform plus(x, m), do: Nx.add(x, m)

  @doc "`diff_phasor_cc`: `y = x * conj(previous x)`; the state is the previous input."
  defn diff_phasor(x, prev, opts \\ []) do
    opts = keyword!(opts, [:format])
    f = opts[:format]
    {xr, xi} = x
    {pr, pi_} = prev
    re = add(mul(xr, pr), mul(xi, pi_))
    im = sub(mul(xi, pr), mul(xr, pi_))
    {quantize({re, im}, f), x}
  end

  # -- scramblers --------------------------------------------------------------------------

  # XOR of the register bits at `taps` (positions of the mask's set bits)
  deftransformp parity(sr, taps) do
    taps
    |> Enum.map(fn k -> sr |> Nx.right_shift(k) |> Nx.bitwise_and(1) end)
    |> Enum.reduce(Nx.tensor(0, type: :u32), &Nx.bitwise_xor/2)
  end

  deftransformp shift_in(sr, bit, len) do
    sr |> Nx.right_shift(1) |> Nx.bitwise_or(Nx.left_shift(bit, len))
  end

  @doc """
  `scrambler_bb` (multiplicative, on the LSB of each byte): GNU Radio's LFSR
  with `:taps` (bit positions of the mask) and register `:len`; the state
  is the register (`u32`, `len + 1` bits).
  """
  defn scramble(x, sr, opts \\ []) do
    opts = keyword!(opts, [:taps, :len])
    bit = x |> Nx.bitwise_and(1) |> Nx.as_type(:u32)
    out = sr |> Nx.bitwise_and(1) |> Nx.as_type(:u8)
    fb = Nx.bitwise_xor(parity(sr, opts[:taps]), bit)
    {out, shift_in(sr, fb, opts[:len])}
  end

  @doc "`descrambler_bb`, see `scramble/3`."
  defn descramble(x, sr, opts \\ []) do
    opts = keyword!(opts, [:taps, :len])
    bit = x |> Nx.bitwise_and(1) |> Nx.as_type(:u32)
    out = parity(sr, opts[:taps]) |> Nx.bitwise_xor(bit) |> Nx.as_type(:u8)
    {out, shift_in(sr, bit, opts[:len])}
  end

  @doc """
  `additive_scrambler_bb`: XORs the low `:bits_per_byte` bits of every byte
  with LFSR output bits (first bit in the LSB), resetting the register to
  `:seed` after every `:count` bytes (0: never). The state is
  `{register, bytes since reset}`.
  """
  defn additive_scramble(x, state, opts \\ []) do
    opts = keyword!(opts, [:taps, :len, :seed, :count, :bits_per_byte])
    {sr, n} = state
    {mask, sr} = lfsr_bits(sr, opts[:taps], opts[:len], opts[:bits_per_byte])
    out = Nx.bitwise_xor(x, mask)
    {out, count_reset(sr, n, opts[:seed], opts[:count])}
  end

  deftransformp lfsr_bits(sr, taps, len, bits) do
    Enum.reduce(0..(bits - 1)//1, {Nx.tensor(0, type: :u8), sr}, fn k, {mask, sr} ->
      out = sr |> Nx.bitwise_and(1) |> Nx.as_type(:u8) |> Nx.left_shift(k)
      {Nx.bitwise_or(mask, out), shift_in(sr, parity(sr, taps), len)}
    end)
  end

  deftransformp count_reset(sr, _n, _seed, 0), do: {sr, Nx.tensor(0, type: :u32)}

  deftransformp count_reset(sr, n, seed, count) do
    # n <= count always; the min tells the width check so
    n = n |> Nx.min(count) |> Nx.as_type(:s32) |> Nx.add(1)
    done = Nx.greater_equal(n, count)
    {Nx.select(done, Nx.tensor(seed, type: :u32), sr), Nx.select(done, 0, n) |> Nx.as_type(:u32)}
  end

  # -- pulse shaping and modulation --------------------------------------------------------

  @doc """
  Interpolating FIR on real samples (`interp_fir_filter_fff`): `sps`
  outputs per input, `y[m * sps + p] = sum_j taps[p + j * sps] * x[m - j]`.
  The state is the delay line (`ceil(length(taps) / sps) - 1` samples).
  Outputs have shape `{sps}`.
  """
  defn interp_fir_real(x, line, opts \\ []) do
    opts = keyword!(opts, [:format, :taps, :sps])
    f = opts[:format]
    {acc, next} = polyphase(x, line, opts[:taps], opts[:sps], f)
    {quantize(acc, f), next}
  end

  @doc "Complex samples with real taps (`interp_fir_filter_ccf`), state `{line_re, line_im}`."
  defn interp_fir_complex(x, lines, opts \\ []) do
    opts = keyword!(opts, [:format, :taps, :sps])
    f = opts[:format]
    {re, im} = x
    {lr, li} = lines
    {acc_re, next_re} = polyphase(re, lr, opts[:taps], opts[:sps], f)
    {acc_im, next_im} = polyphase(im, li, opts[:taps], opts[:sps], f)
    {quantize({acc_re, acc_im}, f), {next_re, next_im}}
  end

  @doc "Number of delay line samples of the interpolating FIR with `ntaps` taps and `sps`."
  def line_length(ntaps, sps), do: max(div(ntaps + sps - 1, sps), 2) - 1

  deftransformp polyphase(x, line, taps, sps, f) do
    k = line_length(length(taps), sps) + 1
    padded = taps ++ List.duplicate(0.0, k * sps - length(taps))

    results =
      for p <- 0..(sps - 1) do
        phase_taps = for j <- 0..(k - 1), do: Enum.at(padded, p + j * sps)
        fir(x, line, phase_taps, f)
      end

    {_, next} = hd(results)
    accs = Enum.map(results, &elem(&1, 0))

    acc =
      case hd(accs) do
        %Fixed{frac: frac} -> %Fixed{t: accs |> Enum.map(& &1.t) |> Nx.stack(), frac: frac}
        _ -> Nx.stack(accs)
      end

    {acc, next}
  end

  @doc """
  `frequency_modulator_fc`: `phase += sensitivity * x` (radians), output
  `exp(j * phase)`. Float: the state is the phase in radians wrapped to
  [-pi, pi). Fixed point: the state is a `u16` phase word (full circle 65536),
  the increment is rounded to whole phase words and cos/sin come from the
  format's sine table (`NxFixed.Num.oscillator/3`).
  """
  defn frequency_modulate(x, phase, opts \\ []) do
    opts = keyword!(opts, [:format, :sensitivity])
    fm_step(x, phase, opts[:sensitivity], opts[:format])
  end

  deftransformp fm_step(x, phase, sensitivity, %Format{kind: :float}) do
    pi = :math.pi()
    p = Nx.add(phase, Nx.multiply(x, Nx.tensor(sensitivity, type: :f32)))
    p = Nx.subtract(p, Nx.multiply(2 * pi, Nx.floor(Nx.divide(Nx.add(p, pi), 2 * pi))))
    p = Nx.as_type(p, :f32)
    {{Nx.cos(p), Nx.sin(p)}, p}
  end

  deftransformp fm_step(%Fixed{t: t, frac: frac}, phase, sensitivity, %Format{kind: :fixed} = f) do
    # phase words per unit of x, with 8 fractional bits
    k = round(sensitivity * 65536 / (2 * :math.pi()) * 256)
    inc = t |> Nx.as_type(:s64) |> Nx.multiply(k) |> P.shift_round(frac + 8)
    {nco, next} = oscillator(phase, P.wrap(inc, :u16), f)
    {nco, next}
  end

  defp container(bits) when bits <= 8, do: :s8
  defp container(bits) when bits <= 16, do: :s16
  defp container(bits) when bits <= 32, do: :s32
  defp container(_bits), do: :s64
end
