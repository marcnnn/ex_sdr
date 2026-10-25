defmodule ExSdr.Comm.OFDM.FixedFrontEnd do
  @moduledoc """
  OFDM receiver front end in integer arithmetic: the stages of
  `ExSdr.Comm.OFDM.FrontEnd` (same spec map, same output: one `u8` code
  bit per byte), written once against `NxFixed.Num`, so `:format` selects
  the fixed-point version for CPU and FPGA or its float reference.

    * Input: interleaved s16 I/Q (`sc16`, as radios deliver it), full
      scale = 1.0.
    * Frequency correction: 32-bit phase word, `inc` per sample, sine
      table (`:coef` format's `lut_bits`). A chunk starting `s` samples
      into the stream starts at phase `inc * s mod 2^32`, so chunks join
      exactly.
    * FFT: `ExSdr.DSP.FFT` in the `:format` data format, scaled by
      2^-`:shifts`.
    * Equaliser without division: channel estimate at the pilots
      `Y * conj(P) / |P|^2` (constants), linear interpolation between
      pilots, then `z = Y * conj(h)`, which is `|h|^2` times the
      equalised point. Hard decisions compare `z` with the decision
      boundaries times `|h|^2`; for QPSK only the signs count.
    * Cell and bit deinterleaving, bit unpacking: as `FrontEnd`.

  Only square constellations. All index tables for a chunk size are
  computed on the host (`prepare/3`) and passed as parameters, so the graph
  has only integer elementwise operations, views and one-dimensional
  `take`s: nx_simd generates native code for it, and EXLA and the Nx
  evaluator run it bit-identically.
  """
  import Nx.Defn
  import Bitwise
  alias ExSdr.Comm.OFDM
  alias ExSdr.DSP.FFT
  alias NxFixed.{Fixed, Num}
  alias NxFixed.Primitives, as: P

  @doc "Default formats: data q(16, 12), coefficients and NCO q(16, 15) with a 4096-entry sine table."
  def default_opts,
    do: [
      format: Num.q(16, 12),
      coef: Num.q(16, 15, lut_bits: 12),
      shifts: 5
    ]

  @doc """
  The front end for exactly `symbols` symbols (a multiple of
  `ExSdr.Comm.OFDM.FrontEnd.group/1`). Options: `:format`, `:coef`,
  `:shifts` (see `default_opts/0`), `:compiler` (`NxSimd.Compiler`,
  `EXLA` or `Nx.Defn.Evaluator`; default: nx_simd when available for
  fixed point, else EXLA, else the evaluator); other options go to the
  compiler.
  """
  def prepare(spec, symbols, opts \\ []) do
    opts = Keyword.merge(default_opts(), opts)
    # the float reference has float coefficients too
    opts =
      if opts[:format].kind == :float, do: Keyword.put(opts, :coef, opts[:format]), else: opts

    o = spec.ofdm
    sym = OFDM.symbol_length(o)
    group = OFDM.FrontEnd.group(spec)

    unless symbols > 0 and rem(symbols, group) == 0,
      do: raise(ArgumentError, "symbols must be a positive multiple of #{group}")

    unless spec.constellation.square,
      do: raise(ArgumentError, "the fixed-point front end needs a square constellation")

    t = tables(spec, symbols, opts)

    static = [
      fft: o.fft,
      cp: o.cp,
      period: OFDM.period(o),
      constellation: spec.constellation,
      cells?: spec.cells != nil,
      bits?: spec.bits != nil,
      format: opts[:format],
      coef: opts[:coef],
      shifts: opts[:shifts]
    ]

    templates = [
      Nx.template({symbols * sym * 2}, :s16),
      Nx.template({}, :s64),
      Nx.template({}, :s64),
      Nx.to_template(t)
    ]

    compiler = Keyword.get_lazy(opts, :compiler, fn -> default_compiler(opts[:format]) end)
    # the remaining options go to the compiler (e.g. nx_simd's `fallback: false`)
    compiler_opts = Keyword.drop(opts, [:format, :coef, :shifts, :compiler])

    fun =
      Nx.Defn.compile(&front_end/5, templates ++ [static], [compiler: compiler] ++ compiler_opts)

    %{fun: fun, tables: t, samples: symbols * sym, fft: o.fft}
  end

  defp default_compiler(%Num.Format{kind: :fixed}) do
    cond do
      Code.ensure_loaded?(NxSimd) and NxSimd.available?() -> NxSimd.Compiler
      Code.ensure_loaded?(EXLA) -> EXLA
      true -> Nx.Defn.Evaluator
    end
  end

  defp default_compiler(_), do: if(Code.ensure_loaded?(EXLA), do: EXLA, else: Nx.Defn.Evaluator)

  @doc "Phase increment (32-bit phase word per sample) for a frequency offset in subcarrier spacings."
  def phase_increment(freq_offset, fft),
    do: round(-freq_offset / fft * 4_294_967_296) &&& 0xFFFFFFFF

  @doc """
  Runs a prepared front end on `sc16` samples (binary or `s16` tensor of
  interleaved I/Q, exactly the prepared length) that start `start` samples
  into the stream. Returns the code bits as a binary of 0/1 bytes.
  """
  def run(%{fun: f, tables: t, samples: n, fft: fft}, samples, freq_offset, start \\ 0) do
    x = if is_binary(samples), do: Nx.from_binary(samples, :s16), else: samples
    ^n = div(Nx.size(x), 2)
    inc = phase_increment(freq_offset, fft)
    phase0 = inc * start &&& 0xFFFFFFFF
    f.(x, Nx.tensor(phase0, type: :s64), Nx.tensor(inc, type: :s64), t) |> Nx.to_binary()
  end

  @doc "Complex float samples to `sc16` (s16 interleaved), scaled by `gain` and saturated."
  def to_sc16(x, gain) do
    [Nx.real(x), Nx.imag(x)]
    |> Nx.stack(axis: 1)
    |> Nx.multiply(gain * 32768)
    |> Nx.round()
    |> Nx.clip(-32768, 32767)
    |> Nx.as_type(:s16)
    |> Nx.flatten()
  end

  # -- host tables ------------------------------------------------------------------------

  defp tables(spec, symbols, opts) do
    o = spec.ofdm
    ot = OFDM.tables(o)
    period = OFDM.period(o)
    groups = div(symbols, period)
    coef = opts[:coef]
    bins = List.to_tuple(o.bins)

    per_class =
      for p <- 0..(period - 1) do
        pilots = Nx.to_flat_list(ot.pilots[p])
        data = Nx.to_flat_list(ot.data[p])
        left = Nx.to_flat_list(ot.left[p]) |> List.to_tuple()
        weight = Nx.to_flat_list(ot.weight[p]) |> List.to_tuple()
        np = length(pilots)

        # conj(P) / |P|^2 per pilot
        inv =
          case ot.values[p] |> Nx.type() do
            {:c, _} ->
              ot.values[p]
              |> Nx.to_flat_list()
              |> Enum.map(fn c ->
                {a, b} = {Complex.real(c), Complex.imag(c)}
                {a / (a * a + b * b), -b / (a * a + b * b)}
              end)

            _ ->
              ot.values[p] |> Nx.to_flat_list() |> Enum.map(&{1 / &1, 0.0})
          end

        rows = for g <- 0..(groups - 1), do: g * period + p
        abs = fn ks -> for r <- rows, k <- ks, do: r * o.fft + elem(bins, k) end

        %{
          pilot: idx(abs.(pilots), {groups, np}),
          data: idx(abs.(data), {groups, length(data)}),
          left:
            idx(
              for(g <- 0..(groups - 1), k <- data, do: g * np + elem(left, k)),
              {groups, length(data)}
            ),
          weight: coef_tensor(Enum.map(data, &elem(weight, &1)), coef),
          inv_re: coef_tensor(Enum.map(inv, &elem(&1, 0)), coef),
          inv_im: coef_tensor(Enum.map(inv, &elem(&1, 1)), coef)
        }
      end

    class = fn field -> per_class |> Enum.map(& &1[field]) |> List.to_tuple() end
    d = Nx.axis_size(ot.data[0], 0)

    cells =
      if spec.cells do
        tabs = spec.cells.tables |> List.to_tuple()
        count = tuple_size(tabs)

        for(r <- 0..(symbols - 1), k <- Tuple.to_list(elem(tabs, rem(r, count))), do: r * d + k)
        |> idx({symbols, d})
      else
        Nx.tensor(0)
      end

    bits =
      if spec.bits do
        [tab] = spec.bits.tables
        size = tuple_size(tab)
        blocks = div(symbols * d * spec.constellation.bits, size)

        for(b <- 0..(blocks - 1), k <- Tuple.to_list(tab), do: b * size + k)
        |> idx({blocks, size})
      else
        Nx.tensor(0)
      end

    %{
      sample: Nx.iota({symbols * OFDM.symbol_length(o)}, type: :s32, backend: Nx.BinaryBackend),
      pilot: class.(:pilot),
      data: class.(:data),
      left: class.(:left),
      weight: class.(:weight),
      inv_re: class.(:inv_re),
      inv_im: class.(:inv_im),
      cells: cells,
      bits: bits
    }
  end

  defp idx(list, shape),
    do: list |> Nx.tensor(type: :s32, backend: Nx.BinaryBackend) |> Nx.reshape(shape)

  defp coef_tensor(values, %Num.Format{kind: :fixed} = f),
    do: Nx.tensor(Num.quantize_constants(values, f), type: :s16, backend: Nx.BinaryBackend)

  defp coef_tensor(values, _), do: Nx.tensor(values, type: :f32, backend: Nx.BinaryBackend)

  # -- the computation ---------------------------------------------------------------------

  defn front_end(x, phase0, inc, t, opts \\ []) do
    opts =
      keyword!(opts, [
        :fft,
        :cp,
        :period,
        :constellation,
        :cells?,
        :bits?,
        :format,
        :coef,
        :shifts
      ])

    graph(x, phase0, inc, t, opts)
  end

  deftransformp graph(x, phase0, inc, t, opts) do
    f = opts[:format]
    coef = opts[:coef]
    fft = opts[:fft]
    sym = fft + opts[:cp]
    n = div(Nx.size(x), 2 * sym)

    # sc16 -> complex Q15, frequency correction
    pairs = Nx.reshape(x, {:auto, 2})
    input = {real_in(pairs, 0, f), real_in(pairs, 1, f)}

    # phase words are 32 bits; the masks and the clip bound the product
    # for the width check
    mask = Nx.tensor(0xFFFFFFFF, type: :s64, backend: Nx.BinaryBackend)
    samples = n * sym

    phase =
      t.sample
      |> Nx.clip(0, samples - 1)
      |> Nx.as_type(:s64)
      |> Nx.multiply(Nx.bitwise_and(inc, mask))
      |> Nx.add(Nx.bitwise_and(phase0, mask))
      |> Nx.bitwise_and(mask)

    mixed = input |> Num.cmul(nco(phase, coef)) |> Num.rescale(f) |> Num.quantize(f)

    # symbols without the cyclic prefix, FFT
    {re, im} =
      mixed
      |> map2(&(&1 |> Nx.reshape({n, sym}) |> Nx.slice_along_axis(opts[:cp], fft, axis: 1)))
      |> FFT.fft(format: f, twiddle: coef, shifts: opts[:shifts])

    flat = map2({re, im}, &Nx.flatten/1)

    words =
      for p <- 0..(opts[:period] - 1) do
        class_words(flat, t, p, f, coef, opts[:constellation])
      end
      |> Enum.map(&Nx.new_axis(&1, 1))
      |> Nx.concatenate(axis: 1)
      |> Nx.reshape({n, :auto})

    words = if opts[:cells?], do: take_in(Nx.flatten(words), t.cells), else: words
    code_bits(words, t.bits, opts[:constellation].bits, opts[:bits?])
  end

  defp real_in(pairs, column, f) do
    s = pairs |> Nx.slice_along_axis(column, 1, axis: 1) |> Nx.squeeze(axes: [1])

    case f.kind do
      :fixed -> %Fixed{t: s, frac: 15}
      :float -> Nx.divide(Nx.as_type(s, :f32), 32768)
    end
  end

  # exp(i * 2 pi * phase / 2^32)
  defp nco(phase, %Num.Format{kind: :fixed} = c) do
    {cos, sin} =
      phase
      |> Nx.right_shift(16)
      |> Nx.as_type(:u16)
      |> P.lut_sincos_q15(bits: c.lut_bits, amplitude: (1 <<< c.frac) - 1, round: true)

    {%Fixed{t: cos, frac: c.frac}, %Fixed{t: sin, frac: c.frac}}
  end

  defp nco(phase, _) do
    angle = phase |> Nx.as_type(:f64) |> Nx.multiply(2 * :math.pi() / 4_294_967_296)
    {angle |> Nx.cos() |> Nx.as_type(:f32), angle |> Nx.sin() |> Nx.as_type(:f32)}
  end

  # hard decisions of the symbols of class p: {groups, data carriers} words
  defp class_words(flat, t, p, f, coef, constellation) do
    take = fn idx -> map2(flat, &take_in(&1, idx)) end
    inv = {coef_value(elem(t.inv_re, p), coef), coef_value(elem(t.inv_im, p), coef)}
    {groups, pilots} = Nx.shape(elem(t.pilot, p))

    # channel at the pilots, then interpolated at the data carriers
    h_pilots =
      take.(elem(t.pilot, p))
      |> Num.cmul(map2(inv, &Nx.reshape(&1, {1, pilots})))
      |> Num.rescale(f)
      |> Num.quantize(f)
      |> map2(&Nx.flatten/1)

    # left neighbour index 0..count-2, so left + 1 stays in the table
    l_idx = Nx.clip(elem(t.left, p), 0, groups * pilots - 2)
    left = map2(h_pilots, &Nx.take(&1, l_idx))
    right = map2(h_pilots, &Nx.take(&1, Nx.add(l_idx, 1)))
    w = coef_value(elem(t.weight, p), coef) |> view(&Nx.reshape(&1, {1, :auto}))

    interp = fn l, r ->
      l |> Num.add(Num.rescale(Num.mul(Num.sub(r, l), w), f)) |> Num.quantize(f)
    end

    {hr, hi} = {interp.(elem(left, 0), elem(right, 0)), interp.(elem(left, 1), elem(right, 1))}

    # z = Y conj(h) = |h|^2 * equalised point
    {yr, yi} = take.(elem(t.data, p))
    zr = Num.add(Num.mul(yr, hr), Num.mul(yi, hi))
    zi = Num.sub(Num.mul(yi, hr), Num.mul(yr, hi))
    energy = fn -> Num.add(Num.mul(hr, hr), Num.mul(hi, hi)) end

    sq = constellation.square
    i = axis_index(zr, energy, sq)
    q = axis_index(zi, energy, sq)
    words = Nx.bitwise_or(Nx.take(u8(sq.i_words), i), Nx.take(u8(sq.q_words), q))
    ^groups = Nx.axis_size(words, 0)
    words
  end

  # level index 0..L-1: the number of boundaries b (-(L-2), ..., L-2) with
  # a * scale >= b * |h|^2 (ties to the larger level, as Constellation)
  defp axis_index(a, energy, %{levels: l, scale: scale}) do
    bounds = Enum.to_list(-(l - 2)..(l - 2)//2)

    if bounds == [0] do
      Nx.greater_equal(value(a), 0) |> Nx.as_type(:s32)
    else
      {lhs, e, k} = scaled(a, energy.(), scale)

      bounds
      |> Enum.map(&Nx.greater_equal(lhs, Nx.multiply(e, &1 * k)))
      |> Enum.map(&Nx.as_type(&1, :s32))
      |> Enum.reduce(&Nx.add/2)
    end
  end

  # a * scale and |h|^2 on a common integer scale (scale as a 2^10 fraction)
  defp scaled(%Fixed{t: a}, %Fixed{t: e}, scale),
    do: {Nx.multiply(Nx.as_type(a, :s64), round(scale * 1024)), Nx.as_type(e, :s64), 1024}

  defp scaled(a, e, scale), do: {Nx.multiply(a, scale), e, 1}

  defp code_bits(words, t, v, apply?) do
    shifts = Nx.tensor(Enum.to_list((v - 1)..0//-1), type: :u8, backend: Nx.BinaryBackend)

    bits =
      words
      |> Nx.flatten()
      |> Nx.new_axis(1)
      |> Nx.right_shift(Nx.new_axis(shifts, 0))
      |> Nx.bitwise_and(1)
      |> Nx.as_type(:u8)
      |> Nx.flatten()

    if apply?, do: take_in(bits, Nx.flatten(t)), else: bits
  end

  # gather with the index clipped to the table (a no-op for the host tables;
  # it bounds parameter indices for the width check, like a hardware counter)
  defp take_in(table, idx), do: Nx.take(table, Nx.clip(idx, 0, Nx.size(table) - 1))

  defp u8(list), do: Nx.tensor(list, type: :u8, backend: Nx.BinaryBackend)

  defp coef_value(t, %Num.Format{kind: :fixed} = c), do: %Fixed{t: t, frac: c.frac}
  defp coef_value(t, _), do: t

  defp map2({a, b}, fun), do: {view(a, fun), view(b, fun)}

  defp value(%Fixed{t: t}), do: t
  defp value(t), do: t

  defp view(%Fixed{} = x, fun), do: %{x | t: fun.(x.t)}
  defp view(t, fun), do: fun.(t)
end
