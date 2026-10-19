defmodule ExSdr.Filters.Kernels do
  @moduledoc """
  Filter kernels written once against `NxFixed.Num`: `:format` selects the
  float reference (`NxFixed.Num.float/0`) or a fixed-point instance
  (`NxFixed.Num.q/3`).

  ## Signals

  A real signal is a tensor (float) or `NxFixed.Fixed`; a complex signal is
  a `{re, im}` tuple of those. Frame kernels take the item index as first
  axis; stream kernels take one item (a scalar).

  ## Taps

  `:taps` are compile-time constants: Elixir floats, or `{re, im}` tuples for
  complex taps (a list with at least one tuple is complex). With a
  fixed-point format they are rounded to `:tap_frac` fractional bits
  (default: the format's) and are **not** saturated: a tap of 1.9 needs one
  integer bit in the multiplier, which is what the hardware would use. The
  accumulator is exact (`NxFixed.Num` rule: binary points add); one rounding
  and saturation to the format ends each output, like `ExSdr.DSP.Blocks.fir_complex`.

  ## State

  Delay lines hold the last `length(taps) - 1` inputs, oldest first (one
  line per component for complex input). Tap lists are padded with a zero
  to at least two entries (`pad_taps/1`) so that a line is never empty.
  Splitting an input stream into pieces and carrying the state gives the
  same output as one call.

  All frame sizes are known when tracing; the unrolled graph has one slice
  per non-zero tap, so very long filters (thousands of taps) trace slowly.
  """
  import Nx.Defn
  import Bitwise

  alias NxFixed.Fixed
  alias NxFixed.Num
  alias NxFixed.Num.Format
  alias NxFixed.Primitives, as: P

  # -- public kernels ----------------------------------------------------------------

  @doc """
  FIR filter, one item per call: `y[n] = sum(taps[k] * x[n-k])`. Options:
  `:format`, `:taps`, `:tap_frac`. State: delay line of `length(pad_taps(taps)) - 1`.
  """
  defn fir_step(x, line, opts \\ []) do
    opts = keyword!(opts, [:format, :taps, :tap_frac])
    fir_step_impl(x, line, opts)
  end

  @doc """
  FIR filter with decimation on a frame: output `k` is `y[k * decimation]`
  of the full-rate filter, counted from the start of the stream (the first
  output uses the first input). Options: `:format`, `:taps`, `:tap_frac`,
  `:decimation` (default 1). The frame length is a multiple of `:decimation`.
  """
  defn fir_frame(x, line, opts \\ []) do
    opts = keyword!(opts, [:format, :taps, :tap_frac, decimation: 1])
    fir_frame_impl(x, line, opts)
  end

  @doc """
  Interpolating FIR (`interp_fir_filter`): zero-stuffing by `:interpolation`
  followed by the filter, computed per polyphase branch. Each input gives
  `:interpolation` outputs. Options: `:format`, `:taps`, `:tap_frac`, `:interpolation`.
  """
  defn interp_frame(x, line, opts \\ []) do
    opts = keyword!(opts, [:format, :taps, :tap_frac, :interpolation])
    interp_frame_impl(x, line, opts)
  end

  @doc """
  Rational resampler on a frame of `n` inputs (a multiple of `:decimation`):
  `n * interpolation / decimation` outputs. Output `m` is the full-rate
  (zero-stuffed by `:interpolation`) filter output at position
  `:phase + m * decimation`, counted from the start of the stream.
  Options: `:format`, `:taps`, `:tap_frac`, `:interpolation`, `:decimation`,
  `:phase` (default 0, below `:decimation`).
  """
  defn resample_frame(x, line, opts \\ []) do
    opts = keyword!(opts, [:format, :taps, :tap_frac, :interpolation, :decimation, phase: 0])
    resample_frame_impl(x, line, opts)
  end

  @doc """
  Decimating FIR on complex (or real) input followed by a rotation by an NCO
  that advances `:phase_inc` (a u16 phase word, 65536 = one turn) per
  *output*, as in GNU Radio's frequency translating FIR. `state` is
  `{line, phase}`. The NCO is `NxFixed.Num.oscillator/3`'s: exact in the float
  instance, a `lut_bits` table in fixed point. Options as `fir_frame/3` plus
  `:phase_inc`.
  """
  defn fir_rotate_frame(x, state, opts \\ []) do
    opts = keyword!(opts, [:format, :taps, :tap_frac, :phase_inc, decimation: 1])
    fir_rotate_impl(x, state, opts)
  end

  @doc """
  Direct-form IIR, one item per call:
  `y[n] = sum(ff[k] * x[n-k]) + sum(fb[k-1] * y[n-k], k = 1..)`, i.e. `:fb`
  holds the coefficients that are *added* (GNU Radio's old style; for the new
  style pass the negated coefficients 1..). State: `{x_line, y_line}` of
  `length(ff) - 1` and `length(fb)` items (see `pad_iir/2`). The output is
  rounded and saturated to the format each step, so the feedback uses the
  quantised output. Options: `:format`, `:ff`, `:fb`, `:tap_frac`.
  """
  defn iir_step(x, state, opts \\ []) do
    opts = keyword!(opts, [:format, :ff, :fb, :tap_frac])
    iir_step_impl(x, state, opts)
  end

  # -- helpers for building blocks -----------------------------------------------------

  @doc "Taps padded with a zero to at least two entries."
  def pad_taps([t]), do: [t, zero_like(t)]
  def pad_taps(taps) when length(taps) >= 2, do: taps

  @doc "`{ff, fb}` padded so that the x line and the y line are both non-empty."
  def pad_iir(ff, fb) do
    {pad_taps(ff), if(fb == [], do: [0.0], else: fb)}
  end

  @doc "Polyphase branches `[[taps[r + j*i] for j]]` for r in 0..i-1, padded to equal length >= 2."
  def polyphase(taps, i) do
    k = max(div(length(taps) + i - 1, i), 2)
    padded = taps ++ List.duplicate(zero_like(hd(taps)), k * i - length(taps))
    for r <- 0..(i - 1), do: for(j <- 0..(k - 1), do: Enum.at(padded, r + j * i))
  end

  defp zero_like({_, _}), do: {0.0, 0.0}
  defp zero_like(_), do: 0.0

  @doc "Whether a tap list is complex."
  def complex_taps?(taps), do: Enum.any?(taps, &is_tuple/1)

  # -- implementations -----------------------------------------------------------------

  deftransformp fir_step_impl(x, line, opts) do
    taps = pad_taps(opts[:taps])
    {y, next} = fir_core(cmap(x, &add_axis/1), line, taps, 1, opts[:format], tap_frac(opts))
    {cmap(y, &drop_axis/1), next}
  end

  deftransformp fir_frame_impl(x, line, opts) do
    taps = pad_taps(opts[:taps])
    fir_core(x, line, taps, opts[:decimation], opts[:format], tap_frac(opts))
  end

  deftransformp interp_frame_impl(x, line, opts) do
    f = opts[:format]
    i = opts[:interpolation]
    branches = polyphase(opts[:taps], i)
    history = length(hd(branches)) - 1
    n = size(x)
    {win, next} = window(x, line)

    outs = for b <- branches, do: apply_taps(win, b, history, n, 1, f, tap_frac(opts))
    {outs |> interleave(n) |> Num.quantize(f), next}
  end

  deftransformp resample_frame_impl(x, line, opts) do
    f = opts[:format]
    i = opts[:interpolation]
    d = opts[:decimation]
    phase = opts[:phase]
    n = size(x)

    if rem(n, d) != 0 or phase < 0 or phase >= d,
      do: raise(ArgumentError, "resampler: frame of #{n} items, decimation #{d}, phase #{phase}")

    branches = List.to_tuple(polyphase(opts[:taps], i))
    history = length(elem(branches, 0)) - 1
    {win, next} = window(x, line)

    rows =
      for m <- 0..(div(n, d) * i - 1) do
        pos = phase + m * d

        apply_taps(
          win,
          elem(branches, rem(pos, i)),
          history + div(pos, i),
          1,
          1,
          f,
          tap_frac(opts)
        )
      end

    {rows |> concat_signals() |> Num.quantize(f), next}
  end

  deftransformp fir_rotate_impl(x, {line, phase}, opts) do
    f = opts[:format]
    taps = pad_taps(opts[:taps])
    {y, next_line} = fir_core(x, line, taps, opts[:decimation], f, tap_frac(opts))
    y = if is_tuple(y), do: y, else: {y, Num.zeros({size(y)}, f)}
    {nco, next_phase} = nco_frame(phase, opts[:phase_inc], size(elem(y, 0)), f)
    {y |> Num.cmul(nco) |> Num.rescale(f) |> Num.quantize(f), {next_line, next_phase}}
  end

  deftransformp iir_step_impl(x, {x_line, y_line}, opts) do
    f = opts[:format]
    tf = tap_frac(opts)
    {ff, fb} = pad_iir(opts[:ff], opts[:fb])

    {win, next_x} = window(cmap(x, &add_axis/1), x_line)
    from_x = apply_taps(win, ff, length(ff) - 1, 1, 1, f, tf)
    from_y = apply_taps(y_line, fb, length(fb) - 1, 1, 1, f, tf)
    y = zip(from_x, from_y, &Num.add/2) |> Num.quantize(f)

    m = length(fb)
    next_y = zip(y_line, y, &cat/2) |> cmap(&slice(&1, 1, m))
    {cmap(y, &drop_axis/1), {next_x, next_y}}
  end

  # -- FIR core ---------------------------------------------------------------------------

  defp fir_core(x, line, taps, decimation, f, tf) do
    n = size(x)

    if rem(n, decimation) != 0,
      do:
        raise(ArgumentError, "frame of #{n} items is not a multiple of decimation #{decimation}")

    {win, next} = window(x, line)
    y = apply_taps(win, taps, length(taps) - 1, div(n, decimation), decimation, f, tf)
    {Num.quantize(y, f), next}
  end

  # The line followed by the new items, and the last `length(line)` of those.
  defp window(x, line) do
    win = zip(line, x, &cat/2)
    keep = size(line)
    {win, cmap(win, &slice(&1, size(win) - keep, keep))}
  end

  @doc false
  # `sum_j taps[j] * win[start + k * stride - j]` for k in 0..nout-1, as exact
  # accumulators (not yet quantised). Real or complex window, real or complex taps.
  def apply_taps(win, taps, start, nout, stride, f, tf) do
    case {is_tuple(win), complex_taps?(taps)} do
      {false, false} ->
        conv(win, taps, start, nout, stride, f, tf)

      {true, false} ->
        cmap(win, &conv(&1, taps, start, nout, stride, f, tf))

      {false, true} ->
        {conv(win, re(taps), start, nout, stride, f, tf),
         conv(win, im(taps), start, nout, stride, f, tf)}

      {true, true} ->
        {wr, wi} = win
        rr = conv(wr, re(taps), start, nout, stride, f, tf)
        ii = conv(wi, im(taps), start, nout, stride, f, tf)
        ri = conv(wr, im(taps), start, nout, stride, f, tf)
        ir = conv(wi, re(taps), start, nout, stride, f, tf)
        {Num.sub(rr, ii), Num.add(ri, ir)}
    end
  end

  defp re(taps),
    do:
      Enum.map(taps, fn
        {r, _} -> r
        r -> r
      end)

  defp im(taps),
    do:
      Enum.map(taps, fn
        {_, i} -> i
        _ -> 0.0
      end)

  defp conv(win, taps, start, nout, stride, f, tf) do
    {data, frac} = {raw(win), frac_of(win)}

    {kept, tap_values} =
      case f.kind do
        :float ->
          kept = taps |> Enum.with_index() |> Enum.reject(fn {c, _} -> c == 0 end)
          {kept, Enum.map(kept, fn {c, _} -> c * 1.0 end)}

        :fixed ->
          q = taps |> Enum.map(&round(&1 * (1 <<< tf))) |> Enum.with_index()
          kept = Enum.reject(q, fn {c, _} -> c == 0 end)
          {kept, Enum.map(kept, fn {c, _} -> c end)}
      end

    case {f.kind, kept} do
      {:float, []} ->
        Nx.broadcast(Nx.tensor(0.0, type: :f32), {nout})

      {:fixed, []} ->
        %Fixed{t: Nx.broadcast(Nx.tensor(0, type: :s32), {nout}), frac: frac + tf}

      {kind, _} ->
        c =
          if kind == :float do
            :f32
          else
            bound = (1 <<< (f.bits - 1)) * (tap_values |> Enum.map(&abs/1) |> Enum.sum())
            if bound < 1 <<< 31, do: :s32, else: :s64
          end

        columns =
          for {_, j} <- kept do
            data
            |> Nx.slice([start - j], [(nout - 1) * stride + 1], strides: [stride])
            |> Nx.reshape({nout, 1})
          end

        matrix = columns |> Nx.concatenate(axis: 1) |> Nx.as_type(c)
        coefficients = Nx.tensor(tap_values, type: c, backend: Nx.BinaryBackend)
        acc = Nx.dot(matrix, coefficients)
        if kind == :float, do: acc, else: %Fixed{t: acc, frac: frac + tf}
    end
  end

  # -- NCO over a frame ---------------------------------------------------------------

  defp nco_frame(phase, inc, n, %Format{} = f) do
    inc = band(inc, 0xFFFF)
    steps = Nx.tensor(Enum.to_list(1..n), type: :s64, backend: Nx.BinaryBackend)
    base = Nx.as_type(phase, :s64)
    phases = base |> Nx.add(Nx.multiply(steps, inc)) |> P.wrap({:u, 16})
    next = base |> Nx.add(n * inc) |> P.wrap({:u, 16})

    nco =
      case f.kind do
        :float ->
          angle = phases |> Nx.as_type(:f64) |> Nx.multiply(2 * :math.pi() / 65536)
          {angle |> Nx.cos() |> Nx.as_type(:f32), angle |> Nx.sin() |> Nx.as_type(:f32)}

        :fixed ->
          {c, s} =
            P.lut_sincos_q15(phases,
              bits: f.lut_bits,
              amplitude: (1 <<< f.frac) - 1,
              round: f.rounding == :nearest
            )

          {%Fixed{t: c, frac: f.frac}, %Fixed{t: s, frac: f.frac}}
      end

    {nco, next}
  end

  # -- signal plumbing -------------------------------------------------------------------

  defp tap_frac(opts), do: opts[:tap_frac] || opts[:format].frac || 0

  defp cmap(s, fun) when is_tuple(s), do: {fun.(elem(s, 0)), fun.(elem(s, 1))}
  defp cmap(s, fun), do: fun.(s)

  defp zip(a, b, fun) when is_tuple(a),
    do: {fun.(elem(a, 0), elem(b, 0)), fun.(elem(a, 1), elem(b, 1))}

  defp zip(a, b, fun), do: fun.(a, b)

  defp raw(%Fixed{t: t}), do: t
  defp raw(t), do: t
  defp like(%Fixed{frac: frac}, t), do: %Fixed{t: t, frac: frac}
  defp like(_, t), do: t
  defp frac_of(%Fixed{frac: frac}), do: frac
  defp frac_of(_), do: 0

  defp size(s) when is_tuple(s), do: size(elem(s, 0))
  defp size(v), do: v |> raw() |> Nx.axis_size(0)

  defp cat(a, b), do: like(a, Nx.concatenate([raw(a), Nx.as_type(raw(b), Nx.type(raw(a)))]))
  defp slice(v, start, n), do: like(v, Nx.slice(raw(v), [start], [n]))
  defp add_axis(v), do: like(v, Nx.reshape(raw(v), {1}))
  defp drop_axis(v), do: like(v, Nx.reshape(raw(v), {}))

  # Signals of shape {n}, one per branch, to one signal of shape {n * branches}
  # with the branches interleaved.
  defp interleave([first | _] = outs, n) when is_tuple(first) do
    {interleave(Enum.map(outs, &elem(&1, 0)), n), interleave(Enum.map(outs, &elem(&1, 1)), n)}
  end

  defp interleave([first | _] = outs, n) do
    t =
      outs
      |> Enum.map(&Nx.reshape(raw(&1), {n, 1}))
      |> Nx.concatenate(axis: 1)
      |> Nx.reshape({n * length(outs)})

    like(first, t)
  end

  defp concat_signals([first | _] = sigs) when is_tuple(first) do
    {concat_signals(Enum.map(sigs, &elem(&1, 0))), concat_signals(Enum.map(sigs, &elem(&1, 1)))}
  end

  defp concat_signals([first | _] = sigs),
    do: like(first, sigs |> Enum.map(&raw/1) |> Nx.concatenate())
end
