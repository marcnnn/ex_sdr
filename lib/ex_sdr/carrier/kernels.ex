defmodule ExSdr.Carrier.Kernels do
  @moduledoc """
  Stream kernels of the carrier recovery loops: `step(sample, state, opts) ->
  {output, next_state}`, written once against `NxFixed.Num` and the angle
  word helpers of `ExSdr.Analog.Math`. `opts[:format]` gives the float
  reference (`NxFixed.Num.float/0`) or a fixed-point instance.

  Phases and frequencies are 32 bit words (a full turn is 2^32). Complex
  samples are `{re, im}`. Constants (gains, taps, constellation points) are
  options, so they are compile time constants.

  The equalizers are in `ExSdr.Carrier.Equalizer`, the estimators in
  `ExSdr.Carrier.Estimators`.
  """
  import ExSdr.Analog.Math

  alias ExSdr.Carrier.Loop
  alias ExSdr.Digital.Kernels, as: Digital
  alias NxFixed.Fixed
  alias NxFixed.Num
  alias NxFixed.Num.Format

  @doc false
  # A state value or input is inside its format by construction; saying so
  # keeps the width check from assuming the whole container type.
  def bound(x, format), do: quantize(x, format)

  # -- Costas loop ------------------------------------------------------------------

  @doc """
  `costas_loop_cc`: `y = x * exp(-j phase)`, the error is the phase detector
  of `:order` (2, 4 or 8) on `y`, clipped to +-1, then the control loop
  (`ExSdr.Carrier.Loop`). State `{phase, freq}` (s32 words).

  Options: `:format`, `:order`, `:loop_bw`, `:max_freq` (1.0), `:min_freq`
  (-1.0) in radians per sample, `:outputs`: a list of extra outputs from
  `:freq`, `:phase` (words, after the update) and `:error` (the clipped
  error, a real); with extras the output is `{y, {extras...}}`.
  """
  def costas(x, {phase, freq} = state, opts) do
    format = opts[:format]
    {xr, xi} = bound(x, format)
    {c, s} = sincos_word(phase, format)

    y =
      quantize(
        {Num.add(Num.mul(xr, c), Num.mul(xi, s)), Num.sub(Num.mul(xi, c), Num.mul(xr, s))},
        format
      )

    err = y |> phase_detector(opts[:order], format) |> clip_error(format)
    {phase1, freq1} = Loop.update_real(err, state, opts, format)
    _ = freq
    {with_extras(y, opts, phase1, freq1, err), {phase1, freq1}}
  end

  defp with_extras(y, opts, phase, freq, err) do
    case Keyword.get(opts, :outputs, []) do
      [] ->
        y

      names ->
        extras =
          for name <- names do
            case name do
              :freq -> freq
              :phase -> phase
              :error -> err
            end
          end

        {y, List.to_tuple(extras)}
    end
  end

  # The phase detectors of costas_loop_cc_impl.cc, on the rotated sample.
  defp phase_detector({re, im}, 2, _f), do: Num.mul(re, im)

  defp phase_detector({re, im}, 4, f) do
    Num.sub(sel(positive?(re), im, neg(im, f)), sel(positive?(im), re, neg(re, f)))
  end

  defp phase_detector({re, im}, 8, f) do
    k = wconst(:math.sqrt(2.0) - 1.0, f)
    re_k = Num.mul(re, k)
    im_k = Num.mul(im, k)
    # costas_loop_cc_impl.h: |re| >= |im|: sign(re) im - sign(im) K re,
    # else sign(re) K im - sign(im) re
    near_re = Num.sub(sel(positive?(re), im, neg(im, f)), sel(positive?(im), re_k, neg(re_k, f)))
    near_im = Num.sub(sel(positive?(re), im_k, neg(im_k, f)), sel(positive?(im), re, neg(re, f)))
    sel(Nx.logical_not(lt(rmag(re), rmag(im))), near_re, near_im)
  end

  defp neg(x, f), do: Num.sub(Num.zeros({}, f), x)

  # GNU Radio's branchless_clip(x, 1): 0.5 (|x + 1| - |x - 1|), in f32 for
  # the float instance; a saturation to +-1 at 30 fractional bits for fixed point.
  defp clip_error(e, %Format{kind: :float}),
    do: Nx.multiply(Nx.subtract(Nx.abs(Nx.add(e, 1.0)), Nx.abs(Nx.subtract(e, 1.0))), 0.5)

  defp clip_error(e, %Format{kind: :fixed} = f),
    do: e |> quantize(wide_format(f, 32, 30)) |> clamp(-1.0, 1.0, 32)

  # -- band-edge FLL -------------------------------------------------------------------

  @doc """
  `fll_band_edge_cc`. The input first goes through a delay of
  `:filter_size` samples (GNU Radio's history makes its output lag by that),
  is rotated by `exp(+j phase)`, and the rotated stream is filtered by the
  two band-edge filters (`ExSdr.Carrier.fll_taps/3`, complex constant
  taps); the error is `|upper filter|^2 - |lower filter|^2` (GNU Radio's
  naming swaps them), fed to the control loop.

  State `{phase, freq, delay_line, filter_line}`: lines are `{re, im}`
  vectors, oldest first, of `filter_size` and `filter_size - 1` items.
  Options: `:format`, `:lower`, `:upper` (taps from `fll_taps/3`), `:tap_frac`
  (fractional bits of the taps, default 20), `:loop_bw`, `:max_freq`,
  `:min_freq` (default `+-2 pi 2 / sps`), `:outputs` as for `costas/3`.
  """
  def fll(x, {phase, freq, dline, fline}, opts) do
    format = opts[:format]
    n = length(opts[:lower])
    tf = Keyword.get(opts, :tap_frac, 20)
    x = bound(x, format)

    delayed = cmap(dline, &first/1)
    next_dline = cmap2(dline, x, &shift_in/2)

    {c, s} = sincos_word(phase, format)
    {dr, di} = delayed

    out =
      quantize(
        {Num.sub(Num.mul(dr, c), Num.mul(di, s)), Num.add(Num.mul(dr, s), Num.mul(di, c))},
        format
      )

    win = cmap2(fline, out, &append/2)
    nf = norm_format(format)

    out_upper = band_filter(win, opts[:lower], n, format, tf, nf)
    out_lower = band_filter(win, opts[:upper], n, format, tf, nf)

    err = Num.sub(power(out_lower), power(out_upper))
    err = if format.kind == :fixed, do: quantize(err, wide_format(format, 32, 28)), else: err

    {phase1, freq1} = Loop.update_real(err, {phase, freq}, opts, format)
    next_fline = cmap(win, &drop_first/1)
    y = out
    {with_extras(y, opts, phase1, freq1, err), {phase1, freq1, next_dline, next_fline}}
  end

  # filter outputs: 28 bits with 24 fractional (room for |y| < 8)
  defp norm_format(%Format{kind: :float} = f), do: f
  defp norm_format(f), do: wide_format(f, 28, 24)

  defp band_filter(win, taps, n, format, tf, nf) do
    acc = ExSdr.Filters.Kernels.apply_taps(win, taps, n - 1, 1, 1, format, tf)
    quantize(cmap(acc, &scalar/1), nf)
  end

  defp power({re, im}), do: Num.add(Num.mul(re, re), Num.mul(im, im))

  @doc """
  The band-edge filter taps of `fll_band_edge_cc` (its `design_filter`):
  `{lower, upper}`, lists of `{re, im}` in convolution order (index 0 meets
  the newest sample), `upper` the conjugate of `lower`. The names are
  those of GNU Radio's members `d_taps_lower` and `d_taps_upper`.
  """
  def fll_taps(samples_per_symbol, rolloff, filter_size) do
    # harris' band-edge filter: the derivative of a raised cosine spectrum
    # is a half sine over the roll-off band, whose impulse response is two
    # sincs half a sample apart (time scaled by the roll-off). Modulated to
    # the lower band edge -(1 + rolloff) / (2 sps) cycles per sample.
    sps = samples_per_symbol * 1.0
    span = round_half_even(filter_size / sps)
    edge = -(1 + rolloff) / (2 * sps)
    centre = div(filter_size - 1, 2)

    proto =
      for n <- 0..(filter_size - 1) do
        u = rolloff * (2.0 * n / sps - span)
        sinc(u + 0.5) + sinc(u - 0.5)
      end

    energy = proto |> Enum.map(&(&1 * &1)) |> Enum.sum()

    lower =
      proto
      |> Enum.with_index(fn p, n ->
        a = p / energy
        phi = 2 * :math.pi() * edge * (n - centre)
        {a * :math.cos(phi), a * :math.sin(phi)}
      end)
      |> Enum.reverse()

    {lower, Enum.map(lower, fn {re, im} -> {re, -im} end)}
  end

  defp sinc(x) when x == 0, do: 1.0
  defp sinc(x), do: :math.sin(:math.pi() * x) / (:math.pi() * x)

  defp round_half_even(x) do
    f = Float.floor(x)
    d = x - f

    cond do
      d < 0.5 -> trunc(f)
      d > 0.5 -> trunc(f) + 1
      rem(trunc(f), 2) == 0 -> trunc(f)
      true -> trunc(f) + 1
    end
  end

  # -- constellation receiver --------------------------------------------------------------

  @doc """
  The phase loop of `constellation_receiver_cb`: `y = x * exp(+j phase)`,
  the decision is the index of the nearest of `:points`, the error is
  `-arg(y * conj(point))` (an angle word), fed to the control loop.
  Output: the decision (`u8`). State `{phase, freq}`.

  Options: `:format`, `:points` (`{re, im}` floats), `:loop_bw`, `:max_freq`,
  `:min_freq` (GNU Radio's `fmax`, `fmin`), `:iterations`, `:guard` (CORDIC),
  `:outputs`: extras from `:error` (word), `:phase`, `:freq` (words) and
  `:symbol` (the rotated sample); with extras the output is
  `{decision, {extras...}}`.
  """
  def receiver(x, {phase, freq} = state, opts) do
    format = opts[:format]
    {xr, xi} = bound(x, format)
    {c, s} = sincos_word(phase, format)

    y =
      quantize(
        {Num.sub(Num.mul(xr, c), Num.mul(xi, s)), Num.add(Num.mul(xr, s), Num.mul(xi, c))},
        format
      )

    {idx, _} =
      Digital.decide(y, Nx.tensor(0, type: :u8), format: format, points: opts[:points])

    {yr, yi} = y
    points = opts[:points]
    pr = Digital.lookup(idx, Enum.map(points, &elem(&1, 0)), format)
    pi = Digital.lookup(idx, Enum.map(points, &elem(&1, 1)), format)

    prod_re = Num.add(Num.mul(yr, pr), Num.mul(yi, pi))
    prod_im = Num.sub(Num.mul(yi, pr), Num.mul(yr, pi))
    cordic = Keyword.take(opts, [:iterations, :guard])
    err = Nx.negate(atan2_word(prod_im, prod_re, format, cordic))

    {phase1, freq1} = Loop.update_words(err, state, opts)
    _ = {phase, freq}

    out =
      case Keyword.get(opts, :outputs, []) do
        [] ->
          idx

        names ->
          extras =
            for name <- names do
              case name do
                :error -> err
                :phase -> phase1
                :freq -> freq1
                :symbol -> y
              end
            end

          {idx, List.to_tuple(extras)}
      end

    {out, {phase1, freq1}}
  end

  # -- vector helpers (complex or real, either instance), shared by the other carrier modules ------

  @doc false
  def cmap({a, b}, fun), do: {fun.(a), fun.(b)}
  @doc false
  def cmap(a, fun), do: fun.(a)

  @doc false
  def cmap2({a, b}, {c, d}, fun), do: {fun.(a, c), fun.(b, d)}
  @doc false
  def cmap2(a, c, fun), do: fun.(a, c)

  @doc false
  def first(%Fixed{t: t, frac: frac}),
    do: %Fixed{t: t |> Nx.slice([0], [1]) |> Nx.reshape({}), frac: frac}

  def first(t), do: t |> Nx.slice([0], [1]) |> Nx.reshape({})

  @doc false
  def scalar(%Fixed{t: t, frac: frac}), do: %Fixed{t: Nx.reshape(t, {}), frac: frac}
  def scalar(t), do: Nx.reshape(t, {})

  @doc false
  def drop_first(%Fixed{t: t, frac: frac}),
    do: %Fixed{t: Nx.slice(t, [1], [Nx.axis_size(t, 0) - 1]), frac: frac}

  def drop_first(t), do: Nx.slice(t, [1], [Nx.axis_size(t, 0) - 1])

  # line (vector) followed by one item
  @doc false
  def append(%Fixed{t: t, frac: frac}, %Fixed{t: x}),
    do: %Fixed{t: Nx.concatenate([t, Nx.reshape(x, {1})]), frac: frac}

  def append(t, x), do: Nx.concatenate([t, Nx.reshape(x, {1})])

  @doc false
  def shift_in(line, x), do: line |> append(x) |> drop_first()
end
