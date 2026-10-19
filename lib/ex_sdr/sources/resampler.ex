defmodule ExSdr.Sources.Resampler do
  @moduledoc """
  Fractional (timing offset) resampler: an 8 tap polyphase interpolator with
  129 phases, like GNU Radio's `mmse_resampler_cc`, driven by a phase
  accumulator.

  Output `n` interpolates the input at time `3 + n * ratio` (window of 8
  samples starting at `floor(n * ratio)`, centred between taps 3 and 4:
  GNU Radio's convention, so at ratio 1 the output is the input advanced by
  3 samples and the first 3 input samples never appear). `ratio` is the
  input step per output sample (`epsilon` in GNU Radio's `channel_model`;
  above 1 there are fewer output samples than input samples). The position is
  kept as an integer plus a 32 bit fraction, so it does not drift over any
  stream length, unlike a float accumulator. The fraction is rounded to 1/128
  to pick the filter, as GNU Radio does. GNU Radio holds `ratio` as float32;
  a ratio that is a float32 value gives the same positions here.

  Taps: least squares fits of `exp(-j w (3 + mu))` over `|w| < pi/2` (half
  the sample rate band), `mu = k / 128`, solved when the block is built
  (`taps/0`; DC gain within 3e-4 of 1). They agree with GNU Radio's table to 1.2e-5 (its table is
  rounded to about 5 digits), so outputs agree to about 1e-4 of full scale.
  Signals should occupy less than half the band (interpolation error grows
  towards the band edge).

  A call that produces no output returns `nil` instead of an empty pair (Nx
  has no empty tensors); `nil` is also accepted as input.

  The number of output items depends on the data length and the state:
  `rate` of the block is `{in, out}` reduced from `2^32 / step`. The state is
  `{buffer, position}`: fewer than 8 unconsumed input samples (complex, as a
  `{re, im}` pair, `nil` when none) and the fractional position within them, so input split
  anywhere gives identical output.
  """
  import Bitwise

  alias NxFixed.Block
  alias NxFixed.Fixed
  alias NxFixed.Num
  alias NxFixed.Num.Format

  @phases 128
  @ntaps 8

  @doc "The `129 x 8` interpolation taps as floats: `taps()[k]` is for `mu = k / 128`."
  def taps do
    a = :math.pi() / 2
    # normal equations of the least squares fit, real symmetric
    r =
      for i <- 0..(@ntaps - 1) do
        for j <- 0..(@ntaps - 1), do: sinc2(a, i - j)
      end

    rhs =
      for i <- 0..(@ntaps - 1) do
        for k <- 0..@phases, do: sinc2(a, i - 3 - k / @phases)
      end

    h =
      Nx.LinAlg.solve(
        Nx.tensor(r, type: :f64, backend: Nx.BinaryBackend),
        Nx.tensor(rhs, type: :f64, backend: Nx.BinaryBackend)
      )

    rows = h |> Nx.transpose() |> Nx.to_list()
    # the end points are exact delays (the solve leaves 1e-16 noise)
    first = List.duplicate(0.0, 8) |> List.replace_at(3, 1.0)
    last = List.duplicate(0.0, 8) |> List.replace_at(4, 1.0)
    [first] ++ Enum.slice(rows, 1, @phases - 1) ++ [last]
  end

  # integral of cos(w d) over [-a, a]
  defp sinc2(a, d) when d == 0, do: 2 * a
  defp sinc2(a, d), do: 2 * :math.sin(a * d) / d

  @doc """
  The block: options `:ratio` (input step per output, default 1.0),
  `:format`, `:phase` (initial fractional position in `[0, 1)`).
  """
  def block(opts) do
    f = Keyword.get(opts, :format, Num.float())
    ratio = Keyword.get(opts, :ratio, 1.0)
    true = ratio > 0.0 and ratio < 1 <<< 24
    step = round(ratio * 4_294_967_296)
    pos0 = round(Keyword.get(opts, :phase, 0.0) * 4_294_967_296)
    g = Integer.gcd(step, 4_294_967_296)

    %Block{
      name: :mmse_resampler_cc,
      kind: :host,
      fun: fn input, state -> process(input, state, step, f, taps_table(f)) end,
      init: {nil, pos0},
      rate: {div(step, g), div(4_294_967_296, g)},
      latency: 4,
      warmup: 8,
      item: {:complex, :complex},
      hardware: :feedback,
      gnuradio: "filter.mmse_resampler_cc"
    }
  end

  defp taps_table(%Format{kind: :float}), do: Nx.tensor(taps(), type: :f32)

  defp taps_table(%Format{kind: :fixed} = f),
    do: Nx.tensor(Enum.map(taps(), &Num.quantize_constants(&1, f)), type: :s32)

  @doc """
  Resamples `input` (a `{re, im}` pair of vectors, `Num.from_int/3` style)
  from `{buffer, position}`; returns `{output, next_state}`.
  """
  def process(nil, state, _step, _f, _taps), do: {nil, state}

  def process({re, im}, {buffer, pos}, step, f, taps) do
    {all_re, all_im} =
      case buffer do
        nil -> {re, im}
        {buf_re, buf_im} -> {concat(buf_re, re), concat(buf_im, im)}
      end

    len = all_re |> raw() |> Nx.axis_size(0)

    # outputs k = 0..count-1 sit at position pos + k * step (32 bit fraction)
    last = (len - @ntaps) <<< 32
    count = if last < pos, do: 0, else: div(last - pos, step) + 1
    next_pos = pos + count * step
    drop = next_pos >>> 32
    drop = min(drop, len)
    keep = fn v -> slice_from(v, drop, len - drop) end
    rest = if len - drop == 0, do: nil, else: {keep.(all_re), keep.(all_im)}
    state = {rest, next_pos - (drop <<< 32)}

    if count == 0 do
      {nil, state}
    else
      positions = Nx.add(Nx.multiply(Nx.iota({count}, type: :s64), c64(step)), c64(pos))
      ii = Nx.right_shift(positions, 32)

      imu =
        positions
        |> Nx.bitwise_and(c64(0xFFFFFFFF))
        |> Nx.add(c64(1 <<< 24))
        |> Nx.right_shift(25)

      idx = Nx.add(Nx.new_axis(ii, 1), Nx.iota({1, @ntaps}, type: :s64))
      rows = Nx.take(taps, imu)

      out_re = mac(all_re, all_im, idx, rows, f, :re)
      out_im = mac(all_re, all_im, idx, rows, f, :im)
      {{out_re, out_im}, state}
    end
  end

  # `re`/`im` of the output: real taps, so each part is a dot of its own input
  defp mac(re, im, idx, rows, f, part) do
    x = if part == :re, do: re, else: im

    case x do
      %Fixed{t: t, frac: frac} ->
        w = Nx.take(t, idx) |> Nx.as_type(:s64)
        acc = Nx.sum(Nx.multiply(w, Nx.as_type(rows, :s64)), axes: [1])
        Num.quantize(%Fixed{t: acc, frac: frac + f.frac}, f)

      t ->
        Nx.sum(Nx.multiply(Nx.take(t, idx), rows), axes: [1])
    end
  end

  defp c64(n), do: Nx.tensor(n, type: :s64)

  defp raw(%Fixed{t: t}), do: t
  defp raw(t), do: t

  defp concat(%Fixed{t: a, frac: frac}, %Fixed{t: b, frac: frac}),
    do: %Fixed{t: Nx.concatenate([Nx.as_type(a, Nx.type(b)), b]), frac: frac}

  defp concat(a, b), do: Nx.concatenate([a, b])

  defp slice_from(%Fixed{t: t, frac: frac}, start, len),
    do: %Fixed{t: Nx.slice(t, [start], [len]), frac: frac}

  defp slice_from(t, start, len), do: Nx.slice(t, [start], [len])
end
