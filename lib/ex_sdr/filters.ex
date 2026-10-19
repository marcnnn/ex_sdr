defmodule ExSdr.Filters do
  @moduledoc """
  The gr-filter blocks as `NxFixed.Block`s: `block(name, opts)`.

  Every block takes `:format` (`NxFixed.Num.float/0` by default, or
  `NxFixed.Num.q/3`) and runs through `NxFixed.Block.run/4`. Item tensors are
  `NxFixed.Fixed` or `f32` with the item index first; complex items are
  `{re, im}`. See `ExSdr.Filters.Kernels` for the number format rules and
  docs/blocks/filters.md for the coverage table.

  Names carry GNU Radio's type suffix: the first letter is the input item
  (`f` real, `c` complex), the second the output item, the third the tap type.

  Common options: `:format`, `:taps` (floats, or `{re, im}` for complex taps),
  `:tap_frac` (fractional bits of the fixed-point taps, default the
  format's).

    * `:fir_filter_fff | :ccf | :ccc | :fcc` - `:decimation` (default 1; 1 is a
      stream block, more a frame block), also as `:fft_filter_fff | :ccf | :ccc`
      (the same arithmetic computed directly instead of by FFT)
    * `:interp_fir_filter_fff | :ccf | :ccc | :fcc` - `:interpolation`
    * `:rational_resampler_fff | :ccf | :ccc | :fcc` - `:interpolation`,
      `:decimation`, optional `:phase`
    * `:freq_xlating_fir_filter_ccf | :ccc | :fcf | :fcc` - `:decimation`,
      `:center_freq`, `:sampling_freq`
    * `:hilbert_fc` - `:ntaps`, `:window`, `:beta` (or `:taps` of the imaginary branch)
    * `:dc_blocker_ff | :cc` - `:length`, `:long_form` (default true)
    * `:single_pole_iir_filter_ff | :cc` - `:alpha`
    * `:iir_filter_ffd` - `:fftaps`, `:fbtaps`, `:oldstyle` (default true)
  """
  @behaviour NxFixed.Block

  alias ExSdr.Filters.{Firdes, Kernels}
  alias NxFixed.Block
  alias NxFixed.Num

  @fir ~w(fir_filter_fff fir_filter_ccf fir_filter_ccc fir_filter_fcc
          fft_filter_fff fft_filter_ccf fft_filter_ccc)a
  @interp ~w(interp_fir_filter_fff interp_fir_filter_ccf interp_fir_filter_ccc interp_fir_filter_fcc)a
  @resamp ~w(rational_resampler_fff rational_resampler_ccf rational_resampler_ccc rational_resampler_fcc)a
  @xlating ~w(freq_xlating_fir_filter_ccf freq_xlating_fir_filter_ccc
              freq_xlating_fir_filter_fcf freq_xlating_fir_filter_fcc)a

  @doc "Names accepted by `block/2`."
  def names do
    @fir ++
      @interp ++
      @resamp ++
      @xlating ++
      ~w(hilbert_fc dc_blocker_ff dc_blocker_cc single_pole_iir_filter_ff
         single_pole_iir_filter_cc iir_filter_ffd)a
  end

  @impl NxFixed.Block
  def block(name, opts \\ [])

  def block(name, opts) when name in @fir do
    {taps, dec, f} = fir_args(name, opts)
    fir_block(name, taps, dec, f, opts, gr_name(name))
  end

  def block(name, opts) when name in @interp do
    {f, taps, tf} = common(name, opts)
    i = Keyword.fetch!(opts, :interpolation)
    k = length(hd(Kernels.polyphase(taps, i)))

    %Block{
      name: name,
      kind: :frame,
      fun: &Kernels.interp_frame/3,
      init: line(f, items(name), k - 1),
      opts: [format: f, taps: taps, tap_frac: tf, interpolation: i],
      rate: {1, i},
      latency: group_delay(taps),
      warmup: k - 1,
      item: items(name),
      hardware: :feed_forward,
      gnuradio: gr_name(name)
    }
  end

  def block(name, opts) when name in @resamp do
    {f, taps, tf} = common(name, opts)
    i = Keyword.fetch!(opts, :interpolation)
    d = Keyword.fetch!(opts, :decimation)
    phase = Keyword.get(opts, :phase, 0)
    k = length(hd(Kernels.polyphase(taps, i)))

    %Block{
      name: name,
      kind: :frame,
      fun: &Kernels.resample_frame/3,
      init: line(f, items(name), k - 1),
      opts: [format: f, taps: taps, tap_frac: tf, interpolation: i, decimation: d, phase: phase],
      rate: {d, i},
      latency: div(group_delay(taps), d),
      warmup: k - 1,
      item: items(name),
      hardware: :feed_forward,
      gnuradio: gr_name(name)
    }
  end

  def block(name, opts) when name in @xlating do
    {f, taps, tf} = common(name, opts)
    d = Keyword.get(opts, :decimation, 1)
    fc = Keyword.fetch!(opts, :center_freq)
    fs = Keyword.fetch!(opts, :sampling_freq)
    w0 = 2 * :math.pi() * fc / fs
    ctaps = xlating_taps(taps, fc, fs)
    inc = round(-w0 * d / (2 * :math.pi()) * 65536) |> Bitwise.band(0xFFFF)
    item = items(name)
    n = length(Kernels.pad_taps(ctaps))

    %Block{
      name: name,
      kind: :frame,
      fun: &Kernels.fir_rotate_frame/3,
      # the NCO advances before use; GNU Radio's first output is not rotated
      init: {line(f, {elem(item, 0), :complex}, n - 1), Nx.u16(Bitwise.band(-inc, 0xFFFF))},
      opts: [format: f, taps: ctaps, tap_frac: tf, phase_inc: inc, decimation: d],
      rate: {d, 1},
      latency: div(group_delay(taps), d),
      warmup: n - 1,
      item: item,
      hardware: :feed_forward,
      gnuradio: gr_name(name)
    }
  end

  def block(:hilbert_fc = name, opts) do
    f = Keyword.get(opts, :format, Num.float())
    ntaps = Keyword.get(opts, :ntaps, 65)

    h =
      Keyword.get_lazy(opts, :taps, fn ->
        Firdes.hilbert(
          ntaps,
          Keyword.get(opts, :window, :hamming),
          Keyword.get(opts, :beta, 6.76)
        )
      end)

    n = length(h)
    delay = div(n - 1, 2)

    taps =
      h |> Enum.with_index() |> Enum.map(fn {t, j} -> {if(j == delay, do: 1.0, else: 0.0), t} end)

    fir_block(name, taps, 1, f, opts, "filter.hilbert_fc", {:real, :complex})
    |> Map.put(:latency, delay)
  end

  def block(name, opts) when name in [:dc_blocker_ff, :dc_blocker_cc] do
    f = Keyword.get(opts, :format, Num.float())
    d = Keyword.fetch!(opts, :length)
    long = Keyword.get(opts, :long_form, true)
    taps = dc_blocker_taps(d, long)
    item = if name == :dc_blocker_ff, do: {:real, :real}, else: {:complex, :complex}
    # more fractional bits than the data: the DC null is only as good as the tap sum
    opts = Keyword.put_new(opts, :tap_frac, (f.frac || 0) + 8)

    fir_block(name, taps, 1, f, opts, "filter.#{name}", item)
    |> Map.put(:latency, if(long, do: 2 * d - 2, else: d - 1))
  end

  def block(name, opts) when name in [:single_pole_iir_filter_ff, :single_pole_iir_filter_cc] do
    f = Keyword.get(opts, :format, Num.float())
    alpha = Keyword.fetch!(opts, :alpha)
    tf = Keyword.get(opts, :tap_frac) || f.frac || 0
    # (1 - alpha) from the rounded alpha, so that the DC gain is exactly 1
    alpha =
      if f.kind == :fixed, do: round(alpha * Bitwise.bsl(1, tf)) / Bitwise.bsl(1, tf), else: alpha

    item = if name == :single_pole_iir_filter_ff, do: {:real, :real}, else: {:complex, :complex}

    iir_block(
      name,
      [alpha],
      [1.0 - alpha],
      f,
      tf,
      item,
      "filter.#{name}",
      ceil(16 * :math.log(2) / -:math.log(1 - alpha))
    )
  end

  def block(:iir_filter_ffd = name, opts) do
    f = Keyword.get(opts, :format, Num.float())
    ff = Keyword.fetch!(opts, :fftaps)
    fbtaps = Keyword.fetch!(opts, :fbtaps)
    tf = Keyword.get(opts, :tap_frac) || f.frac || 0
    # GNU Radio ignores fbtaps[0]; old style adds the others, new style subtracts them
    fb =
      fbtaps
      |> Enum.drop(1)
      |> Enum.map(&if(Keyword.get(opts, :oldstyle, true), do: &1, else: -&1))

    iir_block(name, ff, fb, f, tf, {:real, :real}, "filter.iir_filter_ffd", settling(ff, fb))
  end

  def block(name, _opts), do: raise(ArgumentError, "unknown filters block #{inspect(name)}")

  # -- builders ---------------------------------------------------------------------------

  defp fir_args(name, opts) do
    f = Keyword.get(opts, :format, Num.float())
    taps = Keyword.fetch!(opts, :taps)
    {_, _, t} = suffix(name)

    if Kernels.complex_taps?(taps) != (t == "c"),
      do:
        raise(ArgumentError, "#{name}: taps must be #{if t == "c", do: "complex", else: "real"}")

    {taps, Keyword.get(opts, :decimation, 1), f}
  end

  defp gr_name(name), do: "filter.#{name}"

  defp fir_block(name, taps, dec, f, opts, gr, item \\ nil) do
    item = item || items(name)
    tf = Keyword.get(opts, :tap_frac)
    n = length(Kernels.pad_taps(taps))
    kernel_opts = [format: f, taps: taps, tap_frac: tf]

    base = %Block{
      name: name,
      kind: :stream,
      fun: &Kernels.fir_step/3,
      init: line(f, item, n - 1),
      opts: kernel_opts,
      latency: div(group_delay(taps), dec),
      warmup: n - 1,
      item: item,
      hardware: :feed_forward,
      gnuradio: gr
    }

    if dec == 1,
      do: base,
      else: %{
        base
        | kind: :frame,
          fun: &Kernels.fir_frame/3,
          rate: {dec, 1},
          opts: kernel_opts ++ [decimation: dec]
      }
  end

  defp iir_block(name, ff, fb, f, tf, item, gr, warmup) do
    {ffp, fbp} = Kernels.pad_iir(ff, fb)

    %Block{
      name: name,
      kind: :stream,
      fun: &Kernels.iir_step/3,
      init: {line(f, item, length(ffp) - 1), line(f, item, length(fbp))},
      opts: [format: f, ff: ff, fb: fb, tap_frac: tf],
      warmup: warmup,
      item: item,
      hardware: :feedback,
      gnuradio: gr
    }
  end

  defp common(name, opts) do
    f = Keyword.get(opts, :format, Num.float())
    taps = Keyword.fetch!(opts, :taps)
    {_, _, t} = suffix(name)

    if Kernels.complex_taps?(taps) != (t == "c"),
      do:
        raise(ArgumentError, "#{name}: taps must be #{if t == "c", do: "complex", else: "real"}")

    {f, taps, Keyword.get(opts, :tap_frac)}
  end

  # `{in, out, taps}` letters of the type suffix.
  defp suffix(name) do
    <<i::binary-1, o::binary-1, t::binary-1>> = name |> Atom.to_string() |> String.slice(-3, 3)
    {i, o, t}
  end

  defp items(name) do
    {i, o, _} = suffix(name)
    {item(i), item(o)}
  end

  defp item("f"), do: :real
  defp item("c"), do: :complex

  # Zero delay line of `n` items for items of type `item`'s input side.
  defp line(f, {:complex, _}, n), do: {Num.zeros({n}, f), Num.zeros({n}, f)}
  defp line(f, {:real, _}, n), do: Num.zeros({n}, f)

  @doc """
  The composite taps of `freq_xlating_fir_filter`: each tap is turned by
  `2 pi fc / fs * (i - (n - 1) / 2)`, so the filter is centred on the carrier
  (GNU Radio 3.10). The output is then rotated down by `fc` per input item.
  """
  def xlating_taps(taps, center_freq, sampling_freq) do
    w0 = 2 * :math.pi() * center_freq / sampling_freq
    mid = (length(taps) - 1) / 2

    for {t, i} <- Enum.with_index(taps) do
      {tr, ti} = if is_tuple(t), do: t, else: {t, 0.0}
      {c, s} = {:math.cos(w0 * (i - mid)), :math.sin(w0 * (i - mid))}
      {tr * c - ti * s, tr * s + ti * c}
    end
  end

  @doc """
  Taps of GNU Radio's DC blocker as an equivalent FIR: the input delayed by
  the group delay minus the output of two (short form) or four (long form)
  cascaded `length`-point moving averages. Its DC gain is exactly zero.
  """
  def dc_blocker_taps(d, long_form) do
    box = List.duplicate(1.0 / d, d)

    avg =
      if long_form,
        do: box |> convolve(box) |> convolve(box) |> convolve(box),
        else: convolve(box, box)

    delay = if long_form, do: 2 * d - 2, else: d - 1

    avg
    |> Enum.map(&(-&1))
    |> Enum.with_index()
    |> Enum.map(fn {t, i} -> if i == delay, do: t + 1.0, else: t end)
  end

  defp convolve(a, b) do
    for k <- 0..(length(a) + length(b) - 2) do
      for {x, i} <- Enum.with_index(a), j = k - i, j >= 0 and j < length(b), reduce: 0.0 do
        acc -> acc + x * Enum.at(b, j)
      end
    end
  end

  @doc """
  Group delay of a tap list in input items: exact for symmetric and
  antisymmetric (linear phase) taps, otherwise the centroid of the tap energy.
  """
  def group_delay(taps) do
    n = length(taps)

    z =
      Enum.map(taps, fn
        {r, i} -> {r, i}
        r -> {r, 0.0}
      end)

    rev = Enum.reverse(z)

    linear? =
      Enum.zip(z, rev)
      |> Enum.all?(fn {{ar, ai}, {br, bi}} ->
        abs(ar - br) + abs(ai - bi) < 1.0e-9 or abs(ar + br) + abs(ai + bi) < 1.0e-9 or
          abs(ar - br) + abs(ai + bi) < 1.0e-9 or abs(ar + br) + abs(ai - bi) < 1.0e-9
      end)

    if linear? do
      div(n - 1, 2)
    else
      energy = Enum.map(z, fn {r, i} -> r * r + i * i end)
      total = Enum.sum(energy)

      if total == 0,
        do: 0,
        else: round(Enum.sum(Enum.with_index(energy, fn e, i -> e * i end)) / total)
    end
  end

  # Input items after which the impulse response stays below 2^-17.
  defp settling(ff, fb) do
    ff = List.to_tuple(ff)
    fb = List.to_tuple(fb)
    limit = 1.0e6 |> trunc()

    Enum.reduce_while(0..limit, {[], 0}, fn n, {ys, last_big} ->
      x = fn k -> if k == 0, do: 1.0, else: 0.0 end

      acc =
        Enum.reduce(0..(tuple_size(ff) - 1), 0.0, fn k, a ->
          if n - k >= 0, do: a + elem(ff, k) * x.(n - k), else: a
        end)

      acc =
        Enum.reduce(0..(tuple_size(fb) - 1)//1, acc, fn k, a ->
          if n - k - 1 >= 0, do: a + elem(fb, k) * Enum.at(ys, k), else: a
        end)

      last_big = if abs(acc) >= 7.6e-6, do: n, else: last_big

      if n - last_big > 64,
        do: {:halt, {ys, last_big + 1}},
        else: {:cont, {[acc | ys] |> Enum.take(tuple_size(fb)), last_big}}
    end)
    |> elem(1)
  end
end
