defmodule ExSdr.Sources.Channel do
  @moduledoc """
  Channel models built from small blocks: `channel_model` (timing offset,
  multipath, frequency offset, AWGN), `fading_model` (flat Rayleigh or
  Rician fading by a sum of sinusoids) and `selective_fading_model`
  (several delayed, independently fading paths).

  All blocks work on complex samples (`{re, im}` pairs of vectors, as in
  `NxFixed.Block.run/4`). The parts are available on their own:

    * `:fir_ccc` - FIR filter with complex (or real) taps, `fir_ccc/3`
    * `:add_noise` - adds complex white noise, `add_noise/3`
    * `:fader` - multiplies by a fading process, `fader/4`
    * `:mmse_resampler_cc` - `ExSdr.Sources.Resampler`
    * `:rotator_cc` - `ExSdr.Sources.NCO.rotator/4`

  and `sequence/2` and `parallel_sum/2` combine blocks into one (a `:host`
  block with the states of the parts as its state).

  ## Fading

  The process is `h[n] = sum_k a_k exp(j (phi_k + 2 pi f_k n))` with `N`
  scattered components `f_k = fDTs cos(alpha_k)`, `alpha_k = (2 pi k +
  theta) / N`, amplitude `sqrt(1 / (N (K + 1)))` each, random phases `phi_k`
  and `theta` from the seeded generator, plus, for Rician fading (`los:
  true`), a line of sight component of amplitude `sqrt(K / (K + 1))` at
  Doppler `fDTs cos(theta_0)`. `E|h|^2 = 1`, the envelope is Rayleigh
  (Rician with factor `K`) over the ensemble of seeds, and the correlation
  `E[h[n] conj(h[n + m])]` follows the Jakes spectrum `J0(2 pi fDTs m)`
  (for `N` of 16 and more it is accurate for `fDTs m` up to about `N / 8`).
  `fDTs` is the maximum Doppler shift times the sample period (cycles per
  sample, below 0.5). The Doppler of each component is a phase word step
  (`ExSdr.Sources.NCO`), so the process is exactly periodic in the word
  and splits exactly. This is the structure of GNU Radio's `fading_model`,
  not its random number stream: the realisations differ, the statistics do
  not (see `docs/blocks/sources.md`).

  In fixed point the fading gain is formed without saturation (it can
  exceed 1 in magnitude) and the product with the signal is quantised
  and saturated like any other output.
  """
  import Nx.Defn
  import NxFixed.Num

  alias ExSdr.Sources.{Noise, NCO, Resampler}
  alias NxFixed.Block
  alias NxFixed.Fixed
  alias NxFixed.Num.Format

  # -- kernels --------------------------------------------------------------------------------

  @doc """
  FIR filter with constant taps on complex samples (GNU Radio's
  `fir_filter_ccc`; real taps: `fir_filter_ccf`). `:taps` are floats or
  `{re, im}` pairs. The state is one delay line per component.
  """
  defn fir_ccc(x, lines, opts \\ []) do
    opts = keyword!(opts, [:format, :taps])
    fir_complex_taps(x, lines, opts[:taps], opts[:format])
  end

  deftransform fir_complex_taps(x, {line_re, line_im}, taps, format) do
    {re, im} = x

    taps =
      Enum.map(taps, fn
        {r, i} -> {r, i}
        r -> {r, 0.0}
      end)
      |> pad_taps()

    t_re = Enum.map(taps, &elem(&1, 0))
    t_im = Enum.map(taps, &elem(&1, 1))
    {a_rr, next_re} = fir(re, line_re, t_re, format)
    {a_ii, _} = fir(im, line_im, t_im, format)
    {a_ri, next_im} = fir(im, line_im, t_re, format)
    {a_ir, _} = fir(re, line_re, t_im, format)

    y =
      if Enum.all?(t_im, &(&1 == 0.0)),
        do: {a_rr, a_ri},
        else: {sub(a_rr, a_ii), add(a_ri, a_ir)}

    {quantize(y, format), {next_re, next_im}}
  end

  # `fir/4` needs at least two taps
  defp pad_taps([_] = taps), do: taps ++ [{0.0, 0.0}]
  defp pad_taps(taps), do: taps

  @doc "Initial state of `fir_ccc/3` for `n` taps."
  def fir_state(n, format) do
    n = max(n, 2)
    {zeros({n - 1}, format), zeros({n - 1}, format)}
  end

  @doc """
  Adds complex white noise: `step(x, generator, opts)` with the options of
  `ExSdr.Sources.Noise.noise_source/3` (`:type`, `:amplitude`, ...).
  """
  defn add_noise(x, state, opts \\ []) do
    opts = keyword!(opts, [:format, :type, :amplitude, :table_bits, :impulse_factor])
    noisy(x, state, opts)
  end

  deftransform noisy(x, state, opts) do
    {n, next} = Noise.draw(state, Keyword.put(opts, :complex, true))
    {x |> cadd(n) |> quantize(opts[:format]), next}
  end

  @doc """
  Fading: `y = x * h`, `h = sum_k w_k exp(j phase_k)` over the components.
  `phases` is the state (a `u32` vector, one per component), `incs` the
  phase steps; the amplitudes `w_k` (floats) are the compile time option
  `:weights`, so the hardware has constant multipliers and the width check
  knows the exact range of the gain. Other options: `:format`, `:phase_bits`.
  """
  defn fader(x, phases, incs, opts \\ []) do
    opts = keyword!(opts, [:format, :weights, phase_bits: 32])
    format = opts[:format]
    h = fading_gain(phases, opts[:weights], opts[:phase_bits], format)
    y = x |> cmul(h) |> rescale(format) |> quantize(format)
    {y, NCO.advance(phases, incs, opts[:phase_bits])}
  end

  deftransform fading_gain(phases, weights, bits, format) do
    {c, s} = NCO.sin_cos(NCO.widen(phases, bits), format)
    {sum_weighted(c, weights, format), sum_weighted(s, weights, format)}
  end

  defp sum_weighted(v, weights, %Format{kind: :float}),
    do: Nx.dot(v, Nx.tensor(weights, type: :f32, backend: Nx.BinaryBackend))

  defp sum_weighted(%Fixed{t: t, frac: fv}, weights, %Format{kind: :fixed} = f) do
    q = quantize_constants(weights, f)
    acc = Nx.dot(Nx.as_type(t, :s64), Nx.tensor(q, type: :s64, backend: Nx.BinaryBackend))
    rescale(%Fixed{t: acc, frac: fv + f.frac}, f)
  end

  # -- blocks -------------------------------------------------------------------------------------

  @doc "Blocks of this module: see `ExSdr.Sources.block/2`."
  def block(:fir_ccc, opts) do
    f = format(opts)
    taps = Keyword.fetch!(opts, :taps)

    %Block{
      name: :fir_ccc,
      kind: :stream,
      fun: &__MODULE__.fir_ccc/3,
      init: fir_state(length(taps), f),
      opts: [format: f, taps: taps],
      latency: div(length(taps) - 1, 2),
      warmup: max(length(taps) - 1, 1),
      item: {:complex, :complex},
      hardware: :feed_forward,
      gnuradio:
        if(Enum.all?(taps, &is_number/1),
          do: "filter.fir_filter_ccf",
          else: "filter.fir_filter_ccc"
        )
    }
  end

  def block(:add_noise, opts) do
    f = format(opts)

    %Block{
      name: :add_noise,
      kind: :stream,
      fun: &__MODULE__.add_noise/3,
      init: Noise.seed_state(Keyword.get(opts, :seed, 0)),
      opts:
        [
          format: f,
          type: Keyword.get(opts, :type, :gaussian),
          amplitude: Keyword.fetch!(opts, :amplitude)
        ] ++
          Keyword.take(opts, [:table_bits, :impulse_factor]),
      item: {:complex, :complex},
      hardware: :feedback,
      warmup: :unbounded
    }
  end

  def block(:mmse_resampler_cc, opts), do: Resampler.block(opts)
  def block(:rotator_cc, opts), do: ExSdr.Sources.block(:rotator_cc, opts)

  def block(:fader, opts) do
    f = format(opts)
    n = Keyword.get(opts, :n, 16)
    bits = Keyword.get(opts, :phase_bits, 32)
    {phases, incs, weights} = components(n, opts, bits)

    %Block{
      name: :fader,
      kind: :stream,
      fun: &__MODULE__.fader/4,
      init: Nx.tensor(phases, type: :u32),
      params: [Nx.tensor(incs, type: :u32)],
      opts: [format: f, phase_bits: bits, weights: weights],
      item: {:complex, :complex},
      hardware: :feedback,
      warmup: :unbounded
    }
  end

  def block(:fading_model, opts) do
    b = block(:fader, opts)
    %{b | name: :fading_model, gnuradio: "channels.fading_model"}
  end

  def block(:selective_fading_model, opts) do
    delays = Keyword.fetch!(opts, :delays)
    mags = Keyword.fetch!(opts, :mags)
    ntaps = Keyword.get(opts, :ntaps, 8)
    seed = Keyword.get(opts, :seed, 0)
    f = format(opts)
    true = length(delays) == length(mags)

    paths =
      for {{d, m}, p} <- Enum.with_index(Enum.zip(delays, mags)) do
        taps = fractional_delay_taps(d, ntaps) |> Enum.map(&(&1 * m))

        sequence(
          [
            block(:fir_ccc, format: f, taps: taps),
            block(:fader, Keyword.merge(opts, seed: seed + 1 + p))
          ],
          :path
        )
      end

    b = parallel_sum(paths, f)
    %{b | name: :selective_fading_model, gnuradio: "channels.selective_fading_model"}
  end

  def block(:channel_model, opts) do
    f = format(opts)
    taps = Keyword.get(opts, :taps, [1.0])
    epsilon = Keyword.get(opts, :epsilon, 1.0)
    freq = Keyword.get(opts, :frequency_offset, 0.0)
    noise = Keyword.get(opts, :noise_voltage, 0.0)

    stages =
      [
        if(Keyword.get(opts, :resample, true),
          do: Resampler.block(format: f, ratio: epsilon)
        ),
        if(taps != [1.0], do: block(:fir_ccc, format: f, taps: taps)),
        if(freq != 0.0, do: ExSdr.Sources.block(:rotator_cc, format: f, frequency: freq)),
        if(noise != 0.0,
          do: block(:add_noise, format: f, amplitude: noise, seed: Keyword.get(opts, :seed, 0))
        )
      ]
      |> Enum.reject(&is_nil/1)

    b = sequence(stages, :channel_model)
    %{b | gnuradio: "channels.channel_model"}
  end

  def block(name, _opts), do: raise(ArgumentError, "unknown block #{inspect(name)}")

  defp format(opts), do: Keyword.get(opts, :format, NxFixed.Num.float())

  # -- composition -----------------------------------------------------------------------------------

  @doc """
  Runs `blocks` one after another as a single `:host` block (complex
  input and output; stages may change the number of items). The state is
  the list of the stages' states.
  """
  def sequence([single], _name), do: single

  def sequence(blocks, name) do
    {rin, rout} = Enum.reduce(blocks, {1, 1}, fn b, acc -> mul_rate(acc, b.rate) end)

    %Block{
      name: name,
      kind: :host,
      fun: fn input, states -> run_stages(blocks, input, states) end,
      init: Enum.map(blocks, & &1.init),
      rate: {rin, rout},
      latency: blocks |> Enum.map(& &1.latency) |> Enum.sum(),
      warmup: blocks |> Enum.map(& &1.warmup) |> sum_warmup(),
      item: {:complex, :complex},
      hardware: blocks |> Enum.map(& &1.hardware) |> worst_hardware()
    }
  end

  defp run_stages(blocks, input, states) do
    {out, next} =
      Enum.zip_reduce(blocks, states, {input, []}, fn b, s, {x, acc} ->
        if items(x) == 0 do
          {x, [s | acc]}
        else
          {y, s2} = Block.run(b, x, s)
          {y, [s2 | acc]}
        end
      end)

    {out, Enum.reverse(next)}
  end

  @doc """
  Several blocks on the same input, outputs added (and saturated):
  parallel paths, e.g. of `selective_fading_model`. Paths must produce the
  same number of items.
  """
  def parallel_sum(paths, f) do
    %Block{
      name: :parallel_sum,
      kind: :host,
      fun: fn input, states ->
        {outs, next} =
          paths
          |> Enum.zip(states)
          |> Enum.map(fn {b, s} -> Block.run(b, input, s) end)
          |> Enum.unzip()

        {Enum.reduce(outs, &add_complex(&1, &2, f)), next}
      end,
      init: Enum.map(paths, & &1.init),
      rate: hd(paths).rate,
      latency: paths |> Enum.map(& &1.latency) |> Enum.max(),
      warmup: paths |> Enum.map(& &1.warmup) |> sum_warmup(),
      item: {:complex, :complex},
      hardware: :feedback
    }
  end

  defp add_complex(a, b, f), do: a |> cadd(b) |> quantize(f)

  defp items(nil), do: 0
  defp items({re, _im}), do: re |> raw() |> Nx.axis_size(0)
  defp raw(%Fixed{t: t}), do: t
  defp raw(t), do: t

  defp mul_rate({a, b}, {c, d}) do
    g = Integer.gcd(a * c, b * d)
    {div(a * c, g), div(b * d, g)}
  end

  defp sum_warmup(list), do: if(:unbounded in list, do: :unbounded, else: Enum.sum(list))

  @order [:stateless, :feed_forward, :memory, :transform, :control, :feedback]
  defp worst_hardware(list), do: Enum.max_by(list, &Enum.find_index(@order, fn x -> x == &1 end))

  # -- parameters ------------------------------------------------------------------------------------

  @doc """
  Windowed sinc taps (Hamming window of `ntaps` points width) for a delay
  of `delay` samples. The taps are centred `ntaps / 2` samples in, so the
  realised delay is `delay + ntaps / 2`; a whole number of samples gives
  exactly one tap of 1.
  """
  def fractional_delay_taps(delay, ntaps) do
    centre = ntaps / 2 + delay
    count = ceil(centre + ntaps / 2) + 1
    half = ntaps / 2

    for k <- 0..(count - 1) do
      d = k - centre

      if abs(d) >= half do
        0.0
      else
        sinc(d) * (0.54 + 0.46 * :math.cos(:math.pi() * d / half))
      end
    end
  end

  defp sinc(d) do
    r = abs(d - round(d))

    if r < 1.0e-12,
      do: if(round(d) == 0, do: 1.0, else: 0.0),
      else: :math.sin(:math.pi() * d) / (:math.pi() * d)
  end

  # Phase words, steps and amplitudes of the fading components: N scattered
  # ones, plus the line of sight when `los: true`.
  defp components(n, opts, bits) do
    fdts = Keyword.get(opts, :fdts, 0.01)
    k = Keyword.get(opts, :k, 4.0)
    los? = Keyword.get(opts, :los, false)
    uniform = Noise.host_uniform(Keyword.get(opts, :seed, 0))
    {theta, uniform} = uniform.()
    {theta0, uniform} = uniform.()

    {phases, _} =
      Enum.map_reduce(1..(n + 1), uniform, fn _, u ->
        {x, u} = u.()
        {NCO.phase_word(x, bits), u}
      end)

    [los_phase | scatter_phases] = phases
    two_pi = 2 * :math.pi()
    scatter_k = if los?, do: k, else: 0.0
    a = :math.sqrt(1 / (n * (scatter_k + 1)))

    scatter_incs =
      for i <- 0..(n - 1),
          do: NCO.phase_word(fdts * :math.cos((two_pi * i + two_pi * theta) / n), bits)

    if los? do
      los_inc = NCO.phase_word(fdts * :math.cos(two_pi * theta0), bits)

      {[los_phase | scatter_phases], [los_inc | scatter_incs],
       [:math.sqrt(k / (k + 1)) | List.duplicate(a, n)]}
    else
      {scatter_phases, scatter_incs, List.duplicate(a, n)}
    end
  end
end
