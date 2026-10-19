defmodule ExSdr.Analog do
  @moduledoc """
  The GNU Radio `gr-analog` family: modulation, demodulation, phase locked
  loops, gain control and squelch, as `NxFixed.Block`s that run bit-exact on
  CPU (Nx) and FPGA.

      ExSdr.Analog.block(:agc_cc, format: NxFixed.Num.q(16, 15), rate: 1.0e-3,
                         reference: 0.5)

  returns a `%NxFixed.Block{}`; `NxFixed.Block.run/4` runs it, carrying the
  state, so a stream cut anywhere gives the same output as one call.
  `docs/blocks/analog.md` lists every block with its options, status and
  deviations from GNU Radio.

  Common options: `:format` (`NxFixed.Num.float()`, the default, or
  `NxFixed.Num.q/3`). Angles are 32 bit words (see `ExSdr.Analog.Math`), so a
  phase wraps exactly in both instances; `lut_bits` of the format sets the
  sine table of every NCO, and for fixed point `:iterations` (default 16) and
  `:guard` (default 8) set the CORDIC arctangent of the discriminators.
  """
  @behaviour NxFixed.Block

  alias ExSdr.Analog.Kernels, as: K
  alias NxFixed.Block
  alias NxFixed.Num

  @names ~w(frequency_modulator_fc phase_modulator_fc quadrature_demod_cf am_demod
            fm_deemph fm_preemph agc_cc agc_ff agc2_cc agc2_ff agc3_cc feedforward_agc_cc
            simple_squelch_cc pwr_squelch_cc pwr_squelch_ff rail_ff sample_and_hold_ff
            pll_carriertracking_cc pll_freqdet_cf pll_refout_cc nbfm_tx nbfm_rx_core)a

  @doc "The block names `block/2` knows."
  def names, do: @names

  @impl NxFixed.Block
  def block(name, opts \\ [])

  def block(:frequency_modulator_fc, opts) do
    {_f, o} = common(opts, sensitivity: :required)

    %Block{
      name: :frequency_modulator_fc,
      kind: :stream,
      fun: &K.frequency_modulator/3,
      init: Nx.tensor(0, type: :s32),
      opts: o,
      item: {:real, :complex},
      hardware: :feedback,
      warmup: :infinity,
      gnuradio: "analog.frequency_modulator_fc"
    }
  end

  def block(:phase_modulator_fc, opts) do
    {_f, o} = common(opts, sensitivity: :required)

    %Block{
      name: :phase_modulator_fc,
      kind: :stream,
      fun: &K.phase_modulator/3,
      init: Nx.tensor(0, type: :s32),
      opts: o,
      item: {:real, :complex},
      hardware: :stateless,
      gnuradio: "analog.phase_modulator_fc"
    }
  end

  def block(:quadrature_demod_cf, opts) do
    {f, o} = common(opts, gain: 1.0)

    %Block{
      name: :quadrature_demod_cf,
      kind: :stream,
      fun: &K.quadrature_demod/3,
      init: {Num.zeros({}, f), Num.zeros({}, f)},
      opts: o,
      item: {:complex, :real},
      hardware: :feed_forward,
      warmup: 1,
      gnuradio: "analog.quadrature_demod_cf"
    }
  end

  def block(:am_demod, opts) do
    {_f, o} = common(opts, offset: 1.0)

    %Block{
      name: :am_demod,
      kind: :stream,
      fun: &K.am_demod/3,
      init: Nx.tensor(0, type: :s32),
      opts: o,
      item: {:complex, :real},
      hardware: :stateless,
      gnuradio:
        "blocks.complex_to_mag + blocks.add_const_ff(-1.0), the front of analog.am_demod_cf"
    }
  end

  def block(:fm_deemph, opts) do
    {f, o} = common(opts, fs: :required, tau: 75.0e-6)
    iir_block(:fm_deemph, f, Keyword.merge(o, deemph_taps(o[:fs], o[:tau])), "analog.fm_deemph")
  end

  def block(:fm_preemph, opts) do
    {f, o} = common(opts, fs: :required, tau: 75.0e-6, fh: -1.0)

    iir_block(
      :fm_preemph,
      f,
      Keyword.merge(o, preemph_taps(o[:fs], o[:tau], o[:fh])),
      "analog.fm_preemph"
    )
  end

  def block(name, opts) when name in [:agc_cc, :agc_ff] do
    {f, o} = common(opts, rate: 1.0e-4, reference: 1.0, gain: 1.0, max_gain: 65536.0)
    agc_block(name, &K.agc/3, f, o, "analog.#{name}")
  end

  def block(name, opts) when name in [:agc2_cc, :agc2_ff] do
    {f, o} =
      common(opts,
        attack_rate: 0.1,
        decay_rate: 0.01,
        reference: 1.0,
        gain: 1.0,
        max_gain: 65536.0
      )

    agc_block(name, &K.agc2/3, f, o, "analog.#{name}")
  end

  def block(:agc3_cc, opts) do
    {f, o} =
      common(opts,
        attack_rate: 0.1,
        decay_rate: 0.01,
        reference: 1.0,
        gain: 1.0,
        iir_update_decim: 1,
        max_gain: 65536.0
      )

    %Block{
      name: :agc3_cc,
      kind: :stream,
      fun: &K.agc3/3,
      init:
        {K.gain_init(o[:gain], f, o[:max_gain]), Nx.tensor(0, type: :s32),
         Nx.tensor(0, type: :s32)},
      opts: o,
      item: {:complex, :complex},
      hardware: :feedback,
      warmup: :infinity,
      gnuradio: "analog.agc3_cc"
    }
  end

  def block(:feedforward_agc_cc, opts) do
    {f, o} = common(opts, n: :required, reference: 1.0)
    n = o[:n]
    z = Num.zeros({n - 1}, f)

    %Block{
      name: :feedforward_agc_cc,
      kind: :stream,
      fun: &K.feedforward_agc/3,
      init: {z, z},
      opts: o,
      item: {:complex, :complex},
      latency: n - 1,
      warmup: n - 1,
      hardware: :feed_forward,
      gnuradio: "analog.feedforward_agc_cc"
    }
  end

  def block(name, opts) when name in [:simple_squelch_cc, :pwr_squelch_cc, :pwr_squelch_ff] do
    default_alpha = if name == :simple_squelch_cc, do: 0.001, else: 0.0001
    {f, o} = common(opts, threshold_db: :required, alpha: default_alpha)
    o = Keyword.put(o, :threshold, :math.pow(10, o[:threshold_db] / 10))
    item = if name == :pwr_squelch_ff, do: {:real, :real}, else: {:complex, :complex}

    %Block{
      name: name,
      kind: :stream,
      fun: &K.squelch/3,
      init: Num.zeros({}, K.power_format(f)),
      opts: o,
      item: item,
      hardware: :feedback,
      warmup: :infinity,
      gnuradio: "analog.#{name}"
    }
  end

  def block(:rail_ff, opts) do
    {_f, o} = common(opts, lo: :required, hi: :required)

    %Block{
      name: :rail_ff,
      kind: :stream,
      fun: &K.rail/3,
      init: Nx.tensor(0, type: :s32),
      opts: o,
      hardware: :stateless,
      gnuradio: "analog.rail_ff"
    }
  end

  def block(:sample_and_hold_ff, opts) do
    {f, o} = common(opts, [])

    %Block{
      name: :sample_and_hold_ff,
      kind: :stream,
      fun: &K.sample_and_hold/3,
      init: Num.zeros({}, f),
      opts: o,
      hardware: :feedback,
      warmup: 1,
      gnuradio: "analog.sample_and_hold_ff"
    }
  end

  def block(name, opts) when name in [:pll_carriertracking_cc, :pll_freqdet_cf, :pll_refout_cc] do
    {_f, o} = common(opts, loop_bw: :required, max_freq: :math.pi(), min_freq: -:math.pi())

    {fun, item} =
      case name do
        :pll_carriertracking_cc -> {&K.pll_carriertracking/3, {:complex, :complex}}
        :pll_freqdet_cf -> {&K.pll_freqdet/3, {:complex, :real}}
        :pll_refout_cc -> {&K.pll_refout/3, {:complex, :complex}}
      end

    %Block{
      name: name,
      kind: :stream,
      fun: fun,
      init: {Nx.tensor(0, type: :s32), Nx.tensor(0, type: :s32)},
      opts: o,
      item: item,
      hardware: :feedback,
      warmup: :infinity,
      gnuradio: "analog.#{name}"
    }
  end

  def block(:nbfm_tx, opts) do
    {f, o} =
      common(opts, quad_rate: :required, tau: 75.0e-6, max_dev: 5.0e3, fh: -1.0)

    o =
      o
      |> Keyword.merge(preemph_taps(o[:quad_rate], o[:tau], o[:fh]))
      |> Keyword.put(:sensitivity, 2 * :math.pi() * o[:max_dev] / o[:quad_rate])

    %Block{
      name: :nbfm_tx,
      kind: :stream,
      fun: &K.nbfm_tx/3,
      init: {K.iir1_init(f), Nx.tensor(0, type: :s32)},
      opts: o,
      item: {:real, :complex},
      hardware: :feedback,
      warmup: :infinity,
      gnuradio: "analog.nbfm_tx (audio_rate == quad_rate)"
    }
  end

  def block(:nbfm_rx_core, opts) do
    {f, o} = common(opts, quad_rate: :required, tau: 75.0e-6, max_dev: 5.0e3)

    o =
      o
      |> Keyword.merge(deemph_taps(o[:quad_rate], o[:tau]))
      |> Keyword.put(:gain, o[:quad_rate] / (2 * :math.pi() * o[:max_dev]))

    z = Num.zeros({}, f)

    %Block{
      name: :nbfm_rx_core,
      kind: :stream,
      fun: &K.nbfm_rx_core/3,
      init: {{z, z}, K.iir1_init(f)},
      opts: o,
      item: {:complex, :real},
      hardware: :feedback,
      warmup: :infinity,
      gnuradio: "analog.nbfm_rx without its audio filter"
    }
  end

  def block(name, _opts), do: raise(ArgumentError, "unknown analog block #{inspect(name)}")

  # -- helpers ------------------------------------------------------------------------------

  # Splits `:format` off, fills defaults, checks required options. Returns
  # `{format, kernel opts}` (kernel opts include the format).
  defp common(opts, defaults) do
    format = Keyword.get(opts, :format, Num.float())
    {extra_defaults, required} = Enum.split_with(defaults, fn {_k, v} -> v != :required end)

    for {k, _} <- required,
        not Keyword.has_key?(opts, k),
        do: raise(ArgumentError, "missing option #{inspect(k)}")

    o =
      extra_defaults
      |> Keyword.merge(Keyword.delete(opts, :format))
      |> Keyword.put(:format, format)

    {format, o}
  end

  defp iir_block(name, f, o, gnuradio) do
    %Block{
      name: name,
      kind: :stream,
      fun: &K.iir1/3,
      init: K.iir1_init(f, Keyword.get(o, :guard, 8)),
      opts: o,
      hardware: :feedback,
      warmup: :infinity,
      gnuradio: gnuradio
    }
  end

  defp agc_block(name, fun, f, o, gnuradio) do
    complex? = name |> Atom.to_string() |> String.ends_with?("_cc")

    %Block{
      name: name,
      kind: :stream,
      fun: fun,
      init: K.gain_init(o[:gain], f, o[:max_gain]),
      opts: o,
      item: if(complex?, do: {:complex, :complex}, else: {:real, :real}),
      hardware: :feedback,
      warmup: :infinity,
      gnuradio: gnuradio
    }
  end

  @doc """
  The start gain GNU Radio's `agc3_cc` picks: `reference / mean(|x|)` over the
  first `4 * iir_update_decim` samples (a list of `{re, im}` floats).
  """
  def agc3_initial_gain(samples, reference, iir_update_decim \\ 1) do
    first = Enum.take(samples, 4 * iir_update_decim)
    mean = Enum.sum(for {re, im} <- first, do: :math.sqrt(re * re + im * im)) / length(first)
    reference / mean
  end

  @doc """
  The taps `analog.fm_deemph(fs, tau)` builds (bilinear transform of
  `1 / (1 + s tau)`): `[b0: .., b1: .., p1: ..]` of
  `y[n] = b0 x[n] + b1 x[n-1] + p1 y[n-1]`.
  """
  def deemph_taps(fs, tau) do
    # the bilinear transform maps the zero at s = infinity to z = -1
    first_order(-1.0, bilinear_root(1.0 / tau, fs))
  end

  @doc """
  The taps `analog.fm_preemph(fs, tau, fh)` builds; `fh <= 0` or `>= fs / 2`
  means `0.925 * fs / 2`. Gain corrected to 0 dB at DC.
  """
  def preemph_taps(fs, tau, fh \\ -1.0) do
    fh = if fh <= 0.0 or fh >= fs / 2.0, do: 0.925 * fs / 2.0, else: fh
    # (1 + s tau) / (1 + s / (2 pi fh)): a zero at 1 / tau, a pole at 2 pi fh
    first_order(bilinear_root(1.0 / tau, fs), bilinear_root(2.0 * :math.pi() * fh, fs))
  end

  # The z-plane position of a real s-plane root at -w (rad/s), bilinear
  # transform with w prewarped so the corner frequency stays where it is.
  defp bilinear_root(w, fs) do
    k = 2.0 * fs
    wa = k * :math.tan(w / k)
    (k - wa) / (k + wa)
  end

  # (1 - zero z^-1) / (1 - pole z^-1), scaled to unit gain at DC
  defp first_order(zero, pole) do
    g = (1.0 - pole) / (1.0 - zero)
    [b0: g, b1: -g * zero, p1: pole]
  end
end
