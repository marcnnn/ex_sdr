defmodule ExSdr.Carrier do
  @moduledoc """
  Carrier and frequency synchronisation, equalisers, correlation and
  estimators from GNU Radio's gr-digital, as `NxFixed.Block`s that run
  bit-exact on CPU (Nx) and FPGA.

      b = ExSdr.Carrier.block(:costas_loop_cc, format: NxFixed.Num.q(16, 15),
                              loop_bw: 0.05, order: 4)
      {y, state} = ExSdr.Carrier.run(b, {i, q})

  `docs/blocks/carrier.md` lists every block with its options, status and
  deviations from GNU Radio.

  Common options: `:format` (`NxFixed.Num.float()`, the default, or
  `NxFixed.Num.q/3`). Phases and frequencies are 32 bit angle words
  (`ExSdr.Analog.Math`); `lut_bits` of the format sets the sine table of
  every NCO and `:iterations` / `:guard` the CORDIC arctangent.
  """
  @behaviour NxFixed.Block

  alias ExSdr.Carrier.Equalizer
  alias ExSdr.Carrier.Estimators
  alias ExSdr.Carrier.Kernels, as: K
  alias ExSdr.Carrier.Loop
  alias ExSdr.Digital.Constellation
  alias NxFixed.Block
  alias NxFixed.Num

  @names ~w(costas_loop_cc fll_band_edge_cc constellation_receiver_cb linear_equalizer
            cma_equalizer_cc lms_dd_equalizer_cc corr_est_cc mpsk_snr_est_cc
            probe_mpsk_snr_est_c probe_density_b probe_signal)a

  @doc "The block names `block/2` knows."
  def names, do: @names

  @doc """
  Runs `block` over `input` (see `NxFixed.Block.run/4`); returns `{output, state}`.
  Float instances run with `allow_float: true`.
  """
  def run(block, input, state \\ :init, opts \\ []) do
    float? = match?(%NxFixed.Num.Format{kind: :float}, block.opts[:format])
    opts = if float?, do: Keyword.put_new(opts, :allow_float, true), else: opts
    Block.run(block, input, state, opts)
  end

  @impl NxFixed.Block
  def block(name, opts \\ [])

  def block(:costas_loop_cc, opts) do
    {_f, o} = common(opts, loop_bw: :required, order: 4, max_freq: 1.0, min_freq: -1.0)

    unless o[:order] in [2, 4, 8],
      do: raise(ArgumentError, "costas_loop_cc: order must be 2, 4 or 8")

    {phase, freq} = Loop.init(o[:phase] || 0.0, o[:freq] || 0.0)

    %Block{
      name: :costas_loop_cc,
      kind: :stream,
      fun: &K.costas/3,
      init: {phase, freq},
      opts: o,
      item: {:complex, out_item(:complex, o)},
      hardware: :feedback,
      warmup: :unbounded,
      gnuradio: "digital.costas_loop_cc"
    }
  end

  def block(:fll_band_edge_cc, opts) do
    {f, o} =
      common(opts,
        samples_per_symbol: :required,
        rolloff: :required,
        filter_size: :required,
        loop_bw: :required
      )

    sps = o[:samples_per_symbol]
    n = o[:filter_size]
    {lower, upper} = K.fll_taps(sps, o[:rolloff], n)
    lim = 2 * :math.pi() * 2.0 / sps

    o =
      o
      |> Keyword.put(:lower, lower)
      |> Keyword.put(:upper, upper)
      |> Keyword.put_new(:max_freq, lim)
      |> Keyword.put_new(:min_freq, -lim)

    {phase, freq} = Loop.init()
    zeros = fn len -> {Num.zeros({len}, f), Num.zeros({len}, f)} end

    %Block{
      name: :fll_band_edge_cc,
      kind: :stream,
      fun: &K.fll/3,
      init: {phase, freq, zeros.(n), zeros.(max(n - 1, 1))},
      opts: o,
      item: {:complex, out_item(:complex, o)},
      latency: n,
      hardware: :feedback,
      warmup: :unbounded,
      gnuradio: "digital.fll_band_edge_cc"
    }
  end

  def block(:constellation_receiver_cb, opts) do
    {_f, o} =
      common(opts,
        constellation: :required,
        loop_bw: :required,
        fmin: :required,
        fmax: :required
      )

    o =
      o
      |> Keyword.put(:points, points(o[:constellation]))
      |> Keyword.delete(:constellation)
      |> Keyword.put(:min_freq, o[:fmin])
      |> Keyword.put(:max_freq, o[:fmax])

    {phase, freq} = Loop.init()

    %Block{
      name: :constellation_receiver_cb,
      kind: :stream,
      fun: &K.receiver/3,
      init: {phase, freq},
      opts: o,
      item: {:complex, out_item(:symbol, o)},
      hardware: :feedback,
      warmup: :unbounded,
      gnuradio: "digital.constellation_receiver_cb"
    }
  end

  def block(:linear_equalizer, opts) do
    {_f, o} =
      common(opts,
        num_taps: :required,
        algorithm: :required,
        mu: :required,
        sps: 1,
        modulus: 1.0,
        adapt_after_training: true
      )

    equalizer_block(:linear_equalizer, o, "digital.linear_equalizer")
  end

  def block(:cma_equalizer_cc, opts) do
    {_f, o} = common(opts, num_taps: :required, mu: :required, modulus: 1.0, sps: 1)
    o = Keyword.put(o, :algorithm, :cma)
    equalizer_block(:cma_equalizer_cc, o, "digital.linear_equalizer with adaptive_algorithm_cma")
  end

  def block(:lms_dd_equalizer_cc, opts) do
    {_f, o} = common(opts, num_taps: :required, mu: :required, sps: 1)
    o = Keyword.put(o, :algorithm, :lms)

    equalizer_block(
      :lms_dd_equalizer_cc,
      o,
      "digital.linear_equalizer with adaptive_algorithm_lms"
    )
  end

  def block(:corr_est_cc, opts) do
    {_f, o} =
      common(opts, symbols: :required, sps: 1, mark_delay: 0, threshold: 0.9, tap_frac: 20)

    n = length(o[:symbols])
    isps = trunc(o[:sps] + 0.5)
    o = Keyword.put(o, :mark_delay, min(o[:mark_delay], n - 1))

    %Block{
      name: :corr_est_cc,
      kind: :stream,
      fun: &Estimators.corr_est/3,
      init: Estimators.corr_init(o),
      opts: o,
      item: {:complex, :corr_est},
      latency: n,
      warmup: n + isps + 2,
      hardware: :control,
      gnuradio: "digital.corr_est_cc"
    }
  end

  def block(name, opts) when name in [:mpsk_snr_est_cc, :probe_mpsk_snr_est_c] do
    {_f, o} = common(opts, type: :required, alpha: 0.001)

    unless o[:type] in [:simple, :skew, :m2m4, :svr],
      do: raise(ArgumentError, "#{name}: type must be :simple, :skew, :m2m4 or :svr")

    %Block{
      name: name,
      kind: :stream,
      fun: &Estimators.snr_est/3,
      init: Estimators.snr_init(o),
      opts: o,
      item: {:complex, :snr_registers},
      latency: 1,
      warmup: :unbounded,
      hardware: :feedback,
      gnuradio: "digital.#{name}"
    }
  end

  def block(:probe_density_b, opts) do
    {f, o} = common(opts, alpha: :required)

    %Block{
      name: :probe_density_b,
      kind: :stream,
      fun: &Estimators.probe_density/3,
      init: Estimators.density_init(f),
      opts: o,
      item: {:byte, :density},
      hardware: :feedback,
      warmup: :unbounded,
      gnuradio: "digital.probe_density_b"
    }
  end

  def block(:probe_signal, opts) do
    {f, o} = common(opts, item: :real)

    init =
      case o[:item] do
        :byte -> Nx.tensor(0, type: :u8)
        :complex -> {Num.zeros({}, f), Num.zeros({}, f)}
        :real -> Num.zeros({}, f)
      end

    %Block{
      name: :probe_signal,
      kind: :stream,
      fun: &Estimators.probe_signal/3,
      init: init,
      opts: o,
      item: {o[:item], o[:item]},
      hardware: :feed_forward,
      warmup: 1,
      gnuradio: "blocks.probe_signal_#{%{real: "f", complex: "c", byte: "b"}[o[:item]]}"
    }
  end

  def block(name, _opts), do: raise(ArgumentError, "unknown carrier block #{inspect(name)}")

  defp equalizer_block(name, o, gnuradio) do
    unless o[:num_taps] >= 2, do: raise(ArgumentError, "#{name}: num_taps must be at least 2")
    unless o[:algorithm] in [:lms, :nlms, :cma], do: raise(ArgumentError, "unknown algorithm")

    o =
      if o[:algorithm] == :cma and not Keyword.has_key?(o, :constellation),
        do: o,
        else: Keyword.put(o, :points, points(o[:constellation]))

    o = Keyword.delete(o, :constellation)
    training? = Keyword.get(o, :training, []) != []
    sps = o[:sps]

    %Block{
      name: name,
      kind: :stream,
      fun: &Equalizer.step/3,
      init: Equalizer.init(o),
      opts: o,
      rate: if(sps == 1, do: {1, 1}, else: {:variable, 1}),
      item: {if(training?, do: :complex_flag, else: :complex), out_item(:complex, o)},
      hardware: :feedback,
      warmup: :unbounded,
      gnuradio: gnuradio
    }
  end

  # -- helpers ---------------------------------------------------------------------------

  # `{item, extras}` as an atom, e.g. :complex_freq_phase
  defp out_item(base, opts) do
    case Keyword.get(opts, :outputs, []) do
      [] -> base
      names -> Enum.join([base | names], "_") |> String.to_atom()
    end
  end

  defp points(%Constellation{points: p}), do: p
  defp points(name) when is_atom(name), do: apply(Constellation, name, []).points
  defp points([{_, _} | _] = p), do: p

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
end
