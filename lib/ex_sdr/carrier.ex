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

  alias ExSdr.Carrier.Kernels, as: K
  alias ExSdr.Carrier.Loop
  alias ExSdr.Digital.Constellation
  alias NxFixed.Block
  alias NxFixed.Num

  @names ~w(costas_loop_cc fll_band_edge_cc constellation_receiver_cb)a

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

  def block(name, _opts), do: raise(ArgumentError, "unknown carrier block #{inspect(name)}")

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
