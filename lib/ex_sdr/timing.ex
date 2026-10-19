defmodule ExSdr.Timing do
  @moduledoc """
  Symbol timing recovery and arbitrary resampling (GNU Radio's
  `mmse_resampler`, `pfb_arb_resampler`, `clock_recovery_mm`,
  `symbol_sync`, `pfb_clock_sync`, `msk_timing_recovery`) as
  `NxFixed.Block`s: `block(name, opts)`.

  Every block takes `:format` (`NxFixed.Num.float/0`, the default, or
  `NxFixed.Num.q/3`) and runs through `NxFixed.Block.run/4`; the state is
  explicit, so a stream cut anywhere gives the same output as one call.
  Except for `:fractional_delay_*` the blocks have `rate: {:variable, max}`:
  each input item gives 0..`max` outputs (see `ExSdr.Timing.Kernels` for
  how GNU Radio's position walk is driven by arriving items).

  Positions, fractional delays and loop filter state follow GNU Radio's
  arithmetic in the float instance (f32, or f64 where GNU Radio uses
  `double`) and are integer words with 32 fractional bits in fixed point
  (`ExSdr.Timing.Loop`), so fixed-point positions are exact and the
  blocks are `:feedback` hardware.

  Names carry GNU Radio's type suffix (`cc`: complex in and out, `ff`
  real, `ccf`/`fff` with real taps). docs/blocks/timing.md lists every
  block with its options and deviations from GNU Radio.

  Common options: `:format`, `:tap_frac` (fixed point: fractional bits of
  the interpolator taps, default the format's), `:iterations` (iterations
  per input item; the default covers the block's parameters).
  """
  @behaviour NxFixed.Block

  import Bitwise

  alias ExSdr.Timing.Interp
  alias ExSdr.Timing.Kernels, as: K
  alias ExSdr.Timing.Loop, as: L
  alias NxFixed.Block
  alias NxFixed.Num

  @names ~w(fractional_delay_cc fractional_delay_ff mmse_resampler_cc mmse_resampler_ff
            pfb_arb_resampler_ccf pfb_arb_resampler_fff)a

  @doc "Block names `block/2` knows."
  def names, do: @names

  @impl NxFixed.Block
  def block(name, opts \\ [])

  # Fixed fractional delay: the interpolator at a constant mu.
  def block(name, opts) when name in [:fractional_delay_cc, :fractional_delay_ff] do
    f = format(opts)
    kind = kind(name)
    ip = interpolator(Keyword.get(opts, :interpolator, :mmse))
    mu = Keyword.get(opts, :mu, 0.0)
    true = mu >= 0 and mu <= 1
    arm = rint(L.f32(mu) * ip.arms)

    %Block{
      name: name,
      kind: :stream,
      fun: &K.fractional_delay/3,
      init: L.zero_line(kind, f, ip.ntaps - 1),
      opts: [format: f, interp: ip, arm: arm, tap_frac: opts[:tap_frac]],
      latency: div(ip.ntaps, 2),
      warmup: ip.ntaps - 1,
      item: {kind, kind},
      hardware: :feed_forward,
      gnuradio: "filter.mmse_resampler_#{suffix(name)} (resampling ratio 1)"
    }
  end

  def block(name, opts) when name in [:mmse_resampler_cc, :mmse_resampler_ff] do
    f = format(opts)
    kind = kind(name)
    ip = Interp.mmse()
    # GNU Radio: float arguments, kept as double
    ratio = L.f32(Keyword.fetch!(opts, :resamp_ratio))
    mu0 = L.f32(Keyword.get(opts, :phase_shift, 0.0))
    true = ratio > 0 and mu0 >= 0 and mu0 <= 1
    iterations = Keyword.get_lazy(opts, :iterations, fn -> trunc(1 / ratio) + 1 end)

    step = L.const(ratio, f, type: :f64)
    mu = L.const(mu0, f, type: :f64)

    machine_block(name, f, kind,
      fun: &K.mmse_resampler/3,
      init: K.init(kind, f, ip.ntaps, ip.ntaps, mu),
      opts: [interp: ip, step: step, tap_frac: opts[:tap_frac]],
      bound_state: fn mu -> L.bound(mu, 0, 1, f) end,
      iterations: iterations,
      latency: round((3 + mu0) / ratio),
      gnuradio: "filter.#{name}"
    )
  end

  def block(name, opts) when name in [:pfb_arb_resampler_ccf, :pfb_arb_resampler_fff] do
    f = format(opts)
    kind = kind(name)
    rate = L.f32(Keyword.fetch!(opts, :rate))
    taps = Keyword.fetch!(opts, :taps)
    n = Keyword.get(opts, :filter_size, 32)
    # GNU Radio: unsigned / float in f32
    q = L.f32(n / rate)
    dec = trunc(Float.floor(q))
    flt = L.f32(q - dec)
    if dec < 1, do: raise(ArgumentError, "#{name}: rate above filter_size is not supported")

    diff =
      Enum.zip_with(taps, tl(taps) ++ [nil], fn a, b ->
        if b, do: L.f32(L.f32(b) - L.f32(a)), else: 0.0
      end)

    ip = Interp.polyphase(taps, n, diff)
    iterations = Keyword.get_lazy(opts, :iterations, fn -> ceil(n / dec) end)
    j0 = rem(div(length(taps), 2), n)

    machine_block(name, f, kind,
      fun: &K.pfb_arb_resampler/3,
      # history of `ntaps` items: the first output is on the first input item
      init: K.init(kind, f, ip.ntaps, 1, {L.i32(j0), L.const(0.0, f)}),
      opts: [
        interp: ip,
        dec_rate: L.i32(dec),
        flt_rate: L.const(flt, f),
        tap_frac: opts[:tap_frac],
        mu_bits: opts[:mu_bits]
      ],
      bound_state: fn {j, acc} -> {L.bound_int(j, 0, n - 1), L.bound(acc, 0, 1, f)} end,
      iterations: iterations,
      latency: round(rate * (ip.ntaps - 1) / 2),
      gnuradio: "filter.#{name}"
    )
  end

  def block(name, _opts), do: raise(ArgumentError, "unknown timing block #{inspect(name)}")

  # -- helpers ---------------------------------------------------------------------------

  defp machine_block(name, f, kind, b) do
    iterations = b[:iterations]
    max = Keyword.get(b, :max, iterations)
    out = Keyword.get(b, :out, kind)

    %Block{
      name: name,
      kind: :stream,
      fun: b[:fun],
      init: b[:init],
      opts:
        [format: f, iterations: iterations, max: max, out: out, bound_state: b[:bound_state]] ++
          b[:opts],
      rate: {:variable, max},
      latency: b[:latency] || 0,
      warmup: :unbounded,
      item: {kind, out},
      hardware: :feedback,
      gnuradio: b[:gnuradio]
    }
  end

  @doc """
  An interpolator from an option: `:mmse`, `{:pfb_no_mf, nfilts}`,
  `{:pfb_mf, taps, nfilts}`, `{:polyphase, taps, nfilts}` or an
  `ExSdr.Timing.Interp`.
  """
  def interpolator(:mmse), do: Interp.mmse()
  def interpolator({:pfb_no_mf, n}), do: Interp.pfb_no_mf(n)
  def interpolator({:pfb_mf, taps, n}), do: Interp.pfb_mf(taps, n)
  def interpolator({:polyphase, taps, n}), do: Interp.polyphase(taps, n)
  def interpolator(%Interp{} = ip), do: ip

  defp format(opts), do: Keyword.get(opts, :format, Num.float())

  defp suffix(name) do
    name |> Atom.to_string() |> String.split("_") |> List.last()
  end

  defp kind(name) do
    case suffix(name) do
      "c" <> _ -> :complex
      "f" <> _ -> :real
    end
  end

  # C rint: ties to even
  defp rint(x) do
    fl = Float.floor(x)
    d = x - fl

    cond do
      d > 0.5 -> trunc(fl) + 1
      d < 0.5 -> trunc(fl)
      rem(trunc(fl), 2) == 0 -> trunc(fl)
      true -> trunc(fl) + 1
    end
  end

  @doc false
  def one, do: 1 <<< L.p()
end
