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
            pfb_arb_resampler_ccf pfb_arb_resampler_fff clock_recovery_mm_cc clock_recovery_mm_ff
            symbol_sync_cc symbol_sync_ff pfb_clock_sync_ccf pfb_clock_sync_fff
            msk_timing_recovery_cc)a

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

  def block(name, opts) when name in [:clock_recovery_mm_cc, :clock_recovery_mm_ff] do
    f = format(opts)
    kind = kind(name)
    ip = Interp.mmse()
    omega = L.f32(Keyword.fetch!(opts, :omega))
    gain_omega = L.f32(Keyword.fetch!(opts, :gain_omega))
    gain_mu = L.f32(Keyword.fetch!(opts, :gain_mu))
    mu = L.f32(Keyword.get(opts, :mu, 0.5))
    rel = L.f32(Keyword.get(opts, :omega_relative_limit, 0.001))
    lim = L.f32(rel * omega)

    # smallest position step: omega at its limit, the error at its bound
    mm_max = if kind == :complex, do: 1.0, else: 2.0

    iterations =
      Keyword.get_lazy(opts, :iterations, fn ->
        steps_per_item(omega - lim - gain_mu * mm_max, name)
      end)

    z = L.zero(kind, f)
    hist = if kind == :complex, do: [z, z], else: [z]
    s = List.to_tuple([L.const(mu, f), L.const(omega, f) | hist])
    {lo, hi} = {omega - lim, omega + lim}

    bound = fn s ->
      [mu, om | h] = Tuple.to_list(s)

      List.to_tuple([
        L.bound(mu, 0, 1, f),
        L.bound(om, lo, hi, f) | Enum.map(h, &Num.quantize(&1, f))
      ])
    end

    machine_block(name, f, kind,
      fun: &K.clock_recovery_mm/3,
      # complex: history of 3 (two zero items in front), real: none
      init: K.init(kind, f, ip.ntaps, if(kind == :complex, do: ip.ntaps - 2, else: ip.ntaps), s),
      opts: [
        interp: ip,
        gain_mu: gain_mu,
        gain_omega: gain_omega,
        omega_mid: omega,
        omega_lim: lim,
        tap_frac: opts[:tap_frac]
      ],
      bound_state: bound,
      iterations: iterations,
      latency: 4,
      gnuradio: "digital.#{name}"
    )
  end

  def block(:msk_timing_recovery_cc = name, opts) do
    f = format(opts)
    ip = Interp.mmse()
    sps = L.f32(Keyword.fetch!(opts, :sps))
    gain = L.f32(Keyword.fetch!(opts, :gain))
    limit = L.f32(Keyword.fetch!(opts, :limit))
    osps = Keyword.get(opts, :osps, 1)
    true = osps in [1, 2]
    half = L.f32(sps / 2)
    gain_omega = L.f32(L.f32(gain * gain) * 0.25)

    iterations =
      Keyword.get_lazy(opts, :iterations, fn -> steps_per_item(half - limit - 3 * gain, name) end)

    w = wide(f, 4)
    z = L.zero(:complex, f)
    zw = L.zero(:complex, w)
    s = {L.const(0.5, f), L.const(half, f), z, zw, L.i32(0)}

    bound = fn {mu, om, c, nl, odd} ->
      {L.bound(mu, -1, 2, f), L.bound(om, half - limit, half + limit, f), Num.quantize(c, f),
       Num.quantize(nl, w), L.bound_int(odd, 0, 1)}
    end

    machine_block(name, f, :complex,
      fun: &K.msk_timing_recovery/3,
      init: K.init(:complex, f, ip.ntaps, ip.ntaps, s),
      opts: [
        interp: ip,
        gain: gain,
        gain_omega: gain_omega,
        omega_mid: half,
        limit: limit,
        osps: osps,
        wide: w,
        tap_frac: opts[:tap_frac]
      ],
      bound_state: bound,
      iterations: iterations,
      latency: 4,
      gnuradio: "digital.#{name}"
    )
  end

  def block(name, opts) when name in [:pfb_clock_sync_ccf, :pfb_clock_sync_fff] do
    f = format(opts)
    kind = kind(name)
    sps_real = Keyword.fetch!(opts, :sps) * 1.0
    sps = trunc(Float.floor(sps_real))
    bw = L.f32(Keyword.fetch!(opts, :loop_bw))
    taps = Enum.map(Keyword.fetch!(opts, :taps), &L.f32/1)
    nf = Keyword.get(opts, :filter_size, 32)
    k0 = L.f32(Keyword.get(opts, :init_phase, 0.0))
    max_dev = L.f32(Keyword.get(opts, :max_rate_deviation, 1.5))
    osps = Keyword.get(opts, :osps, 1)
    margin = Keyword.get(opts, :margin, 2)

    # GNU Radio: damping 2 * nfilters, then control-loop gains in f32
    damp = L.f32(2 * nf)
    denom = L.f32(1.0 + 2.0 * damp * bw + bw * bw)
    alpha = L.f32(L.f32(L.f32(4 * damp) * bw) / denom)
    beta = L.f32(L.f32(L.f32(4 * bw) * bw) / denom)
    rate = L.f32((sps_real - sps) * nf)
    rate_i = Float.floor(rate)
    rate_f = L.f32(rate - rate_i)

    ip = Interp.polyphase(taps, nf, clock_sync_diff(taps, nf))
    t = ip.ntaps
    extra = max(0, 2 * sps - osps - margin + 1)
    trigger = t - 1 + osps - 1 + margin + extra
    len = trigger + 1 + margin
    # history of t + 2 sps items: the first symbol starts at -(t + 2 sps - 1)
    wait0 = -(t + 2 * sps - 1) + trigger + 1
    iterations = Keyword.get(opts, :iterations, if(sps >= 2, do: 1, else: 2))
    w = wide(f, 4)

    bound = fn {k, r} -> {L.bound(k, -4 * nf, 5 * nf, f), L.bound(r, -max_dev, max_dev, f)} end

    machine_block(name, f, kind,
      fun: &K.pfb_clock_sync/3,
      init: K.init(kind, f, len, wait0, {L.const(k0, f), L.const(rate_f, f)}),
      opts: [
        interp: ip,
        sps: sps,
        osps: osps,
        rate_i: rate_i,
        alpha: alpha,
        beta: beta,
        max_dev: max_dev,
        margin: margin,
        wide: w,
        tap_frac: opts[:tap_frac]
      ],
      bound_state: bound,
      iterations: iterations,
      max: iterations * osps,
      latency: div(t, 2),
      gnuradio: "digital.#{name}"
    )
  end

  def block(name, opts) when name in [:symbol_sync_cc, :symbol_sync_ff] do
    f = format(opts)
    kind = kind(name)
    ted_type = Keyword.fetch!(opts, :ted_type)
    ted = Map.fetch!(K.teds(), ted_type)
    sps = L.f32(Keyword.fetch!(opts, :sps))
    bw = L.f32(Keyword.fetch!(opts, :loop_bw))
    zeta = L.f32(Keyword.get(opts, :damping_factor, 1.0))
    ted_gain = L.f32(Keyword.get(opts, :ted_gain, 1.0))
    max_dev = L.f32(Keyword.get(opts, :max_deviation, 1.5))
    osps = Keyword.get(opts, :osps, 1)
    true = sps > 1 and osps >= 1

    ip =
      case Keyword.get(opts, :resampling_type, :mmse) do
        :mmse -> Interp.mmse()
        :pfb_no_mf -> Interp.pfb_no_mf(Keyword.get(opts, :n_filters, 128))
        :pfb_mf -> Interp.pfb_mf(Keyword.fetch!(opts, :taps), Keyword.get(opts, :n_filters, 128))
      end

    {alpha, beta} = clock_loop_gains(bw, zeta, ted_gain)
    {max_p, min_p} = {L.f32(sps + max_dev), L.f32(sps - max_dev)}
    nom = if sps < min_p or sps > max_p, do: L.f32((max_p + min_p) / 2), else: sps

    ips = lcm(ted.inputs, osps)
    {ipt, ipo} = {div(ips, ted.inputs), div(ips, osps)}
    iterations = Keyword.get_lazy(opts, :iterations, fn -> max(1, ceil(2 * ips / min_p)) end)
    inst_lo = ips / iterations
    inst_hi = 2 * max_p
    la = if ted.lookahead, do: ceil(1 + ipt * inst_hi / ips), else: 0
    len = ip.ntaps + la

    {constellation, slicer} =
      slicer(Keyword.get(opts, :constellation, ExSdr.Digital.Constellation.bpsk()))

    w = wide(f, 4)
    df = wide(f, 3)
    z = L.zero(kind, f)
    xs = L.zero_line(kind, f, ted.depth)
    ds = if ted.slicer, do: L.zero_line(kind, df, ted.depth), else: L.zero_line(kind, f, 1)
    dxs = if ted.derivative, do: L.zero_line(kind, w, 1), else: L.zero_line(kind, f, 1)
    _ = z

    mu0 = L.f32(sps - Float.floor(sps))
    iip0 = L.f32(sps / ips)

    s =
      {L.const(mu0, f), L.i32(ips - 1), L.const(nom, f), L.const(nom, f), L.const(iip0, f), xs,
       ds, dxs}

    bound = fn {mu, ic, avg, inst, iip, xs, ds, dxs} ->
      {L.bound(mu, 0, 1, f), L.bound_int(ic, 0, ips - 1), L.bound(avg, min_p, max_p, f),
       L.bound(inst, inst_lo, inst_hi, f), L.bound(iip, 0, inst_hi, f), Num.quantize(xs, f),
       Num.quantize(ds, if(ted.slicer, do: df, else: f)),
       Num.quantize(dxs, if(ted.derivative, do: w, else: f))}
    end

    machine_block(name, f, kind,
      fun: &K.symbol_sync/3,
      init: K.init(kind, f, len, len, s),
      opts: [
        interp: ip,
        ted: ted_type,
        points: constellation,
        slicer: slicer,
        ips: ips,
        ipt: ipt,
        ipo: ipo,
        alpha: alpha,
        beta: beta,
        min_avg: min_p,
        max_avg: max_p,
        inst_lo: inst_lo,
        inst_hi: inst_hi,
        wide: w,
        dfmt: df,
        tap_frac: opts[:tap_frac]
      ],
      bound_state: bound,
      iterations: iterations,
      latency: div(ip.ntaps, 2),
      gnuradio: "digital.#{name}"
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

  # Iterations per input item for a smallest position step `step`.
  defp steps_per_item(step, name) do
    if step <= 0,
      do: raise(ArgumentError, "#{name}: the position can stand still; give :iterations"),
      else: trunc(1 / step) + 1
  end

  # A wider fixed format for products (identity for float).
  defp wide(%{kind: :float} = f, _), do: f
  defp wide(f, extra), do: %{f | bits: min(f.bits + extra, 40)}

  @doc """
  GNU Radio's `clock_tracking_loop` gains `{alpha, beta}` (proportional,
  integral) for a normalised loop bandwidth, damping factor and expected
  TED gain, computed with its f32 roundings.
  """
  def clock_loop_gains(bw, zeta, ted_gain) do
    # Second order loop, impulse invariant design (M. Rice, Digital
    # Communications: A Discrete-Time Approach, appendix C):
    #   alpha = (2 / Kted) e^(-zeta bw) sinh(zeta bw)
    #   beta  = (2 / Kted) (1 - e^(-zeta bw) (sinh(zeta bw) + c(bw)))
    # with c the cosh / 1 / cos term of the over-, critically or under-damped loop.
    scale = L.f32(2.0 / ted_gain)
    decay = L.f32(:math.exp(-L.f32(zeta * bw)))
    sh = L.f32(:math.sinh(L.f32(zeta * bw)))
    c = damping_term(bw, zeta)

    {L.f32(L.f32(scale * decay) * sh), L.f32(scale * L.f32(1.0 - L.f32(decay * L.f32(sh + c))))}
  end

  defp damping_term(_bw, zeta) when zeta == 1.0, do: 1.0

  defp damping_term(bw, zeta) when zeta > 1.0,
    do: L.f32(:math.cosh(L.f32(bw * L.f32(:math.sqrt(L.f32(zeta * zeta - 1.0))))))

  defp damping_term(bw, zeta),
    do: L.f32(:math.cos(L.f32(bw * L.f32(:math.sqrt(1.0 - zeta * zeta)))))

  # pfb_clock_sync's derivative taps: [0, taps[i + 2] - taps[i]..., 0] scaled
  # to an absolute sum of nfilts.
  defp clock_sync_diff(taps, nf) do
    t = List.to_tuple(taps)
    mid = for i <- 0..(tuple_size(t) - 3), do: L.f32(elem(t, i + 2) - elem(t, i))
    pwr = Enum.reduce(mid, 0.0, fn x, acc -> L.f32(acc + abs(x)) end)
    scale = L.f32(nf / pwr)
    Enum.map([0.0 | mid] ++ [0.0], &L.f32(&1 * scale))
  end

  # Slicer of a constellation: its points and :sign for BPSK/QPSK-like sets
  # (GNU Radio's sector decision), else :nearest.
  defp slicer(%ExSdr.Digital.Constellation{points: points}) do
    sign? =
      case points do
        [{a, z1}, {b, z2}] ->
          z1 == 0 and z2 == 0 and a == -b and a < 0

        [{a, b}, {c, d}, {e, g}, {h, i}] ->
          a < 0 and b < 0 and c == -a and d == b and e == a and g == -b and h == -a and i == -b

        _ ->
          false
      end

    {points, if(sign?, do: :sign, else: :nearest)}
  end

  defp lcm(a, b), do: div(a * b, Integer.gcd(a, b))

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
