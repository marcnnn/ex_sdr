defmodule ExSdr.Analog.Kernels do
  @moduledoc """
  The stream kernels of `ExSdr.Analog`: `step(sample, state, opts) ->
  {output, next_state}`, written once against `NxFixed.Num` and
  `ExSdr.Analog.Math`, instantiated by `opts[:format]`.

  Constants (gains, loop filter coefficients, thresholds) are options, so
  they are compile-time constants of the kernel, like FIR taps. Complex
  samples are `{re, im}`. Phases and frequencies are 32 bit angle words
  (`ExSdr.Analog.Math`).
  """
  alias NxFixed.Num
  import ExSdr.Analog.Math

  @two_pi 2 * :math.pi()
  @turn 4_294_967_296.0

  # A state value is always inside its format by construction; `bound` says so
  # to the width check (state parameters otherwise span the whole container
  # type). It costs nothing in hardware.
  defp bound(x, format), do: quantize(x, format)

  # -- modulators -------------------------------------------------------------------------

  @doc """
  `frequency_modulator_fc`: `phase += sensitivity * x; out = exp(j phase)`.
  State: the phase word (s32). Options: `:format`, `:sensitivity` (rad per
  unit input).
  """
  def frequency_modulator(x, phase, opts) do
    format = opts[:format]
    x = bound(x, format)
    next = fm_phase(x, phase, opts[:sensitivity], format)
    {sincos_word(next, format), next}
  end

  defp fm_phase(x, phase, sensitivity, format) do
    inc = words_from_real(x, sensitivity / @two_pi * @turn, format)
    wrap32(Nx.add(Nx.as_type(phase, :s64), inc))
  end

  @doc "`phase_modulator_fc`: `out = exp(j sensitivity x)`, stateless."
  def phase_modulator(x, state, opts) do
    format = opts[:format]
    x = bound(x, format)
    w = words_from_real(x, opts[:sensitivity] / @two_pi * @turn, format)
    {sincos_word(w, format), state}
  end

  # -- demodulators ---------------------------------------------------------------------------

  @doc """
  `quadrature_demod_cf`: `gain * arg(x[n] * conj(x[n-1]))`. State: the
  previous sample (zero at the start, so the first output is 0 like GNU
  Radio). Options: `:format`, `:gain`, and for fixed point `:iterations`
  and `:guard` of the CORDIC.
  """
  def quadrature_demod(x, prev, opts) do
    x = bound(x, opts[:format])
    {quad_demod_out(x, bound(prev, opts[:format]), opts), x}
  end

  defp quad_demod_out({ar, ai}, {br, bi}, opts) do
    format = opts[:format]
    pre = Num.add(Num.mul(ar, br), Num.mul(ai, bi))
    pim = Num.sub(Num.mul(ai, br), Num.mul(ar, bi))
    w = atan2_word(pim, pre, format, Keyword.take(opts, [:iterations, :guard]))
    real_from_words(w, opts[:gain] * rad_per_word(), format)
  end

  @doc """
  `complex_to_mag` followed by `add_const(-offset)`, the AM detector of
  `am_demod_cf` (offset 1.0: removes the carrier). Stateless. Options:
  `:format`, `:offset`, `:guard` (fractional bits of the square root, default 4).
  """
  def am_demod(x, state, opts) do
    format = opts[:format]
    x = bound(x, format)
    guard = Keyword.get(opts, :guard, 4)
    mag = cmag(x, format, format.bits || 32, guard)
    {quantize(Num.sub(mag, const_like(opts[:offset], mag)), format), state}
  end

  # -- emphasis ---------------------------------------------------------------------------------

  @doc """
  One IIR section `y[n] = b0 x[n] + b1 x[n-1] + p1 y[n-1]` (what GNU Radio's
  `fm_deemph` and `fm_preemph` build with `iir_filter_ffd`). State:
  `{x[n-1], y[n-1]}`, y with `:guard` more fractional bits than the format
  (default 8), so a pole near 1 has no dead band. Options: `:format`, `:b0`,
  `:b1`, `:p1`, `:guard`.
  """
  def iir1(x, {x1, y1}, opts) do
    x = bound(x, opts[:format])
    x1 = bound(x1, opts[:format])
    {y, y_wide} = iir1_out(x, x1, y1, opts)
    {y, {x, y_wide}}
  end

  defp iir1_out(x, x1, y1, opts) do
    format = opts[:format]
    wide = widen(format, Keyword.get(opts, :guard, 8))
    y1 = bound(y1, wide)

    acc =
      Num.add(
        Num.add(Num.mul(x, wconst(opts[:b0], format)), Num.mul(x1, wconst(opts[:b1], format))),
        Num.mul(y1, wconst(opts[:p1], format))
      )

    y_wide = quantize(acc, wide)
    {quantize(y_wide, format), y_wide}
  end

  @doc "State of `iir1/3` at rest."
  def iir1_init(format, guard \\ 8),
    do: {Num.zeros({}, format), Num.zeros({}, widen(format, guard))}

  @doc "`nbfm_tx` without the interpolator: `fm_preemph` into `frequency_modulator_fc`."
  def nbfm_tx(x, {iir, phase}, opts) do
    {y, iir_next} = iir1(x, iir, opts)
    {out, phase_next} = frequency_modulator(y, phase, opts)
    {out, {iir_next, phase_next}}
  end

  @doc "`nbfm_rx` without the audio filter: `quadrature_demod_cf` into `fm_deemph`."
  def nbfm_rx_core(x, {prev, iir}, opts) do
    {d, prev_next} = quadrature_demod(x, prev, opts)
    {y, iir_next} = iir1(d, iir, opts)
    {y, {prev_next, iir_next}}
  end

  # -- gain control -----------------------------------------------------------------------------

  @doc """
  The gain format of the AGCs: float, or a signed fixed-point format with
  room for `0..max_gain` and as many fractional bits as the product with a
  sample leaves in 62 bits (at most 28).
  """
  def gain_format(format, max_gain) do
    case format.kind do
      :float ->
        format

      :fixed ->
        lg = ceil(:math.log2(max_gain + 1))
        frac = min(28, 61 - format.bits - lg)
        wide_format(format, lg + frac + 1, frac)
    end
  end

  @doc "Initial AGC gain as a state value."
  def gain_init(gain, format, max_gain) do
    gf = gain_format(format, max_gain)

    case gf.kind do
      :float ->
        Nx.tensor(gain * 1.0, type: :f32)

      :fixed ->
        %NxFixed.Fixed{t: Nx.tensor(to_raw(gain, gf.frac, gf.bits), type: :s64), frac: gf.frac}
    end
  end

  # Output, and the magnitude of the unsaturated output (up to 16 full
  # scales) at binary point frac + guard.
  defp agc_apply(x, gain, opts) do
    format = opts[:format]
    guard = Keyword.get(opts, :guard, 4)

    wide_f =
      case format.kind do
        :float -> format
        :fixed -> wide_format(format, format.bits + guard + 4, format.frac + guard)
      end

    case x do
      {re, im} ->
        prod = {Num.mul(re, gain), Num.mul(im, gain)}
        {quantize(prod, format), cmag(quantize(prod, wide_f), format, wide_f.bits || 0, 0)}

      re ->
        prod = Num.mul(re, gain)
        {quantize(prod, format), rmag(quantize(prod, wide_f))}
    end
  end

  defp agc_clamp(g, opts) do
    gf = gain_format(opts[:format], opts[:max_gain])
    quantize(clamp(g, 0.0, opts[:max_gain] * 1.0, gf.bits), gf)
  end

  @doc """
  `agc_cc` and `agc_ff`: `out = x * gain; gain += rate * (reference - |out|)`,
  the gain limited to `0..max_gain`. State: the gain (`gain_init/3`). Options:
  `:format`, `:rate`, `:reference`, `:max_gain`, `:guard`.
  """
  def agc(x, gain, opts) do
    format = opts[:format]
    gf = gain_format(format, opts[:max_gain])
    gain = bound(gain, gf)
    {y, mag} = agc_apply(bound(x, format), gain, opts)
    err = Num.sub(const_like(opts[:reference], mag), mag)
    delta = quantize(Num.mul(err, wconst(opts[:rate], format)), gf)
    {y, agc_clamp(Num.add(gain, delta), opts)}
  end

  @doc """
  `agc2_cc` and `agc2_ff`: as `agc/3` with separate rates, chosen as GNU
  Radio does: `e = |out| - reference; rate = e > gain ? attack : decay;
  gain -= e * rate` (the comparison with the gain is in GNU Radio's source).
  """
  def agc2(x, gain, opts) do
    format = opts[:format]
    gf = gain_format(format, opts[:max_gain])
    gain = bound(gain, gf)
    {y, mag} = agc_apply(bound(x, format), gain, opts)
    e = Num.sub(mag, const_like(opts[:reference], mag))
    d_attack = quantize(Num.mul(e, wconst(opts[:attack_rate], format)), gf)
    d_decay = quantize(Num.mul(e, wconst(opts[:decay_rate], format)), gf)
    delta = sel(gt(e, gain), d_attack, d_decay)
    {y, agc_clamp(Num.sub(gain, delta), opts)}
  end

  @doc """
  `agc3_cc`: the gain follows the ideal gain `reference / |x|` of the input
  through a single pole with `attack_rate` while the output is above the
  reference and `decay_rate` while it is below:

      g = g * (1 - rate) + rate * reference / |x|     (every iir_update_decim samples)
      out = x * g

  GNU Radio starts with `g = reference / mean(|x|)` over the first
  `4 * iir_update_decim` samples (a block of that size is also GNU Radio's
  output multiple) and updates from the next one on; the first block uses
  that gain unchanged. Here the start gain is the `:gain` option (compute
  it with `ExSdr.Analog.agc3_initial_gain/3`), and the kernel counts the first
  block in its state `{gain, warm, phase}`. Options: `:format`,
  `:attack_rate`, `:decay_rate`, `:reference`, `:iir_update_decim`,
  `:max_gain`, `:guard`.
  """
  def agc3(x, {gain, warm, phase}, opts) do
    format = opts[:format]
    gf = gain_format(format, opts[:max_gain])
    d = opts[:iir_update_decim]
    gain = bound(gain, gf)
    warm = Nx.clip(warm, 0, 4 * d)
    phase = Nx.clip(phase, 0, d - 1)
    {re, im} = bound(x, format)

    mag = cmag({re, im}, format, format.bits || 0, Keyword.get(opts, :guard, 4))
    q = ideal_gain(const_like(opts[:reference], mag), mag, opts)
    d_attack = quantize(Num.mul(Num.sub(q, gain), wconst(opts[:attack_rate], format, 16)), gf)
    d_decay = quantize(Num.mul(Num.sub(q, gain), wconst(opts[:decay_rate], format, 16)), gf)
    delta = sel(lt(q, gain), d_attack, d_decay)
    g_new = agc_clamp(Num.add(gain, delta), opts)

    update = Nx.logical_and(Nx.greater_equal(warm, 4 * d), Nx.equal(phase, 0))
    g = sel(update, g_new, gain)
    y = quantize({Num.mul(re, g), Num.mul(im, g)}, format)
    {y, {g, Nx.min(Nx.add(warm, 1), 4 * d), Nx.remainder(Nx.add(phase, 1), d)}}
  end

  # reference / magnitude as a gain value, limited to max_gain (0 magnitude gives max_gain).
  defp ideal_gain(%NxFixed.Fixed{t: ref, frac: frac}, %NxFixed.Fixed{t: mag, frac: frac}, opts) do
    gf = gain_format(opts[:format], opts[:max_gain])
    num = ref |> Nx.as_type(:s64) |> Nx.left_shift(gf.frac)
    q = Nx.quotient(num, Nx.max(Nx.as_type(mag, :s64), 1))
    quantize(clamp(%NxFixed.Fixed{t: q, frac: gf.frac}, 0.0, opts[:max_gain] * 1.0, gf.bits), gf)
  end

  defp ideal_gain(ref, mag, opts),
    do: Nx.clip(Nx.divide(ref, Nx.max(mag, 1.0e-30)), 0.0, opts[:max_gain] * 1.0)

  @doc """
  `feedforward_agc_cc`: output delayed by `n - 1` samples; each output is the
  oldest sample of a window of `n` scaled by `reference / envelope`, the
  envelope being the maximum over the window of `max(|re|, |im|) + 0.4
  min(|re|, |im|)` (what GNU Radio computes; zero outputs for the first
  `n - 1` samples). State: the `n - 1` previous samples. Options: `:format`,
  `:n`, `:reference`.
  """
  def feedforward_agc(x, lines, opts) do
    n = opts[:n]
    {re, im} = bound(x, opts[:format])
    {re_line, im_line} = bound(lines, opts[:format])
    re_w = prepend(re, re_line)
    im_w = prepend(im, im_line)
    out = ff_scale(re_w, im_w, n, opts)

    {out, {take_first(re_w, n - 1), take_first(im_w, n - 1)}}
  end

  defp ff_scale(re_w, im_w, n, opts) do
    format = opts[:format]
    ref = opts[:reference]
    old = {last(re_w, n), last(im_w, n)}

    case format.kind do
      :float ->
        a = Nx.abs(re_w)
        b = Nx.abs(im_w)
        env = Nx.add(Nx.max(a, b), Nx.multiply(Nx.min(a, b), 0.4)) |> max_reduce()
        gain = Nx.select(Nx.greater(env, 0), Nx.divide(ref, Nx.max(env, 1.0e-30)), 0.0)
        {Nx.multiply(elem(old, 0), gain), Nx.multiply(elem(old, 1), gain)}

      :fixed ->
        a = re_w |> raw() |> Nx.abs()
        b = im_w |> raw() |> Nx.abs()
        # 0.4 as 13107 / 2^15
        env =
          Nx.add(
            Nx.max(a, b),
            NxFixed.Primitives.shift_round(Nx.multiply(Nx.min(a, b), 13107), 15)
          )

        env = max_reduce(env)
        shift = 24
        ref_raw = to_raw(ref, re_w.frac, 40)
        gain = Nx.quotient(Nx.left_shift(Nx.tensor(ref_raw, type: :s64), shift), Nx.max(env, 1))
        gain = %NxFixed.Fixed{t: gain, frac: shift}
        {re_o, im_o} = old
        quantize({Num.mul(re_o, gain), Num.mul(im_o, gain)}, format)
    end
  end

  # -- squelch ---------------------------------------------------------------------------------

  @doc "The power average format of the squelch (fixed: 8 more fractional bits than the square)."
  def power_format(format) do
    case format.kind do
      :float -> format
      :fixed -> wide_format(format, 2 * format.bits + 8, 2 * format.frac + 8)
    end
  end

  @doc """
  `simple_squelch_cc`, `pwr_squelch_cc` and `pwr_squelch_ff` without ramp and
  gate: `avg += alpha * (|x|^2 - avg)`, the output is 0 while `avg <
  threshold`. State: `avg`. Options: `:format`, `:alpha`, `:threshold`
  (linear power, `10^(dB/10)`).
  """
  def squelch(x, avg, opts) do
    format = opts[:format]
    pf = power_format(format)
    avg = bound(avg, pf)
    x = bound(x, format)

    p =
      case x do
        {re, im} -> Num.add(Num.mul(re, re), Num.mul(im, im))
        re -> Num.mul(re, re)
      end

    delta = quantize(Num.mul(Num.sub(p, avg), wconst(opts[:alpha], format, 16)), pf)
    next = quantize(Num.add(avg, delta), pf)
    mute = lt(next, const_like(opts[:threshold], next))
    {zero_where(mute, x), next}
  end

  # -- rail, hold ------------------------------------------------------------------------------

  @doc "`rail_ff`: clips to `lo..hi` (representable in the format). Stateless."
  def rail(x, state, opts) do
    format = opts[:format]
    x = bound(x, format)

    case x do
      %NxFixed.Fixed{t: t, frac: frac} ->
        lo = to_raw(opts[:lo], frac, format.bits)
        hi = to_raw(opts[:hi], frac, format.bits)
        {%NxFixed.Fixed{t: Nx.clip(t, lo, hi), frac: frac}, state}

      _ ->
        {Nx.clip(x, opts[:lo], opts[:hi]), state}
    end
  end

  @doc """
  `sample_and_hold_ff`: the sample is `{data, control}`; the held value is
  replaced by `data` while `control` is not 0, and is the output.
  """
  def sample_and_hold({data, control}, held, opts) do
    data = bound(data, opts[:format])
    held = bound(held, opts[:format])

    take =
      case control do
        %NxFixed.Fixed{t: t} -> Nx.not_equal(t, 0)
        t -> Nx.not_equal(t, 0)
      end

    next = sel(take, data, held)
    {next, next}
  end

  # -- phase locked loops -----------------------------------------------------------------------

  @doc "Loop gains `{alpha, beta}` from the loop bandwidth, as gr::blocks::control_loop (damping sqrt(2)/2)."
  def loop_gains(loop_bw, damping \\ :math.sqrt(2) / 2) do
    denom = 1.0 + 2.0 * damping * loop_bw + loop_bw * loop_bw
    {4 * damping * loop_bw / denom, 4 * loop_bw * loop_bw / denom}
  end

  # The common part: nco at the current phase, x * conj(nco) and the loop update.
  defp pll_step(x, {phase, freq}, opts) do
    format = opts[:format]
    {xr, xi} = bound(x, format)
    {c, s} = sincos_word(phase, format)
    re = Num.add(Num.mul(xr, c), Num.mul(xi, s))
    im = Num.sub(Num.mul(xi, c), Num.mul(xr, s))
    err = atan2_word(im, re, format, Keyword.take(opts, [:iterations, :guard]))

    {alpha, beta} = loop_gains(opts[:loop_bw])
    freq1 = Nx.add(Nx.as_type(freq, :s64), mul_real(err, beta))
    phase1 = wrap32(Nx.add(Nx.add(Nx.as_type(phase, :s64), freq1), mul_real(err, alpha)))
    hi = min(round(opts[:max_freq] * words_per_rad()), 2_147_483_647)
    lo = max(round(opts[:min_freq] * words_per_rad()), -2_147_483_648)
    freq2 = freq1 |> Nx.clip(i64(lo), i64(hi)) |> Nx.as_type(:s32)
    {{c, s}, {re, im}, {phase1, freq2}}
  end

  @doc """
  `pll_carriertracking_cc`: output `x * conj(nco)`, the carrier removed.
  State: `{phase, freq}` words (s32). Options: `:format`, `:loop_bw`,
  `:max_freq`, `:min_freq` (rad per sample as in GNU Radio).
  """
  def pll_carriertracking(x, state, opts) do
    {_nco, prod, next} = pll_step(x, state, opts)
    {quantize(prod, opts[:format]), next}
  end

  @doc "`pll_refout_cc`: output the NCO `exp(j phase)` (before the update)."
  def pll_refout(x, state, opts) do
    {nco, _prod, next} = pll_step(x, state, opts)
    {nco, next}
  end

  @doc """
  `pll_freqdet_cf`: output the loop's frequency estimate (before the update),
  `:out_gain` (default 1) times radians per sample.
  """
  def pll_freqdet(x, {_phase, freq} = state, opts) do
    {_nco, _prod, next} = pll_step(x, state, opts)
    k = Keyword.get(opts, :out_gain, 1.0) * rad_per_word()
    {real_from_words(Nx.as_type(freq, :s64), k, opts[:format]), next}
  end
end
