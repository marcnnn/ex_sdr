defmodule ExSdr.Carrier.Estimators do
  @moduledoc """
  Correlation and estimators of gr-digital as stream kernels: `corr_est_cc`,
  the MPSK SNR estimators (`mpsk_snr_est_cc`, `probe_mpsk_snr_est_c`),
  `probe_density_b` and `probe_signal`.

  GNU Radio reports these through stream tags and message ports. Here they
  are output streams (what the tag would be at that item, or the
  estimator's registers after each item); the conversion to a value in dB
  needs a logarithm and a division, which belong on the host at tag or message rate:
  `snr_db/3` and `signal_noise_db/3`.
  """
  import Bitwise
  import ExSdr.Analog.Math

  alias ExSdr.Carrier.Kernels, as: K
  alias NxFixed.Fixed
  alias NxFixed.Num
  alias NxFixed.Num.Format

  # -- corr_est_cc ---------------------------------------------------------------------

  @doc """
  The correlation format of `corr_est_cc`: float, or 28 bits with the
  fractional bits that leave room for `sum |s| * 1`.
  """
  def corr_format(%Format{kind: :float} = f, _symbols), do: f

  def corr_format(%Format{} = f, symbols) do
    gain = Enum.sum(for {re, im} <- symbols, do: abs(re) + abs(im))
    int_bits = ceil(:math.log2(gain + 1.0))
    wide_format(f, 28, 27 - int_bits)
  end

  @doc """
  `corr_est_cc` without its tags, one item per call. The correlator is a
  complex FIR with the time reversed conjugate of `:symbols`:
  `corr[n] = sum_m conj(s[m]) x[n - N + 1 + m]`. A peak is the sample
  `p` with `|corr[p]|^2 > threshold * (sum |s|^2)^2` that is not smaller
  than its successor, at least `round(sps)` samples after the previous peak
  (GNU Radio's `THRESHOLD_ABSOLUTE`). Once `corr[p + 1]` is known the item
  reports it, so the detection of peak `p` is at stream index `p + 1`.

  Output `{delayed, corr, detect, phase, time_est, corr_est}`:

    * `delayed` - the input delayed by N, as GNU Radio's primary output;
    * `corr` - the correlation;
    * `detect` - `u8`, 1 when the sample before this one was a peak;
    * `phase` - angle word of `corr[p]` (`phase_est` tag);
    * `time_est` - `(m[p+1] - m[p-1]) / (m[p-1] + m[p] + m[p+1])` with
      `m = |corr|^2`, GNU Radio's `time_est` (centre of mass minus 2);
      Fixed with 22 fractional bits, in -1..1;
    * `corr_est` - `m[p]` (`corr_est` tag), a value of its own binary point.

  GNU Radio puts `corr_start` at stream index `p` and the other tags at
  `p + mark_delay` of its delayed output, so a consumer shifts by
  `mark_delay - 1` from the item that reports. The `amp_est` tag (the
  reciprocal of the largest input magnitude of the work call) depends on
  the scheduler's buffer sizes and is not produced.

  Options: `:format`, `:symbols` (`{re, im}` floats), `:sps`, `:threshold`,
  `:tap_frac` (20), `:iterations`, `:guard` (CORDIC for the phase).
  State: `{line, c1, m1, m2, skip}`.
  """
  def corr_est(x, {line, c1, m1, m2, skip}, opts) do
    f = opts[:format]
    symbols = opts[:symbols]
    n = length(symbols)
    tf = Keyword.get(opts, :tap_frac, 20)
    cf = corr_format(f, symbols)
    x = K.bound(x, f)

    line = K.cmap(line, &quantize(&1, f))
    delayed = K.cmap(line, &K.first/1)
    win = K.cmap2(line, x, &K.append/2)
    next_line = K.cmap(win, &K.drop_first/1)

    taps = for {re, im} <- Enum.reverse(symbols), do: {re, -im}
    # the window of the last N samples is `win` without its first item
    win_n = K.cmap(win, &K.drop_first/1)
    acc = ExSdr.Filters.Kernels.apply_taps(win_n, taps, n - 1, 1, 1, f, tf)
    corr = quantize(K.cmap(acc, &K.scalar/1), cf)

    {c1, m1, m2} = {bound_c(c1, cf), bound_m(m1, cf), bound_m(m2, cf)}
    m0 = power(corr)

    isps = trunc(opts[:sps] + 0.5)
    thr = threshold(symbols, opts[:threshold])
    skip = Nx.clip(skip, 0, max(isps - 1, 0))
    above = gt(m1, const_like(thr, m1))
    not_smaller = Nx.logical_not(lt(m1, m0))
    detect = Nx.logical_and(Nx.equal(skip, 0), Nx.logical_and(above, not_smaller))

    skip_next = Nx.select(detect, isps - 1, Nx.max(Nx.subtract(skip, 1), 0))

    phase = atan2_word(elem(c1, 1), elem(c1, 0), f, Keyword.take(opts, [:iterations, :guard]))
    phase = Nx.as_type(wrap32(phase), :s32)
    time_est = centre(m2, m1, m0, f)

    out = {delayed, corr, Nx.as_type(detect, :u8), phase, time_est, m1}

    {out, {next_line, corr, m0, m1, Nx.as_type(skip_next, :s32)}}
  end

  @doc "The state of `corr_est/3`."
  def corr_init(opts) do
    f = opts[:format]
    n = length(opts[:symbols])
    cf = corr_format(f, opts[:symbols])
    z = fn len, fmt -> {Num.zeros({len}, fmt), Num.zeros({len}, fmt)} end
    zero = fn -> Num.zeros({}, cf) end
    {z.(n, f), {zero.(), zero.()}, mag_zero(cf), mag_zero(cf), Nx.tensor(0, type: :s32)}
  end

  defp mag_zero(%Format{kind: :float} = f), do: Num.zeros({}, f)

  defp mag_zero(%Format{} = cf),
    do: %Fixed{t: Nx.tensor(0, type: :s64), frac: 2 * cf.frac}

  defp bound_c(c, cf), do: quantize(c, cf)

  defp bound_m(%Fixed{t: t, frac: frac}, cf) do
    # |corr|^2 of values within the format
    hi = 2 * (1 <<< (cf.bits - 1)) * (1 <<< (cf.bits - 1))
    %Fixed{t: t |> Nx.as_type(:s64) |> Nx.clip(i64(0), i64(hi)), frac: frac}
  end

  defp bound_m(m, _cf), do: m

  defp power({re, im}), do: Num.add(Num.mul(re, re), Num.mul(im, im))

  defp threshold(symbols, t) do
    auto = Enum.sum(for {re, im} <- symbols, do: re * re + im * im)
    t * auto * auto
  end

  # (m0 - m2) / (m2 + m1 + m0): float division, or 22 fractional bits of integer division
  defp centre(m2, m1, m0, %Format{kind: :float}) do
    den = Nx.max(Nx.add(Nx.add(m2, m1), m0), 1.0e-30)
    Nx.divide(Nx.subtract(m0, m2), den)
  end

  defp centre(m2, m1, m0, %Format{kind: :fixed}) do
    num = Num.sub(m0, m2)
    den = Num.add(Num.add(m2, m1), m0)
    # den has up to 56 bits: drop the low 23 to keep the shifted numerator in 63
    num_s = Nx.right_shift(Nx.as_type(num.t, :s64), 23)
    den_s = Nx.max(Nx.right_shift(Nx.as_type(den.t, :s64), 23), 1)
    q = Nx.quotient(Nx.left_shift(num_s, 22), den_s)
    %Fixed{t: Nx.clip(q, i64(-(1 <<< 22)), i64(1 <<< 22)) |> Nx.as_type(:s32), frac: 22}
  end

  # -- SNR estimators ---------------------------------------------------------------------

  @doc """
  The registers of the MPSK SNR estimators of `mpsk_snr_est`, updated each
  item. GNU Radio's blocks keep one item of history, so the estimators
  see the input delayed by one (a zero first); this is reproduced.

  `:type`:

    * `:simple` - `{y1, y2, count}`: GNU Radio's running "mean" of `|z|`, its
      sum of squared deviations and the item count. The recurrence is
      `m = y1 + (x - y1) / count; y2 += (x - y1) (x - m); y1 = m` with the
      count *before* the item, so `y1` is not the exact mean (and the
      estimate is biased); this is reproduced;
    * `:m2m4` - `{y1, y2}`: exponential averages (`alpha`) of `|z|^2` and `|z|^4`;
    * `:svr` - `{y1, y2}`: of `|x|^2 |x_prev|^2` and `|x|^4`;
    * `:skew` - `{y1, y2, y3}`: of `|z|`, `Re(z^2)` and the skewness
      register, with the index `i` of the sample since the start (GNU Radio uses
      the index within the work call, which depends on its buffer sizes).

  Fixed point: the exponential averages have 40 fractional bits and 4
  integer bits; `:simple` has `y1` at 40 and `y2` at 32 fractional bits
  (saturating at 2^62) and takes `|z|` with `guard` extra fractional bits.
  Division by the count is an integer quotient (truncating toward zero). Output: the registers after the item (see
  `snr_db/3`). State: `{prev, registers}`.

  Options: `:format`, `:type`, `:alpha` (0.001), `:guard` (4).
  """
  def snr_est(x, {prev, regs}, opts) do
    f = opts[:format]
    x = K.bound(x, f)
    prev = K.bound(prev, f)
    type = opts[:type]
    z = if type == :svr, do: x, else: prev
    zm = if type == :svr, do: prev, else: z
    regs = update_regs(type, z, zm, regs, opts)
    {regs, {x, regs}}
  end

  @doc "Registers and state of `snr_est/3`."
  def snr_init(opts) do
    f = opts[:format]
    z = Num.zeros({}, f)
    {{z, z}, zero_regs(opts[:type], f, opts)}
  end

  defp reg_format(%Format{kind: :float} = f), do: f
  defp reg_format(f), do: wide_format(f, 44, 40)

  defp reg_zero(%Format{kind: :float} = f), do: Num.zeros({}, f)
  defp reg_zero(%Format{} = f), do: %Fixed{t: Nx.tensor(0, type: :s64), frac: f.frac}

  defp zero_regs(:simple, f, _opts) do
    case f.kind do
      :float ->
        {Num.zeros({}, f), Num.zeros({}, f), Nx.tensor(0, type: :s64)}

      :fixed ->
        {%Fixed{t: Nx.tensor(0, type: :s64), frac: 40},
         %Fixed{t: Nx.tensor(0, type: :s64), frac: 32}, Nx.tensor(0, type: :s64)}
    end
  end

  defp zero_regs(:skew, f, _opts) do
    r = reg_zero(reg_format(f))
    {r, r, r, Nx.tensor(0, type: :s32)}
  end

  defp zero_regs(_type, f, _opts) do
    r = reg_zero(reg_format(f))
    {r, r}
  end

  # |z|^2 at 28 fractional bits (float: as is), so that its square fits 63 bits
  defp power_reg(z, f) do
    p = power(z)

    case f.kind do
      :float -> p
      :fixed -> quantize(p, wide_format(f, 36, 28))
    end
  end

  # y + alpha (v - y), the difference at 28 fractional bits
  defp ema(y, v, alpha, f) do
    case f.kind do
      :float ->
        Nx.add(y, Nx.multiply(Nx.subtract(v, y), alpha))

      :fixed ->
        rf = reg_format(f)
        y = quantize(y, rf)
        diff = quantize(Num.sub(v, y), wide_format(f, 36, 28))
        step = quantize(Num.mul(diff, wconst(alpha, f)), rf)
        quantize(Num.add(y, step), rf)
    end
  end

  defp update_regs(:simple, z, _zm, {y1, y2, count}, opts) do
    f = opts[:format]
    guard = Keyword.get(opts, :guard, 4)
    limit = 1 <<< 62
    count = count |> Nx.as_type(:s64) |> Nx.clip(i64(0), i64(limit - 1))
    first = Nx.equal(count, 0)
    next_count = Nx.add(count, 1)
    divisor = Nx.max(count, 1)

    case f.kind do
      :float ->
        mag = cmag(z, f)
        m = Nx.add(y1, Nx.divide(Nx.subtract(mag, y1), Nx.as_type(divisor, :f32)))
        dev = Nx.multiply(Nx.subtract(mag, y1), Nx.subtract(mag, m))
        {Nx.select(first, mag, m), Nx.select(first, 0.0, Nx.add(y2, dev)), next_count}

      :fixed ->
        rf = reg_format(f)
        w32 = wide_format(f, 32, 24)
        mag = cmag(z, f, f.bits, guard)
        y1 = quantize(y1, rf)
        diff = Num.sub(mag, y1)
        step = %Fixed{t: Nx.quotient(Nx.as_type(diff.t, :s64), divisor), frac: diff.frac}
        m = quantize(Num.add(y1, step), rf)
        dev = Num.mul(quantize(Num.sub(mag, y1), w32), quantize(Num.sub(mag, m), w32))
        dev = quantize(dev, wide_format(f, 62, 32))
        y2 = sat(y2, limit)
        y2n = quantize_sum(Num.add(y2, dev), limit)
        mag_rf = quantize(mag, rf)
        zero = %Fixed{t: Nx.tensor(0, type: :s64), frac: 32}
        {sel(first, mag_rf, m), sel(first, zero, y2n), next_count}
    end
  end

  defp update_regs(:m2m4, z, _zm, {y1, y2}, opts) do
    f = opts[:format]
    alpha = Keyword.get(opts, :alpha, 0.001)
    p = power_reg(z, f)
    p2 = square(p, f)
    {ema(y1, p, alpha, f), ema(y2, p2, alpha, f)}
  end

  defp update_regs(:svr, x, x1, {y1, y2}, opts) do
    f = opts[:format]
    alpha = Keyword.get(opts, :alpha, 0.001)
    p = power_reg(x, f)
    p1 = power_reg(x1, f)
    {ema(y1, product(p, p1, f), alpha, f), ema(y2, square(p, f), alpha, f)}
  end

  defp update_regs(:skew, z, _zm, {y1, y2, _y3, i}, opts) do
    f = opts[:format]
    alpha = Keyword.get(opts, :alpha, 0.001)
    {zr, zi} = z
    mag = cmag(z, f, f.bits || 32, 4)
    re2 = Num.sub(Num.mul(zr, zr), Num.mul(zi, zi))

    y1n = ema(y1, mag, alpha, f)
    y2n = ema(y2, re2, alpha, f)
    i = Nx.clip(i, 0, 2_147_483_646)
    y3n = skew_register(Num.sub(mag, y1n), y2n, i, f)
    {y1n, y2n, y3n, Nx.add(i, 1)}
  end

  # (d d_i i) d_i (i - 1) - 3 d_i y2 with d_i = d / (i + 1)
  defp skew_register(d, y2, i, %Format{kind: :float}) do
    n = Nx.as_type(i, :f32)
    d_i = Nx.divide(d, Nx.add(n, 1.0))

    Nx.subtract(
      Nx.multiply(Nx.multiply(Nx.multiply(Nx.multiply(d, d_i), n), d_i), Nx.subtract(n, 1.0)),
      Nx.multiply(Nx.multiply(d_i, 3.0), y2)
    )
  end

  defp skew_register(d, y2, i, %Format{kind: :fixed} = f) do
    rf = reg_format(f)
    w32 = wide_format(f, 32, 24)
    d = quantize(d, w32)

    # d_i = d / (i + 1); d_i i = d - d_i and d_i (i - 1) = d - 2 d_i, without the multiplications by i
    d_i = %Fixed{t: Nx.quotient(Nx.as_type(d.t, :s64), Nx.add(Nx.as_type(i, :s64), 1)), frac: 24}
    a = quantize(Num.sub(d, d_i), w32)
    b = quantize(Num.sub(d, Num.add(d_i, d_i)), w32)
    t1 = Num.mul(quantize(Num.mul(d, a), w32), b)
    t2 = Num.mul(quantize(Num.mul(d_i, wconst(3.0, f)), w32), quantize(y2, w32))
    quantize(Num.sub(t1, t2), rf)
  end

  defp square(p, _f), do: Num.mul(p, p)
  defp product(a, b, _f), do: Num.mul(a, b)

  defp sat(%Fixed{t: t, frac: frac}, limit),
    do: %Fixed{t: t |> Nx.as_type(:s64) |> Nx.clip(i64(-limit), i64(limit)), frac: frac}

  defp quantize_sum(%Fixed{t: t, frac: frac}, limit),
    do: %Fixed{t: Nx.clip(Nx.as_type(t, :s64), i64(-limit), i64(limit)), frac: frac}

  @doc """
  The SNR in dB from the registers after `count` items (as GNU Radio's
  `snr()`), computed in double precision on the host. `registers` is a
  tuple of numbers (floats; `Num.to_float` of the kernel's output, `:simple`'s
  count as the integer), `type` is as for `snr_est/3`.
  """
  def snr_db(type, registers, opts \\ [])

  def snr_db(:simple, {y1, y2, count}, _opts),
    do: 10 * :math.log10(y1 / (2 * y2 / (count - 1)))

  def snr_db(:m2m4, {y1, y2}, _opts) do
    y1_2 = y1 * y1
    10 * :math.log10(:math.sqrt(2 * y1_2 - y2) / (y1 - :math.sqrt(2 * y1_2 - y2)))
  end

  def snr_db(:svr, {y1, y2}, _opts) do
    x = y1 / (y2 - y1)
    10 * :math.log10(x - 1 + :math.sqrt(x * (x - 1)))
  end

  def snr_db(:skew, {y1, y2, y3, _i}, _opts) do
    y3sq = y3 * y3 / (y2 * y2 * y2)
    signal = y1 * y1
    x = signal - y2
    noise = 2 * (x + y3sq * signal)
    10 * :math.log10(signal / noise)
  end

  @doc "`{signal_db, noise_db}` as `mpsk_snr_est::signal()` and `noise()` give them."
  def signal_noise_db(:simple, {y1, y2, count}, _opts),
    do: {10 * :math.log10(y1), 10 * :math.log10(2 * y2 / (count - 1))}

  def signal_noise_db(:m2m4, {y1, y2}, _opts) do
    s = :math.sqrt(2 * y1 * y1 - y2)
    {10 * :math.log10(s), 10 * :math.log10(y1 - s)}
  end

  def signal_noise_db(:skew, {y1, y2, y3, _i}, _opts) do
    y3sq = y3 * y3 / (y2 * y2 * y2)
    signal = y1 * y1
    {10 * :math.log10(signal), 10 * :math.log10(2 * (signal - y2 + y3sq * signal))}
  end

  # -- probe_density_b ----------------------------------------------------------------------

  @doc """
  `probe_density_b`: `density += alpha (in - density)` per byte, starting at
  1.0 (GNU Radio writes `alpha in + (1 - alpha) density`). Output: the density
  after each item. Fixed point: 32 fractional bits, 8 integer bits. Options:
  `:format` (only its kind matters), `:alpha`.
  """
  def probe_density(b, density, opts) do
    f = opts[:format]
    alpha = opts[:alpha]
    v = Nx.as_type(b, :u8)

    case f.kind do
      :float ->
        d = Nx.add(density, Nx.multiply(Nx.subtract(Nx.as_type(v, :f32), density), alpha))
        {d, d}

      :fixed ->
        df = density_format()
        d0 = quantize(density, df)
        vv = %Fixed{t: Nx.as_type(v, :s64), frac: 0}
        diff = quantize(Num.sub(vv, d0), wide_format(f, 36, 24))
        step = quantize(Num.mul(diff, wconst(alpha, f)), df)
        d = quantize(Num.add(d0, step), df)
        {d, d}
    end
  end

  @doc "The density format of `probe_density/3`: 40 bits with 32 fractional bits."
  def density_format, do: NxFixed.Num.q(40, 32)

  @doc "The initial density (1.0 as GNU Radio)."
  def density_init(%Format{kind: :float}), do: Nx.tensor(1.0, type: :f32)

  def density_init(%Format{kind: :fixed}),
    do: %Fixed{t: Nx.tensor(1 <<< 32, type: :s64), frac: 32}

  # -- probe_signal -----------------------------------------------------------------------------

  @doc "`probe_signal`: the level is the last item; output: the item itself, state: the last item."
  def probe_signal(x, _last, _opts), do: {x, x}
end
