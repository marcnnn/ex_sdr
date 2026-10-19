defmodule ExSdr.Carrier.Equalizer do
  @moduledoc """
  The adaptive equalizers of gr-digital (`linear_equalizer`,
  `decision_feedback_equalizer`) as stream kernels, with the three adaptive
  algorithms `adaptive_algorithm_lms`, `_nlms` and `_cma`.

  ## What a step does

  Every input sample enters a window of `num_taps` samples (oldest first,
  state: the previous `num_taps - 1`). Every `sps`-th sample (the first
  one included) is an output instant, as in GNU Radio's decimating block:

      y   = sum_i w[i] * window[i]
      e   = d - y                       LMS, NLMS: d is the training symbol or
                                        the nearest constellation point (decision directed)
          = y (|y|^2 - modulus)         CMA, each part clipped to +-1
      w  += mu * conj(window) * e       LMS
      w  += mu / |window|^2 * conj(window) * e   NLMS
      w  -= mu * conj(window) * e       CMA

  `w[i]` multiplies `window[i]`, i.e. it is GNU Radio's tap vector in the
  order the FIR kernel stores it (reversed against `set_taps`): `w[num_taps - 1]`
  weighs the newest sample. LMS and NLMS start from zero weights, CMA from
  the unit tap on the newest sample (`adaptive_algorithm::initialize_taps`).

  Training: with a training sequence the equalizer idles until the training
  flag input is set (GNU Radio's `training_start_tag`), trains on the
  sequence from the output instant of the flagged sample on and then, if
  `adapt_after_training`, continues decision directed. Without a training
  sequence it is decision directed from the start (CMA ignores both).

  ## Number formats

  Fixed point: samples are in the block's format `Q(f)`; the weights have
  `weight_bits` bits with `weight_frac` fractional bits (default 32 and 28:
  range +-8, steps of 3.7e-9). A step size of `mu` moves a weight by about
  `mu * |x| * |e|` per symbol, so for `mu = 1e-4` and an error of 1e-3 the
  update is 3e-8: still eight LSBs. Below roughly `mu * |x| * |e| < 2e-9` the
  weight stops moving (a dead band, like the 2^-15 one of every Q15 filter,
  but 13 bits finer); raise `weight_frac` (and `weight_bits`, up to 40
  here) for smaller steps. The filter output is rounded to the format
  plus `guard` (8) fractional bits, with two more integer bits of headroom;
  the error and its product with `mu` are 30 bit fractions; the NLMS step
  divides by the window energy with a floor of 2^-16. Everything is exact
  integer arithmetic, so the fixed-point instance is bit-exact on every backend.

  The state is `{line, weights, control}`; `control` is the training state
  (0 idle, 1 training, 2 decision directed), the index into the training
  sequence and the decimation phase.
  """
  import Bitwise
  import ExSdr.Analog.Math

  alias ExSdr.Carrier.Kernels, as: K
  alias ExSdr.Digital.Kernels, as: Digital
  alias NxFixed.Fixed
  alias NxFixed.Num
  alias NxFixed.Num.Format

  @idle 0
  @training 1
  @dd 2

  @doc "The weight format of the options: float, or `weight_bits` / `weight_frac` fixed point."
  def weight_format(opts) do
    case opts[:format] do
      %Format{kind: :float} = f ->
        f

      %Format{} = f ->
        wide_format(f, Keyword.get(opts, :weight_bits, 32), Keyword.get(opts, :weight_frac, 28))
    end
  end

  # the filter output / error format: guard fractional bits and two integer bits more
  defp error_format(%Format{kind: :float} = f, _opts), do: f

  defp error_format(%Format{} = f, opts) do
    g = Keyword.get(opts, :guard, 8)
    wide_format(f, f.bits + g + 2, f.frac + g)
  end

  # error times step size: 30 fractional bits, 34 bits
  defp step_format(%Format{kind: :float} = f), do: f
  defp step_format(f), do: wide_format(f, 34, 30)

  # -- linear equalizer ----------------------------------------------------------------

  @doc """
  The initial state of a linear equalizer for `opts` (see `step/3`).
  """
  def init(opts) do
    f = opts[:format]
    n = opts[:num_taps]
    wf = weight_format(opts)
    line = {Num.zeros({n - 1}, f), Num.zeros({n - 1}, f)}
    w = initial_weights(opts, n, wf)
    training? = Keyword.get(opts, :training, []) != []
    ts = if training?, do: @idle, else: @dd
    {line, w, {Nx.tensor(ts, type: :u8), Nx.tensor(0, type: :s32), Nx.tensor(0, type: :s32)}}
  end

  defp initial_weights(opts, n, wf) do
    taps =
      case Keyword.get(opts, :initial_taps) do
        nil ->
          if opts[:algorithm] == :cma,
            do: List.duplicate({0.0, 0.0}, n - 1) ++ [{1.0, 0.0}],
            else: List.duplicate({0.0, 0.0}, n)

        t ->
          t
      end

    {tensor_of(Enum.map(taps, &elem(&1, 0)), wf), tensor_of(Enum.map(taps, &elem(&1, 1)), wf)}
  end

  defp tensor_of(values, %Format{kind: :float}), do: Nx.tensor(values, type: :f32)

  defp tensor_of(values, %Format{kind: :fixed} = f) do
    raw = Enum.map(values, &to_raw(&1, f.frac, f.bits))
    %Fixed{t: Nx.tensor(raw, type: if(f.bits <= 32, do: :s32, else: :s64)), frac: f.frac}
  end

  @doc """
  One input item of `linear_equalizer`: `x` (complex), or `{x, flag}` with a
  `u8` training flag when `:training` is given. Returns the output
  and the next state; with `sps > 1` the output is `{{values, count}}` as for
  variable rate stream blocks (`count` 1 on output instants).

  Options: `:format`, `:num_taps`, `:sps` (1), `:algorithm` (`:lms`, `:nlms`,
  `:cma`), `:mu`, `:modulus` (CMA, 1.0), `:points` (constellation points
  for decisions), `:training` (training symbols, `{re, im}`; default none),
  `:adapt_after_training` (true), `:weight_bits`, `:weight_frac`, `:guard`,
  `:outputs` (extras from `:taps`, the weights before the update, and
  `:state`, the training state before the flag, as GNU Radio's optional
  output ports).
  """
  def step(input, {line, w, ctl}, opts) do
    f = opts[:format]
    sps = Keyword.get(opts, :sps, 1)
    training = Keyword.get(opts, :training, [])

    {x, flag} = if training != [], do: input, else: {input, nil}
    x = K.bound(x, f)
    wf = weight_format(opts)
    ef = error_format(f, opts)

    win = K.cmap2(K.cmap(line, &quantize(&1, f)), x, &K.append/2)
    next_line = K.cmap(win, &K.drop_first/1)
    w = bound_w(w, wf)

    ye = quantize(cdot(w, win), ef)
    y = quantize(ye, f)

    {ts0, tc0, ph0} = bound_ctl(ctl, training, sps)
    emit = Nx.equal(ph0, 0)

    {ts2, tc2} = training_state(flag, ts0, tc0, training, opts)
    training_now = Nx.equal(ts2, @training)
    active = Nx.logical_and(emit, Nx.not_equal(ts2, @idle))

    e = error(opts[:algorithm], ye, ts2, tc2, training_now, f, ef, opts)
    w_new = update(opts[:algorithm], w, win, e, f, wf, opts)
    w_next = sel(active, w_new, w)

    tc3 = Nx.select(Nx.logical_and(emit, training_now), Nx.add(tc2, 1), tc2)
    ts_next = Nx.select(emit, ts2, ts0)
    tc_next = Nx.select(emit, tc3, tc0)

    ph_next =
      if sps == 1,
        do: ph0,
        else: Nx.remainder(Nx.add(ph0, 1), sps)

    ctl_next = {Nx.as_type(ts_next, :u8), Nx.as_type(tc_next, :s32), Nx.as_type(ph_next, :s32)}
    next = {next_line, w_next, ctl_next}
    {output(y, w, ts0, emit, sps, opts), next}
  end

  defp bound_w({wr, wi}, wf), do: {quantize(wr, wf), quantize(wi, wf)}

  defp bound_ctl({ts, tc, ph}, training, sps) do
    size = length(training)

    {Nx.clip(ts, 0, 2) |> Nx.as_type(:s32), Nx.clip(tc, -1, max(size, 1)),
     Nx.clip(ph, 0, max(sps - 1, 0))}
  end

  # GNU Radio's order: the flag first, then the "training finished" check.
  defp training_state(flag, ts, tc, training, opts) do
    after_state = if Keyword.get(opts, :adapt_after_training, true), do: @dd, else: @idle
    size = length(training)

    {ts1, tc1} =
      if flag do
        set = Nx.not_equal(flag, 0)
        {Nx.select(set, @training, ts), Nx.select(set, 0, tc)}
      else
        {ts, tc}
      end

    done = Nx.logical_or(Nx.less(tc1, 0), Nx.greater_equal(tc1, size))
    {Nx.select(done, after_state, ts1), Nx.select(done, -1, tc1)}
  end

  # LMS and NLMS: desired minus output; CMA: the constant modulus error.
  defp error(:cma, ye, _ts, _tc, _training_now, f, ef, opts), do: cma_error(ye, f, ef, opts)

  defp error(_alg, ye, _ts, tc, training_now, f, ef, opts) do
    points = opts[:points]
    {idx, _} = Digital.decide(ye, Nx.tensor(0, type: :u8), format: ef, points: points)

    dd =
      {Digital.lookup(idx, Enum.map(points, &elem(&1, 0)), ef),
       Digital.lookup(idx, Enum.map(points, &elem(&1, 1)), ef)}

    desired =
      case Keyword.get(opts, :training, []) do
        [] ->
          dd

        seq ->
          tr =
            {Digital.lookup(tc, Enum.map(seq, &elem(&1, 0)), ef),
             Digital.lookup(tc, Enum.map(seq, &elem(&1, 1)), ef)}

          sel(training_now, tr, dd)
      end

    _ = f
    Num.csub(desired, ye)
  end

  # y (|y|^2 - modulus), each component clipped to +-1 (gr::clip)
  defp cma_error({yr, yi} = ye, f, ef, opts) do
    modulus = Keyword.get(opts, :modulus, 1.0)

    case f.kind do
      :float ->
        p = Nx.subtract(Nx.add(Nx.multiply(yr, yr), Nx.multiply(yi, yi)), modulus)
        {Nx.clip(Nx.multiply(yr, p), -1.0, 1.0), Nx.clip(Nx.multiply(yi, p), -1.0, 1.0)}

      :fixed ->
        p = Num.add(Num.mul(yr, yr), Num.mul(yi, yi))
        p = Num.sub(p, const_like(modulus, p))
        # |y|^2 - modulus within +-8 at 24 fractional bits
        p = quantize(p, wide_format(f, 32, 24))
        prod = {Num.mul(yr, p), Num.mul(yi, p)}
        {er, ei} = quantize(prod, step_format(f))
        _ = {ye, ef}
        {clamp(er, -1.0, 1.0, 34), clamp(ei, -1.0, 1.0, 34)}
    end
  end

  # w + gain * conj(window) * e, rounded and saturated to the weight format
  defp update(alg, {wr, wi} = w, {xr, xi} = win, e, f, wf, opts) do
    mu = opts[:mu]
    sf = step_format(f)

    {gr, gi} =
      case alg do
        :lms -> scale_error(e, mu, sf)
        :cma -> scale_error(e, -mu, sf)
        :nlms -> nlms_step(e, win, mu, f, sf)
      end

    # conj(window) * g
    dr = Num.add(Num.mul(xr, gr), Num.mul(xi, gi))
    di = Num.sub(Num.mul(xr, gi), Num.mul(xi, gr))
    _ = w
    {quantize(Num.add(wr, dr), wf), quantize(Num.add(wi, di), wf)}
  end

  defp scale_error(e, mu, sf) do
    case sf.kind do
      :float -> {Nx.multiply(elem(e, 0), mu), Nx.multiply(elem(e, 1), mu)}
      :fixed -> quantize(Num.cscale(e, wconst(mu, sf)), sf)
    end
  end

  # mu e / sum |window|^2
  defp nlms_step(e, {xr, xi}, mu, f, sf) do
    energy = Num.add(dot(xr, xr), dot(xi, xi))
    {mr, mi} = scale_error(e, mu, sf)

    case f.kind do
      :float ->
        m = Nx.max(energy, 1.0e-30)
        {Nx.divide(mr, m), Nx.divide(mi, m)}

      :fixed ->
        # energy has 2 f fractional bits; bring it to 30 (a shift left or right
        # of exact integers), floor at 2^-16, then g = e * 2^30 / energy at 30 fractional bits
        %Fixed{t: m} = quantize(energy, wide_format(f, 62, 30))
        m = Nx.max(Nx.as_type(m, :s64), i64(1 <<< 14))
        {divide30(mr, m), divide30(mi, m)}
    end
  end

  defp divide30(%Fixed{t: t}, m) do
    t = t |> Nx.as_type(:s64) |> Nx.clip(i64(-(1 <<< 31)), i64((1 <<< 31) - 1))
    q = Nx.quotient(Nx.left_shift(t, 30), m)
    %Fixed{t: Nx.clip(q, i64(-(1 <<< 40)), i64(1 <<< 40)), frac: 30}
  end

  # -- output ------------------------------------------------------------------------------

  defp output(y, w, ts, emit, sps, opts) do
    out =
      case Keyword.get(opts, :outputs, []) do
        [] ->
          y

        names ->
          extras =
            for n <- names do
              case n do
                :taps -> w
                :state -> Nx.as_type(ts, :u8)
              end
            end

          {y, List.to_tuple(extras)}
      end

    if sps == 1 do
      out
    else
      {cmap_all(out, &add_axis/1), Nx.as_type(emit, :s32)}
    end
  end

  defp cmap_all({y, extras}, fun) when is_tuple(extras),
    do: {cmap_all(y, fun), List.to_tuple(Enum.map(Tuple.to_list(extras), &cmap_all(&1, fun)))}

  defp cmap_all({a, b}, fun), do: {fun.(a), fun.(b)}
  defp cmap_all(a, fun), do: fun.(a)

  defp add_axis(%Fixed{t: t, frac: frac}), do: %Fixed{t: Nx.new_axis(t, 0), frac: frac}
  defp add_axis(t), do: Nx.new_axis(t, 0)

  # -- vector helpers ------------------------------------------------------------------------

  # sum_i a[i] b[i] for real vectors; exact for fixed point
  defp dot(%Fixed{t: a, frac: fa}, %Fixed{t: b, frac: fb}),
    do: %Fixed{t: Nx.sum(Nx.multiply(Nx.as_type(a, :s64), Nx.as_type(b, :s64))), frac: fa + fb}

  defp dot(a, b), do: Nx.sum(Nx.multiply(a, b))

  defp cdot({wr, wi}, {xr, xi}),
    do: {Num.sub(dot(wr, xr), dot(wi, xi)), Num.add(dot(wr, xi), dot(wi, xr))}
end
