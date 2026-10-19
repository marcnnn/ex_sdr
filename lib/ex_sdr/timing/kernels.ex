defmodule ExSdr.Timing.Kernels do
  @moduledoc """
  Stream kernels of `ExSdr.Timing`, written once against `NxFixed.Num` and
  `ExSdr.Timing.Loop`.

  ## The variable-rate machine

  GNU Radio's timing blocks walk through their input with a position that
  advances by a real-valued step per output (or per interpolation). Here
  the same walk is driven by the arrival of input items, one per kernel
  call (`machine/4`):

    * the state keeps a line of the last `L` input items (oldest first) and
      `wait`, the number of items still to arrive before the window of the
      next iteration is complete;
    * each arriving item is pushed into the line and decrements `wait`;
    * then up to `:iterations` iterations run, each while `wait <= 0`: the
      block's body interpolates from the line, updates its loop state,
      may emit an output, and adds the integer part of its position step to
      `wait`.

  Outputs go into `max` lanes with a count (`rate: {:variable, max}` of
  `NxFixed.Block`). Everything a later iteration depends on is in the
  state, so input split anywhere gives the same output as one call.

  GNU Radio's blocks start with `history - 1` zero items in front of the
  input; the zero line plays that part, and the initial `wait` places the
  first iteration where GNU Radio's first iteration is.
  """
  alias ExSdr.Timing.Interp
  alias ExSdr.Timing.Loop, as: L
  alias NxFixed.Num

  import Bitwise

  @wait_lo -64
  @wait_hi 1 <<< 20

  # -- machine ---------------------------------------------------------------------------------

  @doc """
  One input item through the machine. `state` is `{line, wait, s}`; `body`
  is `fn line, wait, s -> {emits, delta, s2} end` with `emits` a list of
  `{ok, value}` (`ok` a boolean scalar or `true`). Options: `:format`,
  `:iterations` (iterations per item), `:max` (output lanes), `:out`
  (`:real` or `:complex`), `:bound_state` (`fn s -> s end`, states the
  ranges of the loop state).
  """
  def machine(x, {line, wait, s}, opts, body) do
    f = opts[:format]
    line = line |> L.push(x) |> Num.quantize(f)
    wait = wait |> L.bound_int(@wait_lo, @wait_hi) |> Nx.subtract(1)
    s = opts[:bound_state].(s)
    lanes = L.lanes(L.zero(opts[:out], f), opts[:max])

    {wait, s, lanes, count} =
      Enum.reduce(1..opts[:iterations], {wait, s, lanes, L.i32(0)}, fn _,
                                                                       {wait, s, lanes, count} ->
        active = Nx.less_equal(wait, 0)
        {emits, delta, s2} = body.(line, wait, s)
        s = L.sel(active, s2, s)

        {lanes, count} =
          Enum.reduce(emits, {lanes, count}, fn {ok, v}, {lanes, count} ->
            ok = if ok == true, do: active, else: Nx.logical_and(active, ok)
            L.put_lane(lanes, count, v, ok)
          end)

        wait = Nx.add(wait, Nx.select(active, Nx.as_type(delta, :s32), 0))
        {wait, s, lanes, count}
      end)

    {{lanes, count}, {line, Nx.as_type(wait, :s32), s}}
  end

  @doc "Initial machine state: zero line of `len` items, `wait`, block state `s`."
  def init(kind, f, len, wait, s), do: {L.zero_line(kind, f, len), Nx.tensor(wait, type: :s32), s}

  # -- fractional delay ------------------------------------------------------------------------------

  @doc """
  Fixed fractional delay, one output per input: the interpolator applied to
  the last `ntaps` items at the constant delay `:mu` (a row index
  `:arm`). State: the line of the last `ntaps - 1` items. Options:
  `:format`, `:interp`, `:arm`, `:tap_frac`.
  """
  def fractional_delay(x, line, opts) do
    f = opts[:format]
    ip = opts[:interp]
    win = line |> L.zip(x, &append/2) |> Num.quantize(f)
    y = Interp.apply_rows(win, opts[:arm], ip.rows, f, tap_frac: opts[:tap_frac])
    next = L.window(win, 1, ip.ntaps - 1)
    {y, next}
  end

  # -- mmse_resampler --------------------------------------------------------------------------------

  @doc """
  `mmse_resampler_cc/ff`: output `k` interpolates the window starting at
  `floor(position_k)` at `mu = frac(position_k)`; the position advances by
  `:step` (the resampling ratio, input items per output). Body state: `mu`
  (f64 like GNU Radio's `double`, or a loop word). Options: machine options
  plus `:interp`, `:step` (loop constant), `:tap_frac`.
  """
  def mmse_resampler(x, state, opts) do
    f = opts[:format]
    ip = opts[:interp]
    step = opts[:step]

    machine(x, state, opts, fn line, _wait, mu ->
      win = L.window(line, 0, ip.ntaps)
      y = Interp.interpolate(ip, win, mu, f, tap_frac: opts[:tap_frac])
      {n, frac} = L.split(Nx.add(mu, step), f)
      {[{true, y}], n, frac}
    end)
  end

  # -- pfb_arb_resampler ---------------------------------------------------------------------------------

  @doc """
  `pfb_arb_resampler_ccf/fff`: arm `j` of `nfilts` and its derivative arm
  give `o0 + acc * o1` on the window of the last `ntaps` items. Per output
  `acc += flt_rate; j += dec_rate + floor(acc); acc = frac(acc)`; `j div
  nfilts` input items are consumed. Body state `{j, acc}`. Options: machine
  options plus `:interp` (polyphase with derivative), `:dec_rate`,
  `:flt_rate` (loop constant), `:tap_frac`, `:mu_bits` (fixed point:
  bits of `acc` in the multiplier, default 18).
  """
  def pfb_arb_resampler(x, state, opts) do
    f = opts[:format]
    ip = opts[:interp]
    n = ip.arms

    machine(x, state, opts, fn line, _wait, {j, acc} ->
      win = L.window(line, 0, ip.ntaps)
      ro = [tap_frac: opts[:tap_frac], raw: true]
      o0 = Interp.apply_rows(win, j, ip.rows, f, ro)
      o1 = Interp.apply_rows(win, j, ip.drows, f, ro)
      y = arb_combine(o0, o1, acc, f, opts[:mu_bits] || 18)

      {fl, acc2} = L.split(Nx.add(acc, opts[:flt_rate]), f)
      j2 = j |> Nx.add(opts[:dec_rate]) |> Nx.add(fl)
      delta = Nx.quotient(j2, n)
      {[{true, y}], delta, {Nx.as_type(Nx.remainder(j2, n), :s32), acc2}}
    end)
  end

  defp arb_combine(o0, o1, acc, %{kind: :float} = f, _mu_bits) do
    Num.quantize(L.zip(o0, o1, fn a, b -> Nx.add(a, Nx.multiply(b, acc)) end), f)
  end

  defp arb_combine(o0, o1, acc, f, mu_bits) do
    o1 = Num.quantize(o1, f)
    a = %NxFixed.Fixed{t: L.shift_round(acc, L.p() - mu_bits), frac: mu_bits}
    L.zip(o0, o1, fn p, q -> Num.add(p, Num.mul(q, a)) end) |> Num.quantize(f)
  end

  # -- clock_recovery_mm ----------------------------------------------------------------------------

  @doc """
  `clock_recovery_mm_cc/ff` (Mueller and Muller). Per output: interpolate at
  `mu`, the M&M error `mm` (complex: GNU Radio's 0/1 slicer, clipped to
  +-1; real: +-1 slicer, not clipped), then
  `omega = mid + clip(omega + gain_omega * mm - mid, lim)`,
  `mu = mu + omega + gain_mu * mm`, and `floor(mu)` items are consumed.
  Body state: complex `{mu, omega, p0, p1}` (the last two interpolated
  samples), real `{mu, omega, last}`. Options: machine options plus
  `:interp`, `:gain_mu`, `:gain_omega`, `:omega_mid`, `:omega_lim` (floats),
  `:tap_frac`.
  """
  def clock_recovery_mm(x, state, opts) do
    f = opts[:format]
    ip = opts[:interp]

    machine(x, state, opts, fn line, _wait, s ->
      {mu, omega} = {elem(s, 0), elem(s, 1)}
      p0 = Interp.interpolate(ip, L.window(line, 0, ip.ntaps), mu, f, tap_frac: opts[:tap_frac])

      {mm, hist} =
        case s do
          {_, _, p1, p2} -> {num_clip(mm_complex(p0, p1, p2), 1.0, f), {p0, p1}}
          {_, _, last} -> {mm_real(p0, last), {p0}}
        end

      mid = L.const(opts[:omega_mid], f)
      omega = L.add(omega, L.gain(mm, opts[:gain_omega], f))
      omega = L.add(mid, L.clip(L.sub(omega, mid), opts[:omega_lim], f))
      mu = L.add(L.add(mu, omega), L.gain(mm, opts[:gain_mu], f))
      {n, mu} = L.split(mu, f)
      {[{true, p0}], n, List.to_tuple([mu, omega | Tuple.to_list(hist)])}
    end)
  end

  # GNU Radio's slicer_0deg: 1 where the component is > 0, else 0; then
  # u = (p0 - p2) conj(c1) - (c0 - c2) conj(p1), mm = re(u). The products with
  # 0 and 1 are selections.
  defp mm_complex(p0, {p1r, p1i} = p1, p2) do
    {c0r, c0i} = slice01(p0)
    {c1r, c1i} = slice01(p1)
    {c2r, c2i} = slice01(p2)
    {dr, di} = Num.csub(p0, p2)
    y = Num.add(pick(c1r, dr), pick(c1i, di))
    x = Num.add(diff01(c0r, c2r, p1r), diff01(c0i, c2i, p1i))
    Num.sub(y, x)
  end

  defp slice01({re, im}), do: {Nx.greater(L.raw(re), 0), Nx.greater(L.raw(im), 0)}

  defp pick(c, v), do: L.like(v, Nx.select(c, L.raw(v), Nx.multiply(L.raw(v), 0)))

  # (a - b) * v for a, b in {0, 1}
  defp diff01(a, b, v) do
    t = widen(L.raw(v))
    z = Nx.multiply(t, 0)
    up = Nx.logical_and(a, Nx.logical_not(b))
    down = Nx.logical_and(b, Nx.logical_not(a))
    L.like(v, Nx.select(up, t, Nx.select(down, Nx.negate(t), z)))
  end

  # mm = slice(last) * out - slice(out) * last, slice = -1 below 0, else 1
  defp mm_real(out, last) do
    a = sign_times(last, out)
    b = sign_times(out, last)
    Num.sub(a, b)
  end

  defp sign_times(s, v) do
    t = widen(L.raw(v))
    L.like(v, Nx.select(Nx.less(L.raw(s), 0), Nx.negate(t), t))
  end

  # One more bit before negating (-(-2^15) needs 17 bits)
  defp widen(t) do
    case Nx.type(t) do
      {:s, 8} -> Nx.as_type(t, :s16)
      {:s, 16} -> Nx.as_type(t, :s32)
      {:s, 32} -> Nx.as_type(t, :s64)
      _ -> t
    end
  end

  @doc false
  # GNU Radio's branchless_clip on a sample-domain value (float: its
  # roundings; fixed: exact clip).
  def num_clip(x, c, %{kind: :float} = f), do: L.clip(x, c, f)

  def num_clip(%NxFixed.Fixed{t: t, frac: frac}, c, _f) do
    lim = round(c * (1 <<< frac))
    t = Nx.as_type(t, :s64)
    %NxFixed.Fixed{t: Nx.clip(t, L.i64(-lim), L.i64(lim)), frac: frac}
  end

  # -- msk_timing_recovery ---------------------------------------------------------------------------

  @doc """
  `msk_timing_recovery_cc`: the loop runs at two interpolations per symbol.
  Per interpolation `y`: `nl = y^2 conj(c^2)` with `c` the previous
  interpolation, `err = re(nl - nl_prev)`; on odd interpolations
  `err` is clipped to +-3 and updates `omega` (clipped to `sps/2 +- limit`)
  and `mu`. Even interpolations (and all with `osps = 2`) are output.
  Body state `{mu, omega, c, nl_prev, odd}`. Options: machine options plus
  `:interp`, `:gain`, `:gain_omega`, `:omega_mid`, `:limit`, `:osps`,
  `:wide` (fixed point: format of the products), `:tap_frac`.
  """
  def msk_timing_recovery(x, state, opts) do
    f = opts[:format]
    ip = opts[:interp]
    w = opts[:wide] || f

    machine(x, state, opts, fn line, _wait, {mu, omega, c, nl1, odd} ->
      y = Interp.interpolate(ip, L.window(line, 0, ip.ntaps), mu, f, tap_frac: opts[:tap_frac])
      sq = y |> Num.cmul(y) |> Num.quantize(w)
      {cr, ci} = c |> Num.cmul(c) |> Num.quantize(w)
      dly = {cr, Num.sub(Num.zeros({}, w), ci) |> Num.quantize(w)}
      nl = sq |> Num.cmul(dly) |> Num.quantize(w)
      err = Num.sub(elem(nl, 0), elem(nl1, 0))

      is_odd = Nx.equal(odd, 1)
      e3 = num_clip(err, 3.0, f)
      mid = L.const(opts[:omega_mid], f)
      omega2 = L.add(omega, L.gain(e3, opts[:gain_omega], f))
      omega2 = L.add(mid, L.clip(L.sub(omega2, mid), opts[:limit], f))
      mu2 = L.add(mu, L.gain(e3, opts[:gain], f))
      omega = Nx.select(is_odd, omega2, omega)
      mu = Nx.select(is_odd, mu2, mu)

      emit = if opts[:osps] == 2, do: true, else: Nx.logical_not(is_odd)
      {n, mu} = L.split(L.add(mu, omega), f)
      {[{emit, y}], n, {mu, omega, y, nl, Nx.subtract(1, odd) |> Nx.as_type(:s32)}}
    end)
  end

  # -- pfb_clock_sync ---------------------------------------------------------------------------------

  @doc """
  `pfb_clock_sync_ccf/fff`. Per symbol, for each of `:osps` outputs: the
  filter arm `floor(k)` (wrapping `k` into `[0, nfilts)` moves the input
  position), the arm on the window at the position, `k += rate_i +
  rate_f`. Then the error `out * diff` (complex: the mean of the I and Q
  products) from the derivative arm, `sps` loop updates `rate_f += beta
  err; k += rate_f + alpha err`, `rate_f` clipped to `+-max_dev`, and `sps`
  items consumed.

  The window position of a symbol moves by the wraps of `k`, which depend
  on the data: the line keeps `:margin` items on both sides and windows
  are taken at dynamic offsets. Body state `{k, rate_f}`. Options:
  machine options plus `:interp` (polyphase with derivative), `:sps`
  (integer), `:osps`, `:rate_i`, `:alpha`, `:beta`, `:max_dev`, `:margin`,
  `:wide`, `:tap_frac`.
  """
  def pfb_clock_sync(x, state, opts) do
    f = opts[:format]
    ip = opts[:interp]
    nf = ip.arms
    t = ip.ntaps
    margin = opts[:margin]
    w = opts[:wide] || f
    tf = [tap_frac: opts[:tap_frac]]

    machine(x, state, opts, fn line, wait, {k, rate_f} ->
      # line index of the sample at offset `rel` from the symbol's start
      at = fn rel -> Nx.add(Nx.add(rel, margin), wait) end

      {outs, {k, fnum, c}} =
        Enum.map_reduce(0..(opts[:osps] - 1), {k, nil, L.i32(0)}, fn idx, {k, _, c} ->
          {fl, _} = L.split(k, f)
          wraps = floor_div(fl, nf)
          k = L.sub(k, L.mulc_int(wraps, nf, f))
          fnum = Nx.subtract(fl, Nx.multiply(wraps, nf))
          c = Nx.add(c, wraps)
          win = L.window(line, at.(Nx.add(c, idx)), t)
          out = Interp.apply_rows(win, fnum, ip.rows, f, tf)
          k = L.add(L.add(k, L.const(opts[:rate_i], f)), rate_f)
          {out, {k, fnum, c}}
        end)

      win = L.window(line, at.(c), t)
      diff = Interp.apply_rows(win, fnum, ip.drows, f, tf ++ [raw: true]) |> Num.quantize(w)
      err = pfb_error(hd(outs), diff, f)

      {k, rate_f} =
        Enum.reduce(1..opts[:sps], {k, rate_f}, fn _, {k, rate_f} ->
          rate_f = L.add(rate_f, L.gain(err, opts[:beta], f))
          {L.add(L.add(k, rate_f), L.gain(err, opts[:alpha], f)), rate_f}
        end)

      rate_f = L.clip(rate_f, opts[:max_dev], f)
      emits = Enum.map(outs, &{true, &1})
      {emits, Nx.add(c, opts[:sps]), {k, rate_f}}
    end)
  end

  # (out.im * diff.im + out.re * diff.re) / 2, or out * diff
  defp pfb_error({or_, oi}, {dr, di}, f) do
    s = Num.add(Num.mul(oi, di), Num.mul(or_, dr))
    half(s, f)
  end

  defp pfb_error(o, d, _f), do: Num.mul(o, d)

  defp half(%NxFixed.Fixed{} = s, _f), do: %{s | frac: s.frac + 1}
  defp half(s, _f), do: Nx.divide(s, 2.0)

  # floor(a / n) for s32 a and a positive integer n
  defp floor_div(a, n) do
    if Bitwise.band(n, n - 1) == 0 do
      Nx.right_shift(a, trunc(:math.log2(n)))
    else
      q = Nx.quotient(a, n)
      Nx.select(Nx.less(Nx.remainder(a, n), 0), Nx.subtract(q, 1), q)
    end
  end

  # -- symbol_sync -------------------------------------------------------------------------------------

  @teds %{
    mueller_muller: %{inputs: 1, depth: 2, lookahead: false, derivative: false, slicer: true},
    modified_mueller_muller: %{
      inputs: 1,
      depth: 3,
      lookahead: false,
      derivative: false,
      slicer: true
    },
    zero_crossing: %{inputs: 2, depth: 3, lookahead: false, derivative: false, slicer: true},
    gardner: %{inputs: 2, depth: 3, lookahead: false, derivative: false, slicer: false},
    early_late: %{inputs: 2, depth: 2, lookahead: true, derivative: false, slicer: false},
    dandrea_mengali_gen_msk: %{
      inputs: 4,
      depth: 8,
      lookahead: false,
      derivative: false,
      slicer: false
    },
    mengali_dandrea_gmsk: %{
      inputs: 2,
      depth: 4,
      lookahead: false,
      derivative: false,
      slicer: false
    },
    signal_times_slope_ml: %{
      inputs: 1,
      depth: 1,
      lookahead: false,
      derivative: true,
      slicer: false
    },
    signum_times_slope_ml: %{
      inputs: 1,
      depth: 1,
      lookahead: false,
      derivative: true,
      slicer: false
    }
  }

  @doc """
  The timing error detectors of `symbol_sync`: inputs per symbol, history
  depth, whether they need a look-ahead sample, a derivative, a slicer.
  """
  def teds, do: @teds

  @doc """
  `symbol_sync_cc/ff`. Iterations happen at `ips = lcm(ted inputs per
  symbol, osps)` interpolations per symbol, counted by the interpolation
  clock `ic`: every iteration interpolates at `mu`; TED input iterations
  feed the detector, output iterations emit, and the symbol iteration
  (`ic = 0`) computes the error (with a look-ahead interpolation for the
  early-late detector) and advances the PI loop filter
  (`avg += beta err`, clipped to the period limits, `inst = avg + alpha
  err`). The position advances by `inst / ips` per iteration.

  Body state `{mu, ic, avg, inst, inst_interp, xs, ds, dxs}`: TED inputs,
  decisions and derivatives, oldest first. Options: machine options plus
  `:interp`, `:ted`, `:points` (slicer constellation, `{re, im}`),
  `:slicer` (`:sign` for BPSK/QPSK-like sign decisions, else `:nearest`),
  `:ips`, `:ipt`, `:ipo`, `:alpha`, `:beta`, `:min_avg`, `:max_avg`,
  `:inst_lo`, `:inst_hi`, `:wide`, `:dfmt` (fixed point: formats of
  products and of decisions), `:tap_frac`.
  """
  def symbol_sync(x, state, opts) do
    f = opts[:format]
    ip = opts[:interp]
    ted = Map.fetch!(@teds, opts[:ted])
    w = opts[:wide] || f
    tf = [tap_frac: opts[:tap_frac]]
    {ips, ipt, ipo} = {opts[:ips], opts[:ipt], opts[:ipo]}

    machine(x, state, opts, fn line, _wait, {mu, ic, avg, inst, iip, xs, ds, dxs} ->
      ic = Nx.remainder(Nx.add(ic, 1), ips)
      ted_in = Nx.equal(Nx.remainder(ic, ipt), 0)
      out_clk = Nx.equal(Nx.remainder(ic, ipo), 0)
      sym = Nx.equal(ic, 0)

      win = L.window(line, 0, ip.ntaps)
      y = Interp.interpolate(ip, win, mu, f, tf)

      xs = L.sel(ted_in, L.push(xs, y), xs)

      ds =
        if ted.slicer,
          do: L.sel(ted_in, L.push(ds, slice(y, opts)), ds),
          else: ds

      dxs =
        if ted.derivative do
          dy = ip |> Interp.differentiate(win, mu, f, tf ++ [raw: true]) |> Num.quantize(w)
          L.sel(ted_in, L.push(dxs, dy), dxs)
        else
          dxs
        end

      inputs = items(xs, ted.depth)

      inputs =
        if ted.lookahead do
          la = L.add(mu, look_ahead(inst, ips, ipt, f))
          {n_la, mu_la} = L.split(la, f)
          y_la = Interp.interpolate(ip, L.window(line, n_la, ip.ntaps), mu_la, f, tf)
          [y_la | inputs]
        else
          inputs
        end

      err = ted_error(opts[:ted], inputs, items(ds, ted.depth), items(dxs, 1), f, w)

      avg2 = L.add(avg, L.gain(err, opts[:beta], f)) |> L.clamp(opts[:min_avg], opts[:max_avg], f)
      inst2 = L.add(avg2, L.gain(err, opts[:alpha], f))
      inst2 = Nx.select(Nx.less_equal(inst2, L.const(0.0, f)), avg2, inst2)
      inst2 = L.clamp(inst2, opts[:inst_lo], opts[:inst_hi], f)
      iip2 = per_interp(inst2, ips, f)

      {avg, inst, iip} =
        {Nx.select(sym, avg2, avg), Nx.select(sym, inst2, inst), Nx.select(sym, iip2, iip)}

      {n, mu} = L.split(L.add(mu, iip), f)
      {[{out_clk, y}], n, {mu, Nx.as_type(ic, :s32), avg, inst, iip, xs, ds, dxs}}
    end)
  end

  # GNU Radio: interps_per_ted_input * (inst_period / interps_per_symbol)
  defp look_ahead(inst, ips, ipt, %{kind: :float} = f),
    do: Nx.multiply(Nx.tensor(ipt * 1.0, type: :f32), per_interp(inst, ips, f))

  defp look_ahead(inst, ips, ipt, f), do: Nx.multiply(per_interp(inst, ips, f), ipt)

  defp per_interp(inst, ips, %{kind: :float}),
    do: Nx.divide(inst, Nx.tensor(ips * 1.0, type: :f32))

  defp per_interp(inst, 1, _f), do: inst
  defp per_interp(inst, ips, _f), do: Nx.quotient(inst, ips)

  # The `n` newest items of a history line (oldest first in the line), newest first.
  defp items(hist, n) do
    len =
      hist
      |> L.cmap(&L.raw/1)
      |> then(fn
        {t, _} -> Nx.axis_size(t, 0)
        t -> Nx.axis_size(t, 0)
      end)

    for k <- 0..(min(n, len) - 1),
        do:
          L.cmap(hist, fn l ->
            L.like(l, l |> L.raw() |> Nx.slice([len - 1 - k], [1]) |> Nx.reshape({}))
          end)
  end

  @doc false
  # Hard decision of a TED input (complex, or the real part for real
  # input): sign decisions for BPSK/QPSK-like constellations, as GNU Radio's
  # sector slicers, otherwise the nearest point (first point on ties).
  def slice(y, opts) do
    f = opts[:format]
    df = opts[:dfmt] || f
    points = opts[:points]

    case {opts[:slicer], y} do
      {:sign, {re, im}} ->
        {pr, pi} = hd(points) |> then(fn {a, b} -> {abs(a), abs(b)} end)
        {sign_point(re, pr, df), sign_point(im, pi, df)}

      {:sign, re} ->
        sign_point(re, points |> hd() |> elem(0) |> abs(), df)

      {:nearest, _} ->
        nearest(y, points, df)
    end
  end

  # +p where x > 0, else -p
  defp sign_point(x, p, df) do
    c = Num.constant(p, df)
    ct = widen(L.raw(c))
    L.like(c, Nx.select(Nx.greater(L.raw(x), 0), ct, Nx.negate(ct)))
  end

  defp nearest(y, points, df) do
    {yr, yi} = if is_tuple(y), do: y, else: {y, nil}

    cands =
      for {pr, pi} <- points do
        cr = Num.constant(pr, df)
        ci = Num.constant(pi, df)
        dr = Num.sub(yr, cr)
        d = Num.mul(dr, dr)

        d =
          if yi do
            di = Num.sub(yi, ci)
            Num.add(d, Num.mul(di, di))
          else
            Num.add(d, Num.mul(ci, ci))
          end

        {if(yi, do: {cr, ci}, else: cr), d}
      end

    {best, _} =
      Enum.reduce(tl(cands), hd(cands), fn {p, d}, {bp, bd} ->
        better = Nx.less(L.raw(d), L.raw(bd))
        {L.sel(better, p, bp), L.like(bd, Nx.select(better, L.raw(d), L.raw(bd)))}
      end)

    best
  end

  # -- timing error detectors (GNU Radio's compute_error_cf / _ff) ---------------------------------

  defp re({r, _}), do: r
  defp re(r), do: r
  defp im({_, i}), do: i

  defp ted_error(:mueller_muller, [x0, x1 | _], [d0, d1 | _], _, _f, _w) do
    a = Num.sub(Num.mul(re(d1), re(x0)), Num.mul(re(d0), re(x1)))

    if is_tuple(x0),
      do: Num.add(a, Num.sub(Num.mul(im(d1), im(x0)), Num.mul(im(d0), im(x1)))),
      else: a
  end

  defp ted_error(:modified_mueller_muller, [x0, x1, x2 | _], [d0, d1, d2 | _], _, f, _w) do
    if is_tuple(x0) do
      u = Num.sub(re_conj(Num.csub(x0, x2), d1), re_conj(Num.csub(d0, d2), x1))
      num_clip(u, 1.0, f)
    else
      u = Num.sub(Num.mul(Num.sub(x0, x2), d1), Num.mul(Num.sub(d0, d2), x1))
      num_clip(half(u, f), 1.0, f)
    end
  end

  defp ted_error(:zero_crossing, [_x0, x1 | _], [d0, _d1, d2 | _], _, _f, _w),
    do: dot_re(fn a, b -> Num.sub(a, b) end, d2, d0, x1)

  defp ted_error(:gardner, [x0, x1, x2 | _], _, _, _f, _w),
    do: dot_re(fn a, b -> Num.sub(a, b) end, x2, x0, x1)

  defp ted_error(:early_late, [x0, x1, x2 | _], _, _, _f, _w),
    do: dot_re(fn a, b -> Num.sub(a, b) end, x0, x2, x1)

  defp ted_error(:dandrea_mengali_gen_msk, xs, _, _, f, w) do
    [x1, x3, x5, x7] = Enum.map([1, 3, 5, 7], &Enum.at(xs, &1))
    u = Num.sub(fourth(x1, x5, w), fourth(x3, x7, w))
    num_clip(u, 3.0, f)
  end

  defp ted_error(:mengali_dandrea_gmsk, [x0, x1, x2, x3 | _], _, _, f, w) do
    u = Num.sub(fourth(x1, x3, w), fourth(x0, x2, w))
    num_clip(u, 3.0, f)
  end

  defp ted_error(:signal_times_slope_ml, [x0 | _], _, [dx0], f, _w) do
    if is_tuple(x0),
      do: half(Num.add(Num.mul(re(x0), re(dx0)), Num.mul(im(x0), im(dx0))), f),
      else: Num.mul(x0, dx0)
  end

  defp ted_error(:signum_times_slope_ml, [x0 | _], _, [dx0], f, _w) do
    if is_tuple(x0),
      do: half(Num.add(sign_times(re(x0), re(dx0)), sign_times(im(x0), im(dx0))), f),
      else: sign_times(x0, dx0)
  end

  # re(a * conj(b)) = ar br + ai bi
  defp re_conj({ar, ai}, {br, bi}), do: Num.add(Num.mul(ar, br), Num.mul(ai, bi))

  # op(a, b).re * c.re (+ op(a, b).im * c.im for complex)
  defp dot_re(op, a, b, c) do
    r = Num.mul(op.(re(a), re(b)), re(c))
    if is_tuple(c), do: Num.add(r, Num.mul(op.(im(a), im(b)), im(c))), else: r
  end

  # re(a^2 conj(b^2)) (complex) or a a b b (real), products rounded to `w`
  defp fourth({_, _} = a, {_, _} = b, w) do
    a2 = a |> Num.cmul(a) |> Num.quantize(w)
    b2 = b |> Num.cmul(b) |> Num.quantize(w)
    re_conj(a2, b2)
  end

  defp fourth(a, b, w) do
    a |> Num.mul(a) |> Num.quantize(w) |> Num.mul(b) |> Num.quantize(w) |> Num.mul(b)
  end

  defp append(l, v) do
    t = L.raw(l)
    L.like(l, Nx.concatenate([t, v |> L.raw() |> Nx.as_type(Nx.type(t)) |> Nx.reshape({1})]))
  end

  @doc false
  def wait_bounds, do: {@wait_lo, @wait_hi}

  @doc false
  def one, do: 1 <<< L.p()
end
