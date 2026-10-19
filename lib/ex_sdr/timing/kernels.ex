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

  defp append(l, v) do
    t = L.raw(l)
    L.like(l, Nx.concatenate([t, v |> L.raw() |> Nx.as_type(Nx.type(t)) |> Nx.reshape({1})]))
  end

  @doc false
  def wait_bounds, do: {@wait_lo, @wait_hi}

  @doc false
  def one, do: 1 <<< L.p()
end
