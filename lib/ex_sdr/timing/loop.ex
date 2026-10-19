defmodule ExSdr.Timing.Loop do
  @moduledoc """
  Arithmetic of the timing loops (positions, fractional delays, periods,
  loop filter state), written once for both instances:

    * float (`NxFixed.Num.float/0`): `f32` tensors, in the order GNU Radio
      computes them (`f64` where GNU Radio keeps a `double`, see `const/3`),
    * fixed point (`NxFixed.Num.q/3`): `s64` words with `p/0` = 32
      fractional bits. Adding, splitting into integer and fraction, and
      clipping are exact, so a position never drifts and the block is
      bit-exact on every backend. A loop word covers +-2^31 samples.

  Values that come from samples (timing errors) are `NxFixed.Num` values;
  `gain/3` multiplies them by a real constant into the loop domain. In
  fixed point the constant has 22 significant bits and the product is
  rounded once to the word.

  Also signal plumbing shared by the kernels (`{re, im}` complex values,
  `NxFixed.Fixed` or `f32` reals).
  """
  import Bitwise

  alias NxFixed.Fixed
  alias NxFixed.Num
  alias NxFixed.Num.Format
  alias NxFixed.Primitives, as: P

  @p 32

  @doc "Fractional bits of a fixed-point loop word."
  def p, do: @p

  # -- f32 on the host -------------------------------------------------------------

  @doc "`x` rounded to the nearest f32 (as an Elixir float)."
  def f32(x) when is_number(x) do
    <<y::float-32>> = <<x * 1.0::float-32>>
    y
  end

  # -- constants ------------------------------------------------------------------------

  @doc """
  A loop constant: f32 (or `type: :f64`) in the float instance, a rounded
  s64 word in fixed point.
  """
  def const(v, f, opts \\ [])

  def const(v, %Format{kind: :float}, opts),
    do: Nx.tensor(v * 1.0, type: Keyword.get(opts, :type, :f32))

  def const(v, %Format{kind: :fixed}, _opts), do: Nx.tensor(round(v * (1 <<< @p)), type: :s64)

  @doc "An s64 integer constant (above 32 bits Nx constants must be explicit tensors)."
  def i64(v), do: Nx.tensor(v, type: :s64)

  @doc "An s32 integer constant."
  def i32(v), do: Nx.tensor(v, type: :s32)

  # -- loop operations ------------------------------------------------------------

  @doc "Exact (fixed) or rounded like GNU Radio (float) sum."
  def add(a, b), do: Nx.add(a, b)

  @doc "Difference."
  def sub(a, b), do: Nx.subtract(a, b)

  @doc """
  `x * c` for a real constant `c`. Float: one f32 multiply (`c` rounded to
  f32 first). Fixed: `c` with 22 significant bits, rounded to the word.
  """
  def mulc(x, c, %Format{kind: :float}), do: Nx.multiply(x, Nx.tensor(f32(c), type: Nx.type(x)))

  def mulc(x, c, %Format{kind: :fixed}) do
    {ci, g} = const_bits(c)
    if ci == 0, do: Nx.multiply(x, i64(0)), else: x |> Nx.multiply(i64(ci)) |> round_shift(g)
  end

  @doc "`n * c` for an s32 tensor `n` and an integer constant `c`, in the loop domain."
  def mulc_int(n, c, %Format{kind: :float}), do: Nx.multiply(Nx.as_type(n, :f32), c * 1.0)
  def mulc_int(n, c, %Format{kind: :fixed}), do: Nx.multiply(Nx.as_type(n, :s64), i64(c <<< @p))

  @doc """
  `c * err` with `err` a `NxFixed.Num` value (a timing error) and `c` a
  real constant, in the loop domain. Float: one f32 multiply.
  """
  def gain(err, c, %Format{kind: :float}), do: Nx.multiply(Nx.tensor(f32(c), type: :f32), err)

  def gain(%Fixed{t: t, frac: frac}, c, %Format{kind: :fixed}) do
    {ci, g} = const_bits(c)

    if ci == 0 do
      Nx.multiply(Nx.as_type(t, :s64), i64(0))
    else
      prod = Nx.multiply(Nx.as_type(t, :s64), i64(ci))
      shift = frac + g - @p
      if shift > 0, do: round_shift(prod, shift), else: Nx.left_shift(prod, i64(-shift))
    end
  end

  @doc "A `NxFixed.Num` value in the loop domain (exact unless it has more than 32 fractional bits)."
  def from_num(x, %Format{kind: :float}), do: x

  def from_num(%Fixed{t: t, frac: frac}, %Format{kind: :fixed}) do
    t = Nx.as_type(t, :s64)

    cond do
      frac == @p -> t
      frac < @p -> Nx.left_shift(t, i64(@p - frac))
      true -> round_shift(t, frac - @p)
    end
  end

  # c = ci / 2^g with |ci| < 2^22
  defp const_bits(c) when c == 0, do: {0, 0}

  defp const_bits(c) do
    e = :math.log2(abs(c)) |> Float.ceil() |> trunc()
    g = 21 - e
    {round(c * :math.pow(2, g)), g}
  end

  defp round_shift(x, k) when k > 0, do: x |> Nx.add(i64(1 <<< (k - 1))) |> Nx.right_shift(i64(k))
  defp round_shift(x, 0), do: x

  @doc """
  `{n, frac}`: the integer part (s32) and the fraction in `[0, 1)` of `x`
  (GNU Radio's `n = floor(x); frac = x - n`, exact in both instances).
  """
  def split(x, %Format{kind: :float}) do
    n = Nx.floor(x)
    {Nx.as_type(n, :s32), Nx.subtract(x, n)}
  end

  def split(x, %Format{kind: :fixed}) do
    {x |> Nx.right_shift(i64(@p)) |> Nx.as_type(:s32), Nx.bitwise_and(x, i64((1 <<< @p) - 1))}
  end

  @doc """
  GNU Radio's `branchless_clip(x, c) = 0.5 * (|x + c| - |x - c|)`: in float
  with GNU Radio's roundings (`x + c` and `x - c` are rounded to f32), in
  fixed point an exact clip to `[-c, c]`.
  """
  def clip(x, c, %Format{kind: :float} = f) do
    c = const(c, f)
    a = Nx.abs(Nx.add(x, c))
    b = Nx.abs(Nx.subtract(x, c))
    Nx.multiply(Nx.subtract(a, b), Nx.tensor(0.5, type: Nx.type(x)))
  end

  def clip(x, c, %Format{kind: :fixed} = f) do
    c = round(c * (1 <<< @p))
    _ = f
    Nx.clip(x, i64(-c), i64(c))
  end

  @doc "Clip to `[lo, hi]` (real constants)."
  def clamp(x, lo, hi, f), do: Nx.clip(x, const(lo, f), const(hi, f))

  @doc """
  The filter arm for a fractional delay `mu` in `[0, 1]` with `n` arms per
  sample: GNU Radio's `rint(mu * n)` (ties to even, f32 product) in float,
  `round(mu * n)` (ties up) in fixed point. Result s32 in `0..n`.
  """
  def arm(mu, n, %Format{kind: :float}) do
    x = Nx.multiply(Nx.as_type(mu, :f32), Nx.tensor(n * 1.0, type: :f32))
    fl = Nx.floor(x)
    d = Nx.subtract(x, fl)
    odd = Nx.equal(Nx.remainder(fl, Nx.tensor(2.0, type: :f32)), Nx.tensor(1.0, type: :f32))
    up = Nx.logical_or(Nx.greater(d, 0.5), Nx.logical_and(Nx.equal(d, 0.5), odd))
    r = Nx.select(up, Nx.add(fl, 1.0), fl)
    r |> Nx.as_type(:s32) |> Nx.clip(0, n)
  end

  def arm(mu, n, %Format{kind: :fixed}) do
    mu
    |> Nx.clip(i64(0), i64(1 <<< @p))
    |> Nx.multiply(i64(n))
    |> Nx.add(i64(1 <<< (@p - 1)))
    |> Nx.right_shift(i64(@p))
    |> Nx.as_type(:s32)
  end

  @doc "A loop value as a real number in the float instance, `NxFixed.Fixed` (frac 32) in fixed point."
  def to_num(x, %Format{kind: :float}), do: x
  def to_num(x, %Format{kind: :fixed}), do: %Fixed{t: x, frac: @p}

  # -- state bounds ------------------------------------------------------------------

  @doc """
  States the range of a state value for the width check (state arguments
  otherwise span their whole type). The value is in range by
  construction, so this changes nothing; in hardware it is free.
  Float values pass unchanged.
  """
  def bound(x, lo, hi, f)

  def bound(x, _lo, _hi, %Format{kind: :float}), do: x

  def bound(x, lo, hi, %Format{kind: :fixed}),
    do: Nx.clip(x, const(lo, %Format{kind: :fixed}), const(hi, %Format{kind: :fixed}))

  @doc "Integer state (counters) bounded to `lo..hi`."
  def bound_int(x, lo, hi),
    do: Nx.clip(x, Nx.tensor(lo, type: Nx.type(x)), Nx.tensor(hi, type: Nx.type(x)))

  # -- signals ------------------------------------------------------------------------

  @doc "Maps over the components of a complex value (or applies to a real one)."
  def cmap({re, im}, fun), do: {fun.(re), fun.(im)}
  def cmap(x, fun), do: fun.(x)

  @doc "Zips two signals of the same kind."
  def zip({ar, ai}, {br, bi}, fun), do: {fun.(ar, br), fun.(ai, bi)}
  def zip(a, b, fun), do: fun.(a, b)

  @doc "Raw tensor of a real value."
  def raw(%Fixed{t: t}), do: t
  def raw(t), do: t

  @doc "`t` with the binary point of `like`."
  def like(%Fixed{frac: frac}, t), do: %Fixed{t: t, frac: frac}
  def like(_, t), do: t

  @doc "A sample (real or complex) saturated to the format, so its range is known."
  def bound_sample(x, f), do: Num.quantize(x, f)

  @doc """
  Selects between two signals (any matching containers of reals, possibly
  nested tuples) by a scalar predicate.
  """
  def sel(pred, a, b) when is_tuple(a) and not is_struct(a) do
    a
    |> Tuple.to_list()
    |> Enum.zip(Tuple.to_list(b))
    |> Enum.map(fn {x, y} -> sel(pred, x, y) end)
    |> List.to_tuple()
  end

  def sel(pred, %Fixed{t: a, frac: frac}, %Fixed{t: b, frac: frac}),
    do: %Fixed{t: Nx.select(pred, a, Nx.as_type(b, Nx.type(a))), frac: frac}

  def sel(pred, a, b), do: Nx.select(pred, a, b)

  @doc "Shifts `x` (one item) into the end of the line (oldest first)."
  def push(line, x) do
    zip(line, x, fn l, v ->
      t = raw(l)
      n = Nx.axis_size(t, 0)
      v = v |> raw() |> Nx.as_type(Nx.type(t)) |> Nx.reshape({1})
      if n == 1, do: like(l, v), else: like(l, Nx.concatenate([Nx.slice(t, [1], [n - 1]), v]))
    end)
  end

  @doc "`n` items of `line` from `start` (an integer)."
  def window(line, start, n) when is_integer(start),
    do: cmap(line, fn l -> like(l, Nx.slice(raw(l), [start], [n])) end)

  # A dynamic start (a scalar tensor), clipped to the line.
  def window(line, start, n) do
    cmap(line, fn l ->
      t = raw(l)
      len = Nx.axis_size(t, 0)
      idx = Nx.add(iota(n), Nx.as_type(start, :s32)) |> Nx.clip(0, len - 1)
      like(l, Nx.take(t, idx))
    end)
  end

  @doc "Zero lanes for `max` output items like `value` (one item)."
  def lanes(value, max) do
    cmap(value, fn v ->
      t = raw(v)
      like(v, Nx.broadcast(Nx.tensor(0, type: Nx.type(t)), {max}))
    end)
  end

  @doc "Writes `value` into lane `count` where `ok`; returns `{lanes, count + ok}`."
  def put_lane(lanes, count, value, ok) do
    lanes =
      zip(lanes, value, fn l, v ->
        t = raw(l)
        max = Nx.axis_size(t, 0)
        hit = Nx.logical_and(Nx.equal(iota(max), count), ok)
        v = v |> raw() |> Nx.as_type(Nx.type(t)) |> Nx.broadcast({max})
        like(l, Nx.select(hit, v, t))
      end)

    {lanes, Nx.add(count, Nx.as_type(ok, :s32))}
  end

  @doc "Zero of a sample in format `f` (real or complex like `kind`)."
  def zero(:complex, f), do: {Num.zeros({}, f), Num.zeros({}, f)}
  def zero(:real, f), do: Num.zeros({}, f)

  @doc "Zero line of `n` items."
  def zero_line(:complex, f, n), do: {Num.zeros({n}, f), Num.zeros({n}, f)}
  def zero_line(:real, f, n), do: Num.zeros({n}, f)

  @doc false
  def shift_round(x, k), do: P.shift_round(x, k)

  # 0..n-1 as a constant (the width check has no iota)
  defp iota(n), do: Nx.tensor(Enum.to_list(0..(n - 1)), type: :s32, backend: Nx.BinaryBackend)
end
