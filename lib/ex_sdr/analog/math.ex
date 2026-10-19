defmodule ExSdr.Analog.Math do
  @moduledoc """
  Number-format-generic helpers for the analog blocks, on top of
  `NxFixed.Num`: angle words, an atan2 (CORDIC), an integer square root,
  constants with their own binary point, and a few small selectors.

  ## Angle words

  Phases and frequencies are signed 32 bit words, a full turn being 2^32
  (so pi is 2^31). They are plain integers in both instances, like the
  NCO phase word of `NxFixed.Num.oscillator/3`: adding and wrapping is
  exact and identical in float and fixed point. The instances differ only
  where a real number enters or leaves the word domain:

    * `words_from_real/3` and `real_from_words/3` (a real constant times an
      integer: float multiplies in f64, fixed point with an integer constant
      of about 30 significant bits),
    * `sincos_word/2` (float: exact cos and sin, fixed: the sine table of
      the format, `lut_bits`),
    * `atan2_word/3` (float: `Nx.atan2`, fixed: `cordic_atan2/4`).

  ## Constants

  `wconst/2` gives a real constant as a `NxFixed.Fixed` with its own binary
  point (about 24 significant bits), so gains like 1.0e-4 or 7.96 are not
  forced into the sample format. Products with it are exact (`NxFixed.Num.mul/2`)
  and the kernel decides where to round.
  """
  import Bitwise
  import Nx.Defn

  alias NxFixed.Fixed
  alias NxFixed.Num
  alias NxFixed.Num.Format

  @half_turn 1 <<< 31

  # -- integer plumbing -----------------------------------------------------------

  @doc "Raw integer tensor of a fixed-point value as s64."
  deftransform raw(%Fixed{t: t}), do: Nx.as_type(t, :s64)
  deftransform raw(%Nx.Tensor{} = t), do: Nx.as_type(t, :s64)

  @doc "Wraps an integer tensor (any width up to s64) to the s32 word range."
  deftransform wrap32(w) do
    w
    |> Nx.as_type(:s64)
    |> Nx.add(i64(@half_turn))
    |> Nx.bitwise_and(i64(0xFFFFFFFF))
    |> Nx.subtract(i64(@half_turn))
    |> Nx.as_type(:s32)
  end

  @doc """
  An s64 constant tensor. Constants of other expressions are s32 for the width
  check, so anything beyond 32 bits must be an explicit tensor.
  """
  deftransform i64(v), do: Nx.tensor(v, type: :s64)

  @doc "`x >> k` rounded to nearest (s64 tensor, any `k` below 63)."
  deftransform shift_round64(x, k) when k > 0,
    do: x |> Nx.add(i64(1 <<< (k - 1))) |> Nx.right_shift(k)

  @doc """
  Like `NxFixed.Num.quantize/2` but also for formats wider than 32 bits (the
  gain and power accumulators): rounds per the format, saturates, and gives
  the container of the format. Float: f32.
  """
  deftransform quantize({re, im}, format), do: {quantize(re, format), quantize(im, format)}

  deftransform quantize(%Fixed{t: t, frac: from}, %Format{kind: :fixed} = f) do
    t = Nx.as_type(t, :s64)
    to = f.frac

    t =
      cond do
        from > to and f.rounding == :nearest -> shift_round64(t, from - to)
        from > to -> Nx.right_shift(t, from - to)
        from < to -> Nx.left_shift(t, to - from)
        true -> t
      end

    hi = (1 <<< (f.bits - 1)) - 1
    t = t |> Nx.clip(i64(-hi - 1), i64(hi)) |> Nx.as_type(container(f.bits))
    %Fixed{t: t, frac: to}
  end

  deftransform quantize(x, %Format{kind: :float}), do: Nx.as_type(x, :f32)

  defp container(bits) when bits <= 8, do: :s8
  defp container(bits) when bits <= 16, do: :s16
  defp container(bits) when bits <= 32, do: :s32
  defp container(_bits), do: :s64

  @doc "Format with `extra` more fractional bits (and as many more total bits)."
  deftransform widen(%Format{kind: :fixed} = f, extra),
    do: %{f | bits: f.bits + extra, frac: f.frac + extra}

  deftransform widen(%Format{kind: :float} = f, _extra), do: f

  @doc "A fixed-point format with explicit width, beyond the 2..32 bits `NxFixed.Num.q/3` allows."
  deftransform wide_format(%Format{} = base, bits, frac),
    do: %{base | kind: :fixed, bits: bits, frac: frac}

  # -- constants ---------------------------------------------------------------------

  @doc """
  A real constant in `format`'s instance: an f32 scalar, or a `Fixed` with a
  binary point chosen for about `bits` (default 24) significant bits.
  """
  deftransform wconst(value, format), do: wconst(value, format, 24)

  deftransform wconst(value, %Format{kind: :float}, _bits), do: Nx.tensor(value * 1.0, type: :f32)

  deftransform wconst(value, %Format{kind: :fixed}, bits) do
    frac = const_frac(value, bits)
    %Fixed{t: Nx.tensor(round(value * Integer.pow(2, frac)), type: :s32), frac: frac}
  end

  defp const_frac(value, bits) when value == 0, do: bits
  defp const_frac(value, bits), do: bits - 1 - ceil_log2(abs(value))

  # Exponent e with 2^(e-1) < a <= 2^e... rounded up: the smallest e with a <= 2^e.
  defp ceil_log2(a) when a > 0, do: ceil(:math.log2(a))

  @doc "`value` as the raw integer at binary point `frac`, saturated to `bits` (signed)."
  deftransform to_raw(value, frac, bits) do
    hi = (1 <<< (bits - 1)) - 1
    lo = -(1 <<< (bits - 1))
    (value * Integer.pow(2, frac)) |> round() |> max(lo) |> min(hi)
  end

  @doc """
  Integer tensor `w` (s64) times the real constant `k`, rounded: a constant
  multiplier with an integer constant of about 30 significant bits and a
  rounding shift. Exact up to that constant's precision.
  """
  deftransform mul_real(w, k) when k == 0, do: Nx.multiply(w, 0)

  deftransform mul_real(w, k) do
    # kq has about 30 bits; s may be negative for large k.
    s = min(30 - ceil_log2(abs(k)), 40)
    kq = round(k * :math.pow(2, s))
    prod = Nx.multiply(Nx.as_type(w, :s64), Nx.tensor(kq, type: :s64))

    cond do
      s > 0 -> shift_round64(prod, s)
      s < 0 -> Nx.left_shift(prod, -s)
      true -> prod
    end
  end

  # -- real <-> angle words -------------------------------------------------------------

  @doc """
  Phase words for the real sample `x`: `x * k` rounded, where `k` is in words
  per unit (e.g. `sensitivity / (2 pi) * 2^32`). Not wrapped; the result is
  an s64 tensor.
  """
  deftransform words_from_real(x, k, %Format{kind: :float}) do
    x |> Nx.as_type(:f64) |> Nx.multiply(k) |> Nx.round() |> Nx.as_type(:s64)
  end

  deftransform words_from_real(%Fixed{t: t, frac: frac}, k, %Format{kind: :fixed}),
    do: mul_real(t, k / Integer.pow(2, frac))

  @doc """
  The real value `w * k` (`w` an s64 word tensor) in `format`: rounded and
  saturated for fixed point, f32 for float.
  """
  deftransform real_from_words(w, k, %Format{kind: :float}),
    do: w |> Nx.as_type(:f64) |> Nx.multiply(k) |> Nx.as_type(:f32)

  deftransform real_from_words(w, k, %Format{kind: :fixed} = f) do
    Num.quantize(%Fixed{t: mul_real(w, k * Integer.pow(2, f.frac)), frac: f.frac}, f)
  end

  @doc "Words per radian."
  deftransform words_per_rad, do: @half_turn / :math.pi()

  @doc "Radians per word."
  deftransform rad_per_word, do: :math.pi() / @half_turn

  # -- sin and cos of a phase word ---------------------------------------------------------

  @doc """
  `{cos, sin}` of the phase word `w` (any integer tensor; only the low 32
  bits count). Float: exact (f64, rounded to f32). Fixed point: the table of
  the format, addressed by the top 16 bits of the word (`lut_bits` of them
  used, rounded or truncated per format).
  """
  deftransform sincos_word(w, %Format{kind: :float}) do
    angle =
      w
      |> Nx.as_type(:s64)
      |> wrap32()
      |> Nx.as_type(:f64)
      |> Nx.multiply(2 * :math.pi() / 4_294_967_296)

    {angle |> Nx.cos() |> Nx.as_type(:f32), angle |> Nx.sin() |> Nx.as_type(:f32)}
  end

  deftransform sincos_word(w, %Format{kind: :fixed} = f) do
    idx =
      w
      |> Nx.as_type(:s64)
      |> Nx.right_shift(16)
      |> Nx.bitwise_and(0xFFFF)
      |> Nx.as_type(:u16)

    {nco, _} = Num.oscillator(idx, Nx.tensor(0, type: :u16), f)
    nco
  end

  # -- atan2 ----------------------------------------------------------------------------------

  @doc """
  The angle of the vector `(x, y)` as a word (pi = 2^31), range -2^31..2^31
  (`atan2(0, -1)` is +2^31), 0 for the zero vector. `x` and `y` are values of
  the same binary point; for fixed point only the integers matter.

  Options (fixed point): `:iterations` (default 16, residual angle below
  atan(2^-(n-1))) and `:guard` bits (default 8) the vector is shifted left by
  before the iterations, so tiny vectors are not lost in the shifts.
  """
  deftransform atan2_word(y, x, format), do: atan2_word(y, x, format, [])

  deftransform atan2_word(y, x, %Format{kind: :float}, _opts) do
    y |> Nx.atan2(x) |> Nx.multiply(words_per_rad()) |> Nx.round() |> Nx.as_type(:s64)
  end

  deftransform atan2_word(%Fixed{} = y, %Fixed{} = x, %Format{kind: :fixed}, opts),
    do:
      cordic_atan2(
        raw(y),
        raw(x),
        Keyword.get(opts, :iterations, 16),
        Keyword.get(opts, :guard, 8)
      )

  @doc """
  CORDIC in vectoring mode on integer tensors: reflects into the right half
  plane (adding or subtracting pi), then `iterations` rotations by
  atan(2^-i) steered by the sign of y. Only add, subtract, shift, compare and
  select, so it maps to a pipelined datapath; the angle table is constant.
  """
  deftransform cordic_atan2(y, x, iterations, guard) do
    x = x |> Nx.as_type(:s64) |> Nx.left_shift(guard)
    y = y |> Nx.as_type(:s64) |> Nx.left_shift(guard)

    zero = Nx.logical_and(Nx.equal(x, 0), Nx.equal(y, 0))
    reflect = Nx.less(x, 0)
    y_pos = Nx.greater_equal(y, 0)
    z0 = Nx.select(reflect, Nx.select(y_pos, i64(@half_turn), i64(-@half_turn)), i64(0))
    x0 = Nx.select(reflect, Nx.negate(x), x)
    y0 = Nx.select(reflect, Nx.negate(y), y)

    {_x, _y, z} =
      Enum.reduce(0..(iterations - 1), {x0, y0, z0}, fn i, {x, y, z} ->
        up = Nx.greater_equal(y, 0)
        xs = Nx.right_shift(x, i)
        ys = Nx.right_shift(y, i)
        a = atan_word(i)

        {Nx.select(up, Nx.add(x, ys), Nx.subtract(x, ys)),
         Nx.select(up, Nx.subtract(y, xs), Nx.add(y, xs)),
         Nx.select(up, Nx.add(z, a), Nx.subtract(z, a))}
      end)

    Nx.select(zero, i64(0), z)
  end

  defp atan_word(i), do: round(:math.atan(:math.pow(2, -i)) * words_per_rad())

  # -- magnitude ------------------------------------------------------------------------------

  @doc """
  Floor of the square root of `v` (non-negative s64 tensor below 4^`m`),
  digit by digit: compare, subtract, shift, select. `m` is the number of
  result bits.
  """
  deftransform isqrt(v, m) do
    v = Nx.as_type(v, :s64)

    {_num, res} =
      Enum.reduce((m - 1)..0//-1, {v, Nx.multiply(v, 0)}, fn k, {num, res} ->
        bit = 1 <<< (2 * k)
        bit = i64(bit)
        t = Nx.add(res, bit)
        ge = Nx.greater_equal(num, t)
        half = Nx.right_shift(res, 1)
        {Nx.select(ge, Nx.subtract(num, t), num), Nx.select(ge, Nx.add(half, bit), half)}
      end)

    res
  end

  @doc """
  Magnitude of the complex value `{re, im}` (same binary point, components
  within `limit_bits` signed bits). Float: `sqrt(re^2 + im^2)`. Fixed point:
  exact integer square root with `guard` extra fractional bits (result binary
  point `frac + guard`, default 4 guard bits).
  """
  deftransform cmag(x, format), do: cmag(x, format, 24, 4)

  deftransform cmag({re, im}, %Format{kind: :float}, _limit_bits, _guard),
    do: re |> Nx.multiply(re) |> Nx.add(Nx.multiply(im, im)) |> Nx.sqrt()

  deftransform cmag(
                 {%Fixed{frac: frac} = re, %Fixed{} = im},
                 %Format{kind: :fixed},
                 limit_bits,
                 guard
               ) do
    r = raw(re)
    i = raw(im)
    v = r |> Nx.multiply(r) |> Nx.add(Nx.multiply(i, i)) |> Nx.left_shift(2 * guard)
    %Fixed{t: isqrt(v, limit_bits + guard + 1), frac: frac + guard}
  end

  @doc "Magnitude of a real value (float: abs; fixed: abs, binary point kept)."
  deftransform rmag(%Fixed{t: t, frac: frac}),
    do: %Fixed{t: t |> Nx.as_type(:s64) |> Nx.abs(), frac: frac}

  deftransform rmag(%Nx.Tensor{} = x), do: Nx.abs(x)

  # -- selectors and clamps -----------------------------------------------------------------------

  @doc "`pred ? a : b` for two values of equal binary point."
  deftransform sel(pred, {a1, a2}, {b1, b2}), do: {sel(pred, a1, b1), sel(pred, a2, b2)}

  deftransform sel(pred, %Fixed{t: a, frac: frac}, %Fixed{t: b, frac: frac}),
    do: %Fixed{t: Nx.select(pred, a, b), frac: frac}

  deftransform sel(pred, a, b), do: Nx.select(pred, a, b)

  @doc "`x > 0`."
  deftransform positive?(%Fixed{t: t}), do: Nx.greater(t, 0)
  deftransform positive?(x), do: Nx.greater(x, 0)

  @doc "`a < b` for fixed values of any binary points or floats."
  deftransform lt(a, b), do: Nx.less(Num.sub(a, b) |> value_tensor(), 0)

  @doc "`a > b`."
  deftransform gt(a, b), do: Nx.greater(Num.sub(a, b) |> value_tensor(), 0)

  defp value_tensor(%Fixed{t: t}), do: t
  defp value_tensor(t), do: t

  @doc "Clamps `x` to the constants `lo..hi` (floats); for fixed point at `x`'s binary point."
  deftransform clamp(%Fixed{t: t, frac: frac} = x, lo, hi, bits) do
    %{
      x
      | t: Nx.clip(Nx.as_type(t, :s64), i64(to_raw(lo, frac, bits)), i64(to_raw(hi, frac, bits)))
    }
  end

  deftransform clamp(x, lo, hi, _bits), do: Nx.clip(x, lo, hi)

  @doc """
  Saturates `x` to `+-2^(limit_bits - 1)` raw units (fixed point; headroom
  for magnitudes that exceed the sample format), identity for float.
  """
  deftransform limit(%Fixed{t: t, frac: frac}, limit_bits) do
    hi = (1 <<< (limit_bits - 1)) - 1
    %Fixed{t: t |> Nx.as_type(:s64) |> Nx.clip(-hi, hi), frac: frac}
  end

  deftransform limit({re, im}, limit_bits), do: {limit(re, limit_bits), limit(im, limit_bits)}
  deftransform limit(x, _limit_bits), do: x

  @doc "Maximum over the elements of a vector, as a halving tree of `Nx.max/2`."
  deftransform max_reduce(%Nx.Tensor{shape: {1}} = v), do: Nx.reshape(v, {})

  deftransform max_reduce(%Nx.Tensor{shape: {n}} = v) do
    h = div(n, 2)
    a = Nx.slice(v, [0], [h])
    b = Nx.slice(v, [h], [h])
    m = Nx.max(a, b)
    m = if rem(n, 2) == 1, do: Nx.concatenate([m, Nx.slice(v, [n - 1], [1])]), else: m
    max_reduce(m)
  end

  @doc """
  The constant `value` with the binary point of `like` (an s64 `Fixed`), or
  an f32 scalar when `like` is a float.
  """
  deftransform const_like(value, %Fixed{frac: frac}),
    do: %Fixed{t: Nx.tensor(round(value * Integer.pow(2, frac)), type: :s64), frac: frac}

  deftransform const_like(value, %Nx.Tensor{}), do: Nx.tensor(value * 1.0, type: :f32)

  @doc "`x` where `keep` is true, else 0 (real or complex, either instance)."
  deftransform zero_where(pred, {re, im}), do: {zero_where(pred, re), zero_where(pred, im)}

  deftransform zero_where(pred, %Fixed{t: t} = x),
    do: %{x | t: Nx.select(pred, Nx.multiply(t, 0), t)}

  deftransform zero_where(pred, x), do: Nx.select(pred, Nx.multiply(x, 0), x)

  @doc "`[x | line]` as a vector (newest first), for real values of either instance."
  deftransform prepend(%Fixed{t: t, frac: frac}, %Fixed{t: line}),
    do: %Fixed{t: Nx.concatenate([Nx.reshape(t, {1}), line]), frac: frac}

  deftransform prepend(x, line), do: Nx.concatenate([Nx.reshape(x, {1}), line])

  @doc "The first `n` elements of a vector value."
  deftransform take_first(%Fixed{t: t, frac: frac}, n),
    do: %Fixed{t: Nx.slice(t, [0], [n]), frac: frac}

  deftransform take_first(x, n), do: Nx.slice(x, [0], [n])

  @doc "The last element of a vector value."
  deftransform last(%Fixed{t: t, frac: frac}, n),
    do: %Fixed{t: t |> Nx.slice([n - 1], [1]) |> Nx.reshape({}), frac: frac}

  deftransform last(x, n), do: x |> Nx.slice([n - 1], [1]) |> Nx.reshape({})
end
