defmodule ExSdr.Math.Stream do
  @moduledoc """
  Sample-domain kernels of the math family, written once against
  `NxFixed.Num` and instantiated by `:format` (float or fixed point).

  Every kernel is `step(sample, state, params..., opts) -> {output, next}`
  with `opts` a keyword list holding `:format` and any static options. A
  stateless kernel takes `{}` as state.

  Conventions shared by all kernels:

    * Products and sums are exact (`NxFixed.Num`); a single `quantize/2` at
      the output rounds per format and saturates to the format's width.
    * Constants arrive as parameters already quantized by
      `ExSdr.Math.param/2`, so they are exact in the format's grid.
    * Pairs (two-input blocks) are the sample `{a, b}`; complex values are
      `{re, im}`.
  """
  import Nx.Defn
  import Bitwise

  alias NxFixed.Fixed
  alias NxFixed.Num

  @doc "Complex or real sample times a real constant `k`."
  defn multiply_const(x, state, k, opts \\ []) do
    opts = keyword!(opts, [:format])
    {scale(qin(x, opts[:format]), k, opts[:format]), state}
  end

  @doc "Sample plus a constant (real, or `{re, im}` for complex samples)."
  defn add_const(x, state, c, opts \\ []) do
    opts = keyword!(opts, [:format])
    {offset(qin(x, opts[:format]), c, opts[:format]), state}
  end

  @doc "Complex conjugate `{re, -im}`."
  defn conjugate_cc(x, state, opts \\ []) do
    opts = keyword!(opts, [:format])
    {conj(qin(x, opts[:format]), opts[:format]), state}
  end

  @doc "Squared magnitude `re^2 + im^2`."
  defn complex_to_mag_squared(x, state, opts \\ []) do
    opts = keyword!(opts, [:format])
    {mag_squared(qin(x, opts[:format]), opts[:format]), state}
  end

  @doc "Real part."
  defn complex_to_real(x, state, opts \\ []) do
    opts = keyword!(opts, [:format])
    {real_part(x, opts[:format]), state}
  end

  @doc "Imaginary part."
  defn complex_to_imag(x, state, opts \\ []) do
    opts = keyword!(opts, [:format])
    {imag_part(x, opts[:format]), state}
  end

  @doc "Absolute value; the most negative value saturates to the largest."
  defn abs_ff(x, state, opts \\ []) do
    opts = keyword!(opts, [:format])
    {absolute(qin(x, opts[:format]), opts[:format]), state}
  end

  @doc """
  Hysteresis threshold with constants `lo` and `hi`: the output (u8, 0 or
  1) becomes 1 when `x > hi`, 0 when `x < lo`, and keeps its value in
  between. The state is that output bit.
  """
  defn threshold_ff(x, state, lo, hi, _opts \\ []) do
    y = hysteresis(x, state, lo, hi)
    {y, y}
  end

  @doc """
  Moving sum with constant taps `opts[:taps]` (the window, newest first,
  e.g. `scale` repeated `length` times): the delay line is the state, one
  per component. Equivalent to GNU Radio's `moving_average_*`.
  """
  defn moving_average(x, lines, opts \\ []) do
    opts = keyword!(opts, [:format, :taps])
    fir_step(x, lines, opts[:taps], opts[:format])
  end

  @doc "Delay by `opts[:delay]` samples (at least 1); the state is the delay line."
  defn delay(x, line, opts \\ []) do
    opts = keyword!(opts, [:format, :delay])
    delay_step(x, line, opts[:delay])
  end

  # Two-input blocks: the sample is {a, b}.

  @doc "a + b"
  defn add_pair(pair, state, opts \\ []) do
    opts = keyword!(opts, [:format])
    {pair_op(qin(pair, opts[:format]), :add, opts[:format]), state}
  end

  @doc "a - b"
  defn sub_pair(pair, state, opts \\ []) do
    opts = keyword!(opts, [:format])
    {pair_op(qin(pair, opts[:format]), :sub, opts[:format]), state}
  end

  @doc "a * b"
  defn mul_pair(pair, state, opts \\ []) do
    opts = keyword!(opts, [:format])
    {pair_op(qin(pair, opts[:format]), :mul, opts[:format]), state}
  end

  @doc "a / b (truncated and rounded per format; saturates on division by zero)."
  defn div_pair(pair, state, opts \\ []) do
    opts = keyword!(opts, [:format])
    {pair_op(qin(pair, opts[:format]), :div, opts[:format]), state}
  end

  @doc "Two floats `{re, im}` into one complex sample (both quantized to the format)."
  defn float_to_complex(pair, state, opts \\ []) do
    opts = keyword!(opts, [:format])
    {re, im} = pair
    {{qin(re, opts[:format]), qin(im, opts[:format])}, state}
  end

  # One-sample conversions between bytes and numbers.

  @doc "Signed char (the byte read as two's complement) times `scale` (the registry passes 1 / GNU Radio's scale)."
  defn char_to_float(x, state, scale, opts \\ []) do
    opts = keyword!(opts, [:format])
    {from_char(x, scale, opts[:format]), state}
  end

  @doc """
  `round(x * scale)` saturated to a signed byte (s8). Fixed-point output
  is in the format `NxFixed.Num.q(8, 0)`.
  """
  defn float_to_char(x, state, scale, opts \\ []) do
    opts = keyword!(opts, [:format])
    {to_char(x, scale, opts[:format]), state}
  end

  @doc """
  `round(x * scale)` saturated to a signed 16-bit short (s16). Fixed-point
  output is in the format `NxFixed.Num.q(16, 0)`.
  """
  defn float_to_short(x, state, scale, opts \\ []) do
    opts = keyword!(opts, [:format])
    {to_short(x, scale, opts[:format]), state}
  end

  # Float only (no fixed-point instance yet): sqrt, atan2 and log.

  @doc "Magnitude `sqrt(re^2 + im^2)`, float only."
  defn complex_to_mag(x, state, _opts \\ []) do
    {re, im} = x
    {Nx.sqrt(re * re + im * im), state}
  end

  @doc "Phase `atan2(im, re)` in radians, float only."
  defn complex_to_arg(x, state, _opts \\ []) do
    {re, im} = x
    {Nx.atan2(im, re), state}
  end

  @doc "`n * log10(x) + k`, float only."
  defn nlog10_ff(x, state, n, k, _opts \\ []) do
    {n * Nx.log(x) / Nx.log(10.0) + k, state}
  end

  @doc "Magnitude and phase `{m, p}` to `m * cos(p), m * sin(p)`, float only."
  defn magphase_to_complex(pair, state, _opts \\ []) do
    {m, p} = pair
    {{m * Nx.cos(p), m * Nx.sin(p)}, state}
  end

  @doc "Passes the item through (stream to vector framing, flat streams)."
  defn identity(x, state, _opts \\ []), do: {x, state}

  # -- helpers -----------------------------------------------------------------------

  # Inputs are already in the format; quantizing them states that, so the
  # width check bounds the products and sums by the format, not the container.
  deftransformp qin(x, format), do: Num.quantize(x, format)

  deftransformp scale({re, im}, k, format),
    do: {scale(re, k, format), scale(im, k, format)}

  deftransformp scale(x, k, format), do: x |> Num.mul(k) |> Num.quantize(format)

  deftransformp offset({re, im}, {cr, ci}, format),
    do: {offset(re, cr, format), offset(im, ci, format)}

  deftransformp offset({re, im}, c, format) when is_number(c) or is_struct(c, Fixed),
    do: {offset(re, c, format), offset(im, c, format)}

  deftransformp offset(x, c, format), do: x |> Num.add(c) |> Num.quantize(format)

  deftransformp conj({re, im}, format), do: {Num.quantize(re, format), negate(im, format)}

  deftransformp negate(x, format),
    do: Num.sub(Num.zeros({}, format), x) |> Num.quantize(format)

  deftransformp mag_squared({re, im}, format),
    do: Num.add(Num.mul(re, re), Num.mul(im, im)) |> Num.quantize(format)

  deftransformp real_part({re, _im}, format), do: Num.quantize(re, format)
  deftransformp imag_part({_re, im}, format), do: Num.quantize(im, format)

  deftransformp absolute(%Fixed{t: t, frac: frac}, format) do
    %Fixed{t: neg} = Num.sub(Num.zeros({}, format), %Fixed{t: t, frac: frac})
    %Fixed{t: t_abs} = Nx.select(Nx.less(t, 0), neg, t) |> then(&%Fixed{t: &1, frac: frac})
    %Fixed{t: t_abs, frac: frac} |> Num.quantize(format)
  end

  deftransformp absolute(x, %Num.Format{kind: :float}), do: Nx.abs(x)

  deftransformp hysteresis(x, state, lo, hi) do
    above = Nx.greater(raw(Num.sub(x, hi)), 0)
    below = Nx.less(raw(Num.sub(x, lo)), 0)

    Nx.select(above, Nx.tensor(1, type: :u8), Nx.select(below, Nx.tensor(0, type: :u8), state))
  end

  deftransformp fir_step({re, im}, {line_re, line_im}, taps, format) do
    {acc_re, next_re} = Num.fir(re, line_re, taps, format)
    {acc_im, next_im} = Num.fir(im, line_im, taps, format)
    {Num.quantize({acc_re, acc_im}, format), {next_re, next_im}}
  end

  deftransformp fir_step(x, line, taps, format) do
    {acc, next} = Num.fir(x, line, taps, format)
    {Num.quantize(acc, format), next}
  end

  deftransformp delay_step({re, im}, {line_re, line_im}, d) do
    {y_re, n_re} = delay_step(re, line_re, d)
    {y_im, n_im} = delay_step(im, line_im, d)
    {{y_re, y_im}, {n_re, n_im}}
  end

  deftransformp delay_step(x, line, d) do
    lt = raw(line)
    out = lt |> Nx.slice([d - 1], [1]) |> Nx.reshape({})
    window = Nx.concatenate([Nx.reshape(raw(x), {1}), Nx.slice(lt, [0], [d - 1])])
    {wrap(out, line), wrap(window, line)}
  end

  deftransformp pair_op({a, b}, op, format) when is_tuple(a),
    do: complex_pair(a, b, op, format)

  deftransformp pair_op({a, b}, op, format) do
    case op do
      :add -> Num.add(a, b) |> Num.quantize(format)
      :sub -> Num.sub(a, b) |> Num.quantize(format)
      :mul -> Num.mul(a, b) |> Num.quantize(format)
      :div -> divide(a, b, format)
    end
  end

  defp complex_pair(a, b, :add, format), do: componentwise(a, b, &Num.add/2, format)
  defp complex_pair(a, b, :sub, format), do: componentwise(a, b, &Num.sub/2, format)
  defp complex_pair(a, b, :mul, format), do: Num.cmul(a, b) |> Num.quantize(format)

  defp complex_pair(a, b, :div, format) do
    {br, bi} = b
    {nr, ni} = Num.cmul(a, conj_exact(b, format))
    den = Num.add(Num.mul(br, br), Num.mul(bi, bi))
    {divide(nr, den, format), divide(ni, den, format)}
  end

  defp componentwise({ar, ai}, {br, bi}, op, format),
    do: {op.(ar, br) |> Num.quantize(format), op.(ai, bi) |> Num.quantize(format)}

  defp conj_exact({re, im}, format), do: {re, Num.sub(Num.zeros({}, format), im)}

  @doc false
  # a / b at the format's precision. Float: plain division. Fixed: exact
  # integer quotient of the aligned raw values, rounded per format.
  deftransformp divide(a, b, %Num.Format{kind: :float}), do: Nx.divide(a, b)

  deftransformp divide(a, b, %Num.Format{kind: :fixed} = format) do
    %Fixed{t: at, frac: af} = a
    %Fixed{t: bt, frac: bf} = b
    shift = bf + format.frac - af
    num = Nx.left_shift(Nx.as_type(at, :s64), max(shift, 0))
    den = Nx.left_shift(Nx.as_type(bt, :s64), max(-shift, 0))
    quotient = fixed_quotient(num, den, format.rounding)
    %Fixed{t: quotient, frac: format.frac} |> Num.quantize(format)
  end

  # Only operations the width checker knows (no sign/1): magnitudes by abs,
  # the sign from comparisons.
  defp fixed_quotient(num, den, rounding) do
    zero_den = Nx.equal(den, 0)
    n = Nx.abs(num)
    # divisor >= 1: the width checker needs a divisor that cannot be zero
    d = Nx.max(Nx.abs(den), 1)

    magnitude =
      if rounding == :nearest,
        do: Nx.quotient(Nx.add(Nx.multiply(n, 2), d), Nx.multiply(d, 2)),
        else: Nx.quotient(n, d)

    negative = Nx.not_equal(Nx.less(num, 0), Nx.less(den, 0))
    signed = Nx.select(negative, Nx.negate(magnitude), magnitude)

    # division by zero saturates by the sign of the numerator (0 stays 0)
    big = Nx.tensor(1 <<< 40, type: :s64)
    saturated = Nx.select(Nx.less(num, 0), Nx.negate(big), Nx.select(Nx.greater(num, 0), big, 0))
    Nx.select(zero_den, saturated, signed)
  end

  deftransformp from_char(x, scale, format) do
    s = Nx.select(Nx.greater(x, 127), Nx.subtract(Nx.as_type(x, :s32), 256), Nx.as_type(x, :s32))

    case format do
      %Num.Format{kind: :float} -> Nx.multiply(Nx.as_type(s, :f32), scale)
      _ -> %Fixed{t: s, frac: 0} |> Num.mul(scale) |> Num.quantize(format)
    end
  end

  deftransformp to_char(x, scale, %Num.Format{kind: :float}) do
    x |> Nx.multiply(scale) |> Nx.round() |> Nx.clip(-128, 127) |> Nx.as_type(:s8)
  end

  deftransformp to_char(x, scale, %Num.Format{kind: :fixed}) do
    x |> Num.mul(scale) |> Num.quantize(Num.q(8, 0, rounding: :nearest))
  end

  deftransformp to_short(x, scale, %Num.Format{kind: :float}) do
    x |> Nx.multiply(scale) |> Nx.round() |> Nx.clip(-32768, 32767) |> Nx.as_type(:s16)
  end

  deftransformp to_short(x, scale, %Num.Format{kind: :fixed}) do
    x |> Num.mul(scale) |> Num.quantize(Num.q(16, 0, rounding: :nearest))
  end

  defp raw(%Fixed{t: t}), do: t
  defp raw(t), do: t

  defp wrap(t, %Fixed{frac: frac}), do: %Fixed{t: t, frac: frac}
  defp wrap(t, _), do: t
end
