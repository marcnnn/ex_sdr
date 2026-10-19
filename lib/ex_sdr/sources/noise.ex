defmodule ExSdr.Sources.Noise do
  @moduledoc """
  Noise sources with a deterministic generator, so a stream is reproducible
  from its seed and can be split anywhere.

  ## Generator

  The state is one `u32` word of a 32 bit xorshift generator
  (`x ^= x << 13; x ^= x >> 17; x ^= x << 5`, period `2^32 - 1`, three
  shifts and three XORs: a few LUTs in hardware). Each real draw advances it
  once; complex samples use two draws. Seeds are mixed by `seed_state/1`
  (never zero). This is not GNU Radio's RNG: the streams cannot match, the
  distributions do (see `docs/blocks/sources.md`).

  ## Distributions

  A draw `w` (32 bits) gives:

    * `:uniform`: the top `frac + 1` bits as a signed value in
      `[-1, 1)` times `amplitude` (float: 24 bits); GNU Radio's range
    * `:gaussian`, `:laplacian`: a table of `2^table_bits` quantiles
      (inverse CDF at `(i + 0.5) / 2^table_bits`, amplitude included),
      addressed by the top bits. With 14 table bits the tails stop at
      4.0 sigma (gaussian) or 9.7 scale units (laplacian), and the
      distribution is a staircase of 16384 levels; the variance is within
      about 0.1 percent of the exact one. Gaussian: standard deviation
      `amplitude` per real sample. Complex samples of every type have both
      components scaled by `1 / sqrt(2)` (total power `amplitude^2` for
      Gaussian), as GNU Radio's `noise_source_c` does (checked in the tests).
    * `:impulse`: sparse positive spikes. GNU Radio's `impulse(9)`, as
      measured on its output, is an exponential variate (mean 10/7) that is
      kept when it exceeds `impulse_factor` (9) and zero otherwise, so one
      sample in about 560 is nonzero, with value `9 + Exp(mean 1.43)`. The
      table has 2^16 entries (about 117 of them nonzero) and is the same
      table method as above. This matches the measured statistics
      (probability, mean, second moment), not GNU Radio's source code.

  Amplitude is a compile time constant. The fixed-point table holds
  quantised values, so the output is exactly reproducible on every backend.
  """
  import Nx.Defn
  import Bitwise
  import NxFixed.Num

  alias NxFixed.Fixed
  alias NxFixed.Num.Format

  @mask 0xFFFFFFFF

  @doc "The generator state for `seed`: a hashed, nonzero `u32` tensor."
  def seed_state(seed) when is_integer(seed) do
    x = seed + 0x9E3779B9 &&& @mask
    x = bxor(x, x >>> 16) * 0x85EBCA6B &&& @mask
    x = bxor(x, x >>> 13) * 0xC2B2AE35 &&& @mask
    x = bxor(x, x >>> 16)
    Nx.tensor(if(x == 0, do: 1, else: x), type: :u32)
  end

  @doc "One xorshift32 step on a `u32` tensor."
  deftransform next(state) do
    m = Nx.tensor(@mask, type: :s64)
    x = Nx.as_type(state, :s64)
    x = Nx.bitwise_xor(x, Nx.bitwise_and(Nx.left_shift(x, 13), m))
    x = Nx.bitwise_xor(x, Nx.right_shift(x, 17))
    x = Nx.bitwise_xor(x, Nx.bitwise_and(Nx.left_shift(x, 5), m))
    Nx.as_type(x, :u32)
  end

  @doc """
  A host side uniform generator on the same recurrence, for parameters
  (fading phases): `uniform.()` returns `{x, uniform}` with `x` in `[0, 1)`.
  """
  def host_uniform(seed) do
    state = seed |> seed_state() |> Nx.to_number()
    fn -> host_draw(state) end
  end

  defp host_draw(x) do
    x = bxor(x, x <<< 13 &&& @mask)
    x = bxor(x, x >>> 17)
    x = bxor(x, x <<< 5 &&& @mask)
    {x / 4_294_967_296, fn -> host_draw(x) end}
  end

  @doc """
  Noise source stream kernel `step(_x, state, opts)`; the input sample is
  ignored. Options: `:format`, `:type`, `:complex` (false), `:amplitude`
  (1.0), `:table_bits` (14, impulse 16), `:impulse_factor` (9).
  """
  defn noise_source(_x, state, opts \\ []) do
    opts =
      keyword!(opts, [
        :format,
        :type,
        complex: false,
        amplitude: 1.0,
        table_bits: nil,
        impulse_factor: 9
      ])

    draw(state, opts)
  end

  @doc """
  `fastnoise_source`: like `noise_source`, but the draw addresses a table of
  stored noise samples (`table`: a vector, or a `{re, im}` pair, in the
  format; see `ExSdr.Sources.block/2`, `:samples`) instead of the
  distribution table. One generator step, one table read: the table is a
  block RAM and the reads cost nothing per distribution.
  Options: `:format`, `:table_bits` (log2 of the table length), and the
  rest as in `noise_source` (ignored).
  """
  defn fastnoise_source(_x, state, table, opts \\ []) do
    opts = keyword!(opts, [:format, :table_bits, :type, :complex, :amplitude, :impulse_factor])
    pick(state, table, opts[:table_bits])
  end

  deftransform pick(state, {re, im}, bits) do
    next = next(state)
    idx = next |> Nx.as_type(:s64) |> Nx.right_shift(32 - bits)
    {{take(re, idx), take(im, idx)}, next}
  end

  deftransform pick(state, table, bits) do
    next = next(state)
    idx = next |> Nx.as_type(:s64) |> Nx.right_shift(32 - bits)
    {take(table, idx), next}
  end

  defp take(%Fixed{t: t, frac: frac}, i), do: %Fixed{t: Nx.take(t, i), frac: frac}
  defp take(t, i), do: Nx.take(t, i)

  deftransform draw(state, opts) do
    if opts[:complex] do
      s1 = next(state)
      s2 = next(s1)
      {value(s1, opts, :re), value(s2, opts, :im)} |> then(&{&1, s2})
    else
      s1 = next(state)
      {value(s1, opts, :re), s1}
    end
  end

  deftransform value(word, opts, _part) do
    w = Nx.as_type(word, :s64)
    f = opts[:format]
    amp = opts[:amplitude]

    case opts[:type] do
      :uniform ->
        uniform(w, amp * component_scale(opts[:complex]), f)

      type ->
        # complex Gaussian: each component carries half the power
        bits = opts[:table_bits] || if(type == :impulse, do: 16, else: 14)
        scale = amp * component_scale(opts[:complex])
        table = table(type, scale, f, bits, opts[:impulse_factor])
        idx = Nx.right_shift(w, 32 - bits)

        case table do
          %Fixed{t: t, frac: frac} -> %Fixed{t: Nx.take(t, idx), frac: frac}
          t -> Nx.take(t, idx)
        end
    end
  end

  defp component_scale(true), do: :math.sqrt(0.5)
  defp component_scale(_complex), do: 1.0

  defp uniform(w, amp, %Format{kind: :float}) do
    u = w |> Nx.right_shift(8) |> Nx.as_type(:f32) |> Nx.add(0.5) |> Nx.multiply(1 / 8_388_608)
    u |> Nx.subtract(1.0) |> Nx.multiply(amp) |> Nx.as_type(:f32)
  end

  defp uniform(w, amp, %Format{kind: :fixed} = f) do
    t = w |> Nx.right_shift(31 - f.frac) |> Nx.subtract(1 <<< f.frac)
    v = quantize(%Fixed{t: t, frac: f.frac}, f)
    if amp == 1.0, do: v, else: v |> mul(constant(amp, f)) |> rescale(f) |> quantize(f)
  end

  @doc """
  The quantile table of `type` (`:gaussian` or `:laplacian`, or `:impulse`),
  scaled by `scale`, in `format`.
  """
  def table(type, scale, format, bits, impulse_factor \\ 9) do
    n = 1 <<< bits
    values = for i <- 0..(n - 1), do: scale * quantile(type, (i + 0.5) / n, impulse_factor)

    case format do
      %Format{kind: :float} ->
        Nx.tensor(values, type: :f32, backend: Nx.BinaryBackend)

      %Format{kind: :fixed} = f ->
        ints = quantize_constants(values, f)
        type = container_type(f.bits)

        %Fixed{t: Nx.tensor(ints, type: type, backend: Nx.BinaryBackend), frac: f.frac}
    end
  end

  defp container_type(bits) when bits <= 8, do: :s8
  defp container_type(bits) when bits <= 16, do: :s16
  defp container_type(bits) when bits <= 32, do: :s32
  defp container_type(_), do: :s64

  @doc "Inverse CDF of the unit distribution `type` at probability `p`."
  def quantile(type, p, impulse_factor \\ 9)
  def quantile(:gaussian, p, _), do: inv_normal(p)
  def quantile(:laplacian, p, _) when p < 0.5, do: :math.log(2 * p)
  def quantile(:laplacian, p, _), do: -:math.log(2 * (1 - p))

  def quantile(:impulse, p, factor) do
    z = -10 / 7 * :math.log(1 - p)
    if z > factor, do: z, else: 0.0
  end

  # Acklam's algorithm with one Halley refinement step (error below 1e-15).
  defp inv_normal(p) do
    a = [
      -3.969683028665376e1,
      2.209460984245205e2,
      -2.759285104469687e2,
      1.383577518672690e2,
      -3.066479806614716e1,
      2.506628277459239e0
    ]

    b = [
      -5.447609879822406e1,
      1.615858368580409e2,
      -1.556989798598866e2,
      6.680131188771972e1,
      -1.328068155288572e1
    ]

    c = [
      -7.784894002430293e-3,
      -3.223964580411365e-1,
      -2.400758277161838e0,
      -2.549732539343734e0,
      4.374664141464968e0,
      2.938163982698783e0
    ]

    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0, 3.754408661907416e0]

    x =
      cond do
        p < 0.02425 ->
          q = :math.sqrt(-2 * :math.log(p))
          horner(c, q) / (horner(d, q) * q + 1)

        p > 1 - 0.02425 ->
          q = :math.sqrt(-2 * :math.log(1 - p))
          -horner(c, q) / (horner(d, q) * q + 1)

        true ->
          q = p - 0.5
          r = q * q
          horner(a, r) * q / (horner(b, r) * r + 1)
      end

    e = 0.5 * :math.erfc(-x / :math.sqrt(2)) - p
    u = e * :math.sqrt(2 * :math.pi()) * :math.exp(x * x / 2)
    x - u / (1 + x * u / 2)
  end

  defp horner(coeffs, x), do: Enum.reduce(coeffs, 0.0, fn k, acc -> acc * x + k end)
end
