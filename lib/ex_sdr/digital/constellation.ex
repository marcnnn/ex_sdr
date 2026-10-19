defmodule ExSdr.Digital.Constellation do
  @moduledoc """
  GNU Radio style constellation objects (`gr::digital::constellation`) as
  plain data: the points in GNU Radio's order, the pre-differential code
  and the properties blocks ask for.

  The point at index `w` is the point of the symbol word `w` as GNU Radio
  numbers symbols. The hard decision of every GNU Radio constellation is
  the index of the nearest point; GNU Radio's sector based decoders
  (`constellation_rect`, `constellation_psk`) agree with it everywhere
  except in sectors they map differently on purpose, which are not
  modelled here.

  The built-in constellations (`bpsk/0`, `qpsk/0`, `dqpsk/0`, `psk8/0`,
  `qam16/0`) have exactly the points GNU Radio 3.10 gives them: BPSK and
  8-PSK at unit amplitude, 16-QAM at unit average power, QPSK at +-sqrt(2)
  per axis (those two are not normalised the same way by GNU Radio).
  Constellations built with `new/2` are normalised like GNU Radio does by
  default (mean amplitude 1).

  `base` is the `ExSdr.Comm.Constellation` of the same points (unit
  average power), for its tensor tables.
  """
  import Bitwise

  alias ExSdr.Comm.Constellation, as: Comm

  defstruct [
    :points,
    :bits_per_symbol,
    :pre_diff_code,
    :apply_pre_diff_code,
    :rotational_symmetry,
    :base
  ]

  @type t :: %__MODULE__{}

  @doc """
  From `{re, im}` points (a power of two of them). Options:
  `:pre_diff_code` (default `[]`), `:apply_pre_diff_code` (default: whether a
  code was given), `:rotational_symmetry` (default 1), `:normalization`
  (`:amplitude` default, `:power`, `:none`).
  """
  def new(points, opts \\ []) do
    points = normalise(points, Keyword.get(opts, :normalization, :amplitude))
    pdc = Keyword.get(opts, :pre_diff_code, [])

    %__MODULE__{
      points: points,
      bits_per_symbol: round(:math.log2(length(points))),
      pre_diff_code: pdc,
      apply_pre_diff_code: Keyword.get(opts, :apply_pre_diff_code, pdc != []),
      rotational_symmetry: Keyword.get(opts, :rotational_symmetry, 1),
      base: Comm.new(points)
    }
  end

  @doc "GNU Radio `constellation_bpsk`."
  def bpsk, do: new([{-1.0, 0.0}, {1.0, 0.0}], normalization: :none, rotational_symmetry: 2)

  @doc "GNU Radio `constellation_qpsk`."
  def qpsk do
    r = :math.sqrt(2)

    new([{-r, -r}, {r, -r}, {-r, r}, {r, r}],
      normalization: :none,
      pre_diff_code: [0, 2, 3, 1],
      apply_pre_diff_code: false,
      rotational_symmetry: 4
    )
  end

  @doc "GNU Radio `constellation_dqpsk`."
  def dqpsk do
    r = :math.sqrt(2)

    new([{r, r}, {-r, r}, {-r, -r}, {r, -r}],
      normalization: :none,
      pre_diff_code: [0, 1, 3, 2],
      apply_pre_diff_code: true,
      rotational_symmetry: 4
    )
  end

  @doc "GNU Radio `constellation_8psk` (the Gray-like order GNU Radio uses)."
  def psk8 do
    c = :math.cos(:math.pi() / 8)
    s = :math.sin(:math.pi() / 8)

    new(
      [{c, s}, {-c, s}, {c, -s}, {-c, -s}, {s, c}, {-s, c}, {s, -c}, {-s, -c}],
      normalization: :none,
      rotational_symmetry: 8
    )
  end

  @doc "GNU Radio `constellation_8psk_natural`."
  def psk8_natural do
    c = :math.cos(:math.pi() / 8)
    s = :math.sin(:math.pi() / 8)

    new(
      [{c, -s}, {c, s}, {s, c}, {-s, c}, {-c, s}, {-c, -s}, {-s, -c}, {s, -c}],
      normalization: :none,
      rotational_symmetry: 8
    )
  end

  @doc "GNU Radio `constellation_16qam`."
  def qam16 do
    a = 1 / :math.sqrt(10)

    levels = [
      {1, -1},
      {-1, -1},
      {3, -3},
      {-3, -3},
      {-3, -1},
      {3, -1},
      {-1, -3},
      {1, -3},
      {-3, 3},
      {3, 3},
      {-1, 1},
      {1, 1},
      {1, 3},
      {-1, 3},
      {3, 1},
      {-3, 1}
    ]

    new(for({i, q} <- levels, do: {i * a, q * a}), normalization: :none, rotational_symmetry: 4)
  end

  @doc """
  GNU Radio's Python `psk_constellation(m, mod_code, differential)`: points
  `exp(2 pi j i / m)`. `mod_code` is `:gray` or `:none`. Differential Gray
  coding is a pre-differential code, non-differential Gray coding permutes
  the points.
  """
  def psk(m, mod_code \\ :gray, differential \\ true) do
    k = round(:math.log2(m))
    true = 1 <<< k == m

    points =
      for i <- 0..(m - 1),
          do: {:math.cos(2 * :math.pi() * i / m), :math.sin(2 * :math.pi() * i / m)}

    {pdc, points} =
      case {mod_code, differential} do
        {:gray, true} -> {gray_code(m), points}
        {:gray, false} -> {[], permute(points, invert(gray_code(m)))}
        {:none, _} -> {[], points}
      end

    new(points, pre_diff_code: pdc, rotational_symmetry: m)
  end

  @doc """
  GNU Radio's Python `qam_constellation(m, differential, mod_code)` for
  `m` a power of 4. Differential QAM codes the quadrant bits differentially
  (`pre_diff_code` empty, so the differential encoder works on the whole
  word).
  """
  def qam(m, differential \\ true, mod_code \\ :none) do
    k = round(:math.log2(m))
    true = 1 <<< k == m and rem(k, 2) == 0 and m >= 4

    points =
      if differential,
        do: qam_differential_points(m, k),
        else: qam_plain_points(m, k, mod_code == :gray)

    new(points, rotational_symmetry: 4)
  end

  # Square QAM on [-1, 1] x [-1, 1]: the low k/2 bits of a symbol give the
  # imaginary level, the high k/2 bits the real level. With Gray coding the
  # bits are the Gray code of the level's position, so neighbours differ in
  # one bit.
  defp qam_plain_points(m, k, gray?) do
    levels = round(:math.sqrt(m))
    spacing = 2.0 / (levels - 1)
    position = if gray?, do: &gray_decode/1, else: & &1
    level = fn bits -> -1 + position.(bits) * spacing end
    half = div(k, 2)

    for s <- 0..(m - 1) do
      {level.(field(s, half, half)), level.(field(s, 0, half))}
    end
  end

  # Differential QAM: the top two bits pick the quadrant (a multiple of 90
  # degrees, so a phase ambiguity of the receiver only changes them), the
  # others a point of the first quadrant grid.
  defp qam_differential_points(m, k) do
    per_side = div(round(:math.sqrt(m)), 2)
    spacing = 1 / (per_side - 0.5)
    half = div(k - 2, 2)

    for s <- 0..(m - 1) do
      point = {(field(s, half, half) + 0.5) * spacing, (field(s, 0, half) + 0.5) * spacing}
      rotate90(point, field(s, k - 2, 2))
    end
  end

  defp rotate90(point, 0), do: point
  defp rotate90({x, y}, n), do: rotate90({-y, x}, n - 1)

  # `width` bits of `x` starting at bit `from`
  defp field(x, from, width), do: (x >>> from) &&& (1 <<< width) - 1

  # the position whose Gray code is `g`
  defp gray_decode(0), do: 0
  defp gray_decode(g), do: bxor(g, gray_decode(g >>> 1))

  @doc "Binary reflected Gray code of `0..n-1` (GNU Radio's `gray_code`)."
  def gray_code(n), do: for(i <- 0..(n - 1), do: bxor(i, i >>> 1))

  defp invert(code), do: code |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1))
  defp permute(list, order), do: Enum.map(order, &Enum.at(list, &1))

  defp normalise(points, :none), do: points

  defp normalise(points, kind) do
    n = length(points)

    scale =
      case kind do
        :amplitude -> n / Enum.sum(for {a, b} <- points, do: :math.sqrt(a * a + b * b))
        :power -> 1 / :math.sqrt(Enum.sum(for {a, b} <- points, do: a * a + b * b) / n)
      end

    for {a, b} <- points, do: {a * scale, b * scale}
  end

  @doc "The index of the point nearest to `{re, im}` (ties: the smaller index)."
  def decide(%__MODULE__{points: points}, {re, im}) do
    points
    |> Enum.with_index()
    |> Enum.min_by(fn {{a, b}, _} -> (a - re) * (a - re) + (b - im) * (b - im) end)
    |> elem(1)
  end

  @doc """
  The bit label of each point (by index): the point index, or its entry of
  `pre_diff_code` when the constellation applies it (`dqpsk`).
  """
  def labels(%__MODULE__{apply_pre_diff_code: true, pre_diff_code: pdc}), do: pdc
  def labels(%__MODULE__{points: points}), do: Enum.to_list(0..(length(points) - 1))

  @doc """
  Soft decision as `constellation_soft_decoder_cf` gives it: for each of
  the `bits_per_symbol` bits of the point's label (`labels/1`, most
  significant first) the log-likelihood ratio `ln(sum over points with bit 1 of exp(-d^2 / npwr)
  / sum over points with bit 0 of exp(-d^2 / npwr))`, `d` the distance to
  the point. Positive means bit 1. With `npwr = 1` (GNU Radio's default
  without a lookup table) this is what GNU Radio 3.10 outputs.
  """
  def soft_decision(%__MODULE__{points: points, bits_per_symbol: k} = c, {re, im}, npwr \\ 1.0) do
    metric = for {a, b} <- points, do: -((a - re) * (a - re) + (b - im) * (b - im)) / npwr
    indexed = Enum.zip(metric, labels(c))

    for j <- 0..(k - 1) do
      {ones, zeros} = Enum.split_with(indexed, fn {_, i} -> (i >>> (k - 1 - j) &&& 1) == 1 end)
      logsumexp(Enum.map(ones, &elem(&1, 0))) - logsumexp(Enum.map(zeros, &elem(&1, 0)))
    end
  end

  defp logsumexp(xs) do
    m = Enum.max(xs)
    m + :math.log(Enum.sum(for x <- xs, do: :math.exp(x - m)))
  end
end
