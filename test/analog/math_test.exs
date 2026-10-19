defmodule ExSdr.Analog.MathTest do
  use ExUnit.Case, async: true

  import Bitwise
  alias ExSdr.Analog.Math, as: M

  defp t(list, type \\ :s64), do: Nx.tensor(list, type: type)
  # floor square root by Newton's method
  defp isqrt(0), do: 0
  defp isqrt(n), do: isqrt(n, n)

  defp isqrt(n, x) do
    y = div(x + div(n, x), 2)
    if y >= x, do: x, else: isqrt(n, y)
  end

  defp ints(tensor), do: Nx.to_flat_list(tensor)

  describe "cordic_atan2" do
    test "agrees with atan2 within the residual angle on a grid of vectors" do
      # (x, y) over several magnitudes and all octants
      vectors =
        for mag <- [3, 100, 5_000, 1_000_000],
            k <- 0..63,
            angle = (k + 0.5) * 2 * :math.pi() / 64 - :math.pi(),
            do: {round(mag * :math.cos(angle)), round(mag * :math.sin(angle))}

      {xs, ys} = Enum.unzip(vectors)
      got = M.cordic_atan2(t(ys), t(xs), 16, 8) |> ints()

      # residual atan(2^-15) = 3.05e-5 rad = 14_000 words; small vectors are
      # also limited by the integer rounding of their own coordinates
      for {{x, y}, w} <- Enum.zip(vectors, got) do
        want = :math.atan2(y, x) * 2_147_483_648 / :math.pi()
        # a vector of magnitude m is known to about 1/m rad
        slack = 14_700 + 2_147_483_648 / :math.pi() / max(abs(x) + abs(y), 1)

        assert abs(w - want) <= slack or abs(abs(w - want) - 4_294_967_296) <= slack,
               "atan2(#{y}, #{x}) = #{w} words, want #{want}"
      end
    end

    test "zero vector, the negative axis and the axes" do
      got = M.cordic_atan2(t([0, 0, 0, 5000, -5000]), t([0, -7000, 7000, 0, 0]), 16, 8) |> ints()
      [zero, neg_x, pos_x, pos_y, neg_y] = got
      assert zero == 0
      # atan2(0, -x) is +pi, like GNU Radio
      assert abs(neg_x - 2_147_483_648) < 20_000
      assert abs(pos_x) < 20_000
      assert abs(pos_y - 1_073_741_824) < 20_000
      assert abs(neg_y + 1_073_741_824) < 20_000
    end

    test "more iterations shrink the residual" do
      xs = for k <- 1..200, do: 1_000_000 + 37 * k
      ys = for k <- 1..200, do: 123_457 * k - 9_000_000

      worst = fn iters ->
        M.cordic_atan2(t(ys), t(xs), iters, 8)
        |> ints()
        |> Enum.zip(Enum.zip(xs, ys))
        |> Enum.map(fn {w, {x, y}} -> abs(w - :math.atan2(y, x) * 2_147_483_648 / :math.pi()) end)
        |> Enum.max()
      end

      assert worst.(24) < worst.(16) / 50
    end
  end

  test "isqrt is the exact floor square root" do
    values = [0, 1, 2, 3, 4, 15, 16, 17, 99_999, 1 <<< 30, (1 <<< 40) + 12345, (1 <<< 58) - 1]
    got = M.isqrt(t(values), 30) |> ints()
    assert got == Enum.map(values, &isqrt/1)
  end

  test "wrap32 is two's complement on 32 bits" do
    values = [
      0,
      1,
      -1,
      2_147_483_647,
      2_147_483_648,
      -2_147_483_648,
      -2_147_483_649,
      4_294_967_296 + 5
    ]

    assert M.wrap32(t(values)) |> ints() == [
             0,
             1,
             -1,
             2_147_483_647,
             -2_147_483_648,
             -2_147_483_648,
             2_147_483_647,
             5
           ]
  end

  test "mul_real rounds w * k, for small and large constants" do
    for k <- [1.0e-6, 3.3e-3, 0.5, 1.0, 7.96, 1234.5, -0.25, -3.0e-4] do
      w = [0, 1, -1, 1000, -123_456, 987_654_321, 2_000_000_000]
      got = M.mul_real(t(w), k) |> ints()

      for {wi, g} <- Enum.zip(w, got) do
        # 30 significant bits in the constant
        assert abs(g - wi * k) <= 0.5 + abs(wi * k) * 2.0e-9, "#{wi} * #{k} = #{g}"
      end
    end
  end

  test "sincos_word: float is exact, fixed point follows the table" do
    words = for k <- 0..63, do: k * 67_108_864 - 2_147_483_648
    f = NxFixed.Num.q(16, 15, lut_bits: 12, rounding: :nearest)
    {c, s} = M.sincos_word(t(words), NxFixed.Num.float())

    for {w, cc, ss} <- Enum.zip([words, Nx.to_flat_list(c), Nx.to_flat_list(s)]) do
      a = w * :math.pi() / 2_147_483_648
      assert_in_delta cc, :math.cos(a), 1.0e-6
      assert_in_delta ss, :math.sin(a), 1.0e-6
    end

    {c, s} = M.sincos_word(t(words), f)
    {c, s} = {NxFixed.Num.to_float(c), NxFixed.Num.to_float(s)}

    for {w, cc, ss} <- Enum.zip([words, c, s]) do
      a = w * :math.pi() / 2_147_483_648
      # phase rounded to 2^-12 turn, amplitude 32767/32768
      assert_in_delta cc, :math.cos(a), :math.pi() / 4096 + 1.0e-4
      assert_in_delta ss, :math.sin(a), :math.pi() / 4096 + 1.0e-4
    end
  end
end
