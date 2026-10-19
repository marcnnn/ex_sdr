defmodule ExSdr.Timing.Tables do
  @moduledoc """
  MMSE fractional-delay interpolator and interpolating differentiator, 8
  taps, 129 rows: row `k` for the fractional delay `mu = k / 128`.

  Computed at compile time with the minimum mean squared error design of
  Meyr, Moeneclaey and Fechtel, Digital Communication Receivers (Wiley
  1998), chapter 9: for each `mu`, the taps `h` minimise
  `∫ |H(f) - e^(-j 2π f (3 + mu))|² df` over `|f| <= B`, `B = 1/4` of the
  sample rate. The normal equations are `R h = r` with
  `R[n][m] = 2B sinc(2B (n - m))` and `r[n] = 2B sinc(2B (n - 3 - mu))`.
  The differentiator solves the same system for `d r / d mu`, the best
  approximation of the derivative of the delayed signal.

  This is the design GNU Radio's `mmse_fir_interpolator` uses, so its
  printed tables agree with these to about 1e-6.

  Row order follows GNU Radio's convention: tap `j` of a row multiplies
  sample `7 - j` of the 8 sample window (oldest first). `mu = 0` gives
  window sample 3, `mu = 1` sample 4.
  """

  defmodule Design do
    @moduledoc false
    @ntaps 8
    @steps 128
    @band 0.25

    @doc false
    def rows(kind) do
      r = for n <- 0..(@ntaps - 1), do: for(m <- 0..(@ntaps - 1), do: kernel(n - m))

      for k <- 0..@steps do
        delay = 3 + k / @steps

        cond do
          # integer delays: the interpolator is exactly a unit tap
          kind == :interp and k in [0, @steps] ->
            unit = round(delay)
            Enum.reverse(for n <- 0..(@ntaps - 1), do: if(n == unit, do: 1.0, else: 0.0))

          kind == :interp ->
            Enum.reverse(solve(r, for(n <- 0..(@ntaps - 1), do: kernel(n - delay))))

          kind == :diff ->
            Enum.reverse(solve(r, for(n <- 0..(@ntaps - 1), do: -dkernel(n - delay))))
        end
      end
    end

    # 2B sinc(2B x), the band-limited autocorrelation, and its derivative in x
    defp kernel(x), do: 2 * @band * sinc(2 * @band * x)
    defp dkernel(x), do: 4 * @band * @band * dsinc(2 * @band * x)

    defp sinc(x) when x == 0, do: 1.0
    defp sinc(x), do: :math.sin(:math.pi() * x) / (:math.pi() * x)

    defp dsinc(x) when x == 0, do: 0.0
    defp dsinc(x), do: (:math.cos(:math.pi() * x) - sinc(x)) / x

    # Gaussian elimination with partial pivoting
    defp solve(a, b) do
      m = Enum.zip_with(a, b, fn row, bi -> row ++ [bi] end)
      n = length(b)

      m =
        Enum.reduce(0..(n - 1), m, fn col, m ->
          {top, rest} = Enum.split(m, col)
          pivot = Enum.max_by(rest, &abs(Enum.at(&1, col)))
          rest = List.delete(rest, pivot)
          p = Enum.at(pivot, col)
          pivot = Enum.map(pivot, &(&1 / p))

          eliminate = fn row ->
            f = Enum.at(row, col)
            Enum.zip_with(row, pivot, &(&1 - f * &2))
          end

          Enum.map(top, eliminate) ++ [pivot | Enum.map(rest, eliminate)]
        end)

      Enum.map(m, &List.last/1)
    end
  end

  @mmse_rows Design.rows(:interp)
  @mmse_diff_rows Design.rows(:diff)

  @doc "Interpolator taps (`mmse_fir_interpolator_cc/ff`), 129 rows of 8."
  def mmse, do: @mmse_rows

  @doc "Interpolating differentiator taps (`mmse_interp_differentiator_cc/ff`), 129 rows of 8."
  def mmse_diff, do: @mmse_diff_rows
end
