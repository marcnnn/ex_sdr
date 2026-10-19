defmodule ExSdr.Filters.Window do
  @moduledoc """
  Window functions as in GNU Radio's `fft.window` (3.10): symmetric,
  unnormalised (peak 1 for odd lengths), computed in double precision.

  A window type is `:hamming`, `:hann`, `:blackman`, `:blackman_harris`,
  `:rectangular` or `:kaiser` (which uses `beta`).

  GNU Radio returns NaN for a one-point cosine window (division by zero);
  here every window of length 1 is `[1.0]`.
  """

  @type type :: :hamming | :hann | :blackman | :blackman_harris | :rectangular | :kaiser

  @doc "The window of `n` points. `beta` is used by `:kaiser` only."
  @spec build(type, pos_integer, number) :: [float]
  def build(type, n, beta \\ 6.76)

  def build(_type, 1, _beta), do: [1.0]
  def build(:hamming, n, _), do: cosine(n, [0.54, 0.46])
  def build(:hann, n, _), do: cosine(n, [0.5, 0.5])
  def build(:blackman, n, _), do: cosine(n, [0.42, 0.5, 0.08])

  def build(:blackman_harris, n, _),
    do: cosine(n, [0.35875, 0.48829, 0.14128, 0.01168])

  def build(:rectangular, n, _), do: List.duplicate(1.0, n)

  def build(:kaiser, n, beta) when beta >= 0 do
    m = n - 1
    i0 = bessel_i0(beta)

    for i <- 0..m do
      r = 2 * i / m - 1
      bessel_i0(beta * :math.sqrt(max(1 - r * r, 0.0))) / i0
    end
  end

  @doc """
  Stop-band attenuation in dB that GNU Radio assumes for a window type
  (`fft.window.max_attenuation`), used to size filters.
  """
  def max_attenuation(:hamming, _), do: 53.0
  def max_attenuation(:hann, _), do: 44.0
  def max_attenuation(:blackman, _), do: 74.0
  def max_attenuation(:blackman_harris, _), do: 92.0
  def max_attenuation(:rectangular, _), do: 21.0
  def max_attenuation(:kaiser, beta), do: beta / 0.1102 + 8.7

  # c0 - c1 cos(2 pi i / m) + c2 cos(4 pi i / m) - c3 cos(6 pi i / m)
  defp cosine(n, coefficients) do
    m = n - 1

    for i <- 0..m do
      coefficients
      |> Enum.with_index()
      |> Enum.reduce(0.0, fn {c, k}, acc ->
        sign = if rem(k, 2) == 0, do: 1, else: -1
        acc + sign * c * :math.cos(2 * :math.pi() * k * i / m)
      end)
    end
  end

  @doc "Modified Bessel function of the first kind, order 0 (power series)."
  def bessel_i0(x) do
    q = x * x / 4
    series(1.0, 1.0, q, 1)
  end

  defp series(sum, term, q, k) do
    term = term * q / (k * k)
    if term < 1.0e-17 * sum, do: sum + term, else: series(sum + term, term, q, k + 1)
  end
end
