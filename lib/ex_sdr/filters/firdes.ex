defmodule ExSdr.Filters.Firdes do
  @moduledoc """
  FIR filter design by the window method, matching GNU Radio's `filter.firdes`
  (3.10) tap for tap. Taps are Elixir floats (double precision; GNU Radio
  rounds them to `f32`).

  Frequencies are in the unit of `fs` (Hz or cycles per sample with
  `fs = 1.0`). `window` is a type of `ExSdr.Filters.Window`; `beta` is the
  Kaiser parameter. The number of taps follows GNU Radio: the window's
  assumed attenuation over `22 * transition_width / fs`, made odd.

  Complex taps are `{re, im}` tuples.
  """
  alias ExSdr.Filters.Window

  @pi :math.pi()

  @doc "Number of taps GNU Radio uses for a transition band (always odd)."
  def num_taps(fs, transition_width, window \\ :hamming, beta \\ 6.76) do
    a = Window.max_attenuation(window, beta)
    n = trunc(a / (22.0 * (transition_width / fs)))
    if rem(n, 2) == 0, do: n + 1, else: n
  end

  @doc "Low-pass with DC gain `gain`."
  def low_pass(gain, fs, cutoff, transition_width, window \\ :hamming, beta \\ 6.76) do
    {n, m, w} = setup(fs, transition_width, window, beta)
    w0 = 2 * @pi * cutoff / fs

    taps =
      for {wi, i} <- Enum.with_index(w) do
        k = i - m
        if k == 0, do: w0 / @pi * wi, else: :math.sin(k * w0) / (k * @pi) * wi
      end

    scale(taps, gain / center_sum(taps, m, fn _k -> 1.0 end), n)
  end

  @doc "High-pass with Nyquist gain `gain`."
  def high_pass(gain, fs, cutoff, transition_width, window \\ :hamming, beta \\ 6.76) do
    {n, m, w} = setup(fs, transition_width, window, beta)
    w0 = 2 * @pi * cutoff / fs

    taps =
      for {wi, i} <- Enum.with_index(w) do
        k = i - m
        if k == 0, do: (1 - w0 / @pi) * wi, else: -:math.sin(k * w0) / (k * @pi) * wi
      end

    scale(taps, gain / center_sum(taps, m, fn k -> :math.cos(k * @pi) end), n)
  end

  @doc "Band-pass with gain `gain` at the band centre."
  def band_pass(gain, fs, low, high, transition_width, window \\ :hamming, beta \\ 6.76) do
    {n, m, w} = setup(fs, transition_width, window, beta)
    w0 = 2 * @pi * low / fs
    w1 = 2 * @pi * high / fs

    taps =
      for {wi, i} <- Enum.with_index(w) do
        k = i - m

        if k == 0,
          do: (w1 - w0) / @pi * wi,
          else: (:math.sin(k * w1) - :math.sin(k * w0)) / (k * @pi) * wi
      end

    scale(taps, gain / center_sum(taps, m, fn k -> :math.cos(k * (w0 + w1) * 0.5) end), n)
  end

  @doc "Band-reject with DC gain `gain`."
  def band_reject(gain, fs, low, high, transition_width, window \\ :hamming, beta \\ 6.76) do
    {n, m, w} = setup(fs, transition_width, window, beta)
    w0 = 2 * @pi * low / fs
    w1 = 2 * @pi * high / fs

    taps =
      for {wi, i} <- Enum.with_index(w) do
        k = i - m

        if k == 0,
          do: (1 + (w0 - w1) / @pi) * wi,
          else: (:math.sin(k * w0) - :math.sin(k * w1)) / (k * @pi) * wi
      end

    scale(taps, gain / center_sum(taps, m, fn _k -> 1.0 end), n)
  end

  @doc """
  Complex band-pass: a low-pass of half the bandwidth shifted to the band
  centre. `low` and `high` may be negative. Returns `{re, im}` taps.
  """
  def complex_band_pass(gain, fs, low, high, transition_width, window \\ :hamming, beta \\ 6.76) do
    lp = low_pass(gain, fs, (high - low) / 2, transition_width, window, beta)
    freq = @pi * (high + low) / fs
    m = div(length(lp), 2)

    for {t, i} <- Enum.with_index(lp) do
      phase = freq * (i - m)
      {t * :math.cos(phase), t * :math.sin(phase)}
    end
  end

  @doc """
  Root raised cosine (square root Nyquist) pulse of `ntaps` (made odd),
  `symbol_rate` symbols per second at sample rate `fs`, roll-off `alpha`.
  The taps sum to `gain`.
  """
  def root_raised_cosine(gain, fs, symbol_rate, alpha, ntaps) do
    ntaps = Bitwise.bor(ntaps, 1)
    spb = fs / symbol_rate
    half = div(ntaps, 2)

    raw = for i <- 0..(ntaps - 1), do: rrc_tap(i - half, spb, alpha)
    s = Enum.sum(raw)
    Enum.map(raw, &(&1 * gain / s))
  end

  # Unnormalised RRC at `k` samples from the centre. Singular points
  # (k = 0, and k = +-spb / (4 alpha)) use the closed-form limits.
  defp rrc_tap(0, _spb, alpha), do: 1.0 - alpha + 4 * alpha / @pi

  defp rrc_tap(k, spb, alpha) do
    t = k / spb
    d = 1 - :math.pow(4 * alpha * t, 2)

    if abs(d) < 1.0e-9 do
      s = :math.sqrt(2.0)

      alpha / s *
        ((1 + 2 / @pi) * :math.sin(@pi / (4 * alpha)) +
           (1 - 2 / @pi) * :math.cos(@pi / (4 * alpha)))
    else
      (:math.sin(@pi * t * (1 - alpha)) + 4 * alpha * t * :math.cos(@pi * t * (1 + alpha))) /
        (@pi * t * d)
    end
  end

  @doc """
  Gaussian pulse for GMSK-like shaping: `spb` samples per symbol, bandwidth
  time product `bt`, `ntaps` taps summing to `gain`.
  """
  def gaussian(gain, spb, bt, ntaps) do
    # The Gaussian filter with 3 dB bandwidth B has an impulse response with
    # standard deviation sigma = sqrt(ln 2) / (2 pi B), here in symbols.
    sigma = :math.sqrt(:math.log(2.0)) / (2 * @pi * bt)
    centre = ntaps / 2

    pulse =
      for n <- 1..ntaps do
        t = (n - centre) / spb / sigma
        :math.exp(-t * t / 2)
      end

    area = Enum.sum(pulse)
    Enum.map(pulse, &(gain * &1 / area))
  end

  @doc """
  Hilbert transformer of `ntaps` (odd) taps: the ideal response `2 / (pi k)`
  at odd `k` (zero at even `k`), windowed, scaled to unit gain at a quarter
  of the sample rate (as GNU Radio does). GNU Radio's default window is
  rectangular.
  """
  def hilbert(ntaps \\ 19, window \\ :rectangular, beta \\ 6.76) do
    if rem(ntaps, 2) == 0, do: raise(ArgumentError, "hilbert needs an odd number of taps")
    m = div(ntaps - 1, 2)
    w = Window.build(window, ntaps, beta)

    taps =
      Enum.zip_with(-m..m, w, fn k, wk ->
        if rem(k, 2) == 0, do: 0.0, else: wk / k
      end)

    # |H| at f = fs / 4: an antisymmetric filter gives |sum h[k] sin(pi k / 2)|
    gain = abs(Enum.sum(Enum.zip_with(-m..m, taps, fn k, t -> t * quarter_sin(k) end)))
    Enum.map(taps, &(&1 / gain))
  end

  # sin(pi k / 2) for an integer k, exactly
  defp quarter_sin(k) do
    case Integer.mod(k, 4) do
      1 -> 1
      3 -> -1
      _ -> 0
    end
  end

  # -- shared ----------------------------------------------------------------------

  defp setup(fs, transition_width, window, beta) do
    n = num_taps(fs, transition_width, window, beta)
    {n, div(n - 1, 2), Window.build(window, n, beta)}
  end

  # taps[m] + 2 * sum_{k>=1} taps[m + k] * weight(k)
  defp center_sum(taps, m, weight) do
    t = List.to_tuple(taps)
    elem(t, m) + 2 * Enum.sum(for k <- 1..m//1, do: elem(t, m + k) * weight.(k))
  end

  defp scale(taps, g, _n), do: Enum.map(taps, &(&1 * g))
end
