defmodule ExSdr.Timing.Interp do
  @moduledoc """
  Fractional-delay interpolators as tap tables, and the kernel that applies
  one row of a table to a window of samples.

  An interpolator is `%Interp{rows: rows, drows: drows, ntaps: n, arms: a}`:
  `rows` has `a + 1` (or `a`, for the polyphase resamplers) rows of `n` taps; row `k` is
  used for the fractional delay `mu = k / a`. Tap `j` multiplies window
  sample `n - 1 - j` (the window is oldest first), GNU Radio's convention
  (its FIR kernel reverses the taps). `drows` are the matching derivative
  filters (or `nil`).

    * `mmse/0` - GNU Radio's 8 tap MMSE interpolator (`mmse_fir_interpolator_cc/ff`),
      129 rows, with the interpolating differentiator.
    * `pfb_no_mf/1` - symbol_sync's `IR_PFB_NO_MF`: every `128 / a`-th row of
      the MMSE table, `a` the power of two at or above `nfilts`.
    * `pfb_mf/2` - symbol_sync's `IR_PFB_MF`: the polyphase arms of a
      prototype filter (typically a matched filter at `nfilts` times the
      sample rate), `nfilts + 1` arms, with GNU Radio's derivative filter.
    * `polyphase/2` - `nfilts` arms `taps[i + j * nfilts]` (pfb_arb_resampler,
      pfb_clock_sync), with an optional derivative.

  ## Fixed point

  Taps are rounded to `tap_frac` fractional bits (not saturated: a tap
  above 1 gets integer bits in the multiplier). The sum over the window
  is exact; one rounding (and saturation) to the format ends each output,
  as in `ExSdr.Filters.Kernels`.
  """
  alias ExSdr.Timing.Loop, as: L
  alias ExSdr.Timing.Tables
  alias NxFixed.Fixed
  alias NxFixed.Num
  alias NxFixed.Num.Format

  defstruct [:rows, :drows, :ntaps, :arms, :name]

  @type t :: %__MODULE__{}

  @doc "GNU Radio's 8 tap MMSE interpolator and differentiator (129 rows)."
  def mmse do
    %__MODULE__{rows: Tables.mmse(), drows: Tables.mmse_diff(), ntaps: 8, arms: 128, name: :mmse}
  end

  @doc """
  `IR_PFB_NO_MF` of symbol_sync: `a + 1` rows of the MMSE table, `a` the
  smallest power of two above `nfilts - 1` (at most 128).
  """
  def pfb_no_mf(nfilts) when nfilts > 1 do
    a = Bitwise.bsl(1, trunc(:math.log2(L.f32(nfilts - 1))) + 1)
    a = if a <= 0 or a > 128, do: 128, else: a
    step = div(128, a)
    pick = fn table -> for src <- 0..128//step, do: Enum.at(table, src) end

    %__MODULE__{
      rows: pick.(Tables.mmse()),
      drows: pick.(Tables.mmse_diff()),
      ntaps: 8,
      arms: a,
      name: :pfb_no_mf
    }
  end

  @doc """
  `IR_PFB_MF` of symbol_sync: the prototype `taps` split into `nfilts + 1`
  arms (arm `i` has taps `i + j * nfilts`), and GNU Radio's derivative
  filter: the taps convolved with `[1, 0, -1]`, end taps zeroed, scaled to
  an absolute sum of `nfilts`.
  """
  def pfb_mf(taps, nfilts) when nfilts > 1 and length(taps) >= nfilts do
    n = length(taps)
    t = List.to_tuple(taps)
    at = fn i -> if i >= 0 and i < n, do: L.f32(elem(t, i)), else: 0.0 end
    # convolution with [1, 0, -1], centre aligned: d[i] = taps[i + 1] - taps[i - 1]
    d = for i <- 0..(n - 1), do: L.f32(at.(i + 1) - at.(i - 1))
    d = d |> List.replace_at(0, 0.0) |> List.replace_at(n - 1, 0.0)
    mag = d |> Enum.map(&abs/1) |> Enum.sum() |> L.f32()
    scale = L.f32(nfilts / mag)
    d = Enum.map(d, &L.f32(&1 * scale))
    k = ceil(n / nfilts)

    %__MODULE__{
      rows: arms(taps, nfilts, k, nfilts + 1),
      drows: arms(d, nfilts, k, nfilts + 1),
      ntaps: k,
      arms: nfilts,
      name: :pfb_mf
    }
  end

  @doc """
  `nfilts` polyphase arms of `taps` (arm `i` has taps `i + j * nfilts`, zero
  padded), as in pfb_arb_resampler and pfb_clock_sync. `diff`: derivative
  taps of the same length (or `nil`).
  """
  def polyphase(taps, nfilts, diff \\ nil) do
    k = ceil(length(taps) / nfilts)

    %__MODULE__{
      rows: arms(taps, nfilts, k, nfilts),
      drows: diff && arms(diff, nfilts, k, nfilts),
      ntaps: k,
      arms: nfilts,
      name: :polyphase
    }
  end

  defp arms(taps, nfilts, k, count) do
    t = List.to_tuple(taps)
    n = tuple_size(t)

    for i <- 0..(count - 1),
        do:
          for(j <- 0..(k - 1), do: if(i + j * nfilts < n, do: elem(t, i + j * nfilts), else: 0.0))
  end

  @doc "Interpolates with row `arm` (an s32 scalar tensor or an integer) of `rows`."
  def apply_rows(window, arm, rows, f, opts \\ []) do
    tf = Keyword.get(opts, :tap_frac) || f.frac || 0
    coef = coefficients(rows, arm, f, tf)
    acc = L.cmap(window, &dot(&1, coef, f, tf))
    if Keyword.get(opts, :raw, false), do: acc, else: Num.quantize(acc, f)
  end

  # The row as a vector in window order (tap j multiplies sample n - 1 - j):
  # a constant for a fixed row, else one table lookup per tap.
  defp coefficients(rows, arm, %Format{} = f, tf) when is_integer(arm) do
    rows |> Enum.at(arm) |> Enum.reverse() |> const_tensor(f, tf)
  end

  defp coefficients(rows, arm, %Format{} = f, tf) do
    n = length(hd(rows))

    for j <- (n - 1)..0//-1 do
      rows |> Enum.map(&Enum.at(&1, j)) |> const_tensor(f, tf) |> Nx.take(arm)
    end
    |> Nx.stack()
  end

  defp const_tensor(values, %Format{kind: :float}, _tf),
    do: Nx.tensor(values, type: :f32, backend: Nx.BinaryBackend)

  defp const_tensor(values, %Format{kind: :fixed}, tf) do
    ints = Enum.map(values, &round(&1 * Bitwise.bsl(1, tf)))
    m = ints |> Enum.map(&abs/1) |> Enum.max()
    Nx.tensor(ints, type: if(m < 32768, do: :s16, else: :s32), backend: Nx.BinaryBackend)
  end

  defp dot(%Fixed{t: x, frac: frac}, coef, _f, tf) do
    %Fixed{t: Nx.dot(Nx.as_type(x, :s64), Nx.as_type(coef, :s64)), frac: frac + tf}
  end

  defp dot(x, coef, _f, _tf), do: Nx.dot(x, coef)

  @doc "Interpolated sample of `window` (`ntaps` items) at the fractional delay `mu` (loop domain)."
  def interpolate(%__MODULE__{} = ip, window, mu, f, opts \\ []),
    do: apply_rows(window, L.arm(mu, ip.arms, f), ip.rows, f, opts)

  @doc "Interpolated derivative (the `drows` filters)."
  def differentiate(%__MODULE__{} = ip, window, mu, f, opts \\ []),
    do: apply_rows(window, L.arm(mu, ip.arms, f), ip.drows, f, opts)
end
