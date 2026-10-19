Code.require_file("test_kit.exs", __DIR__)

defmodule ExSdr.Timing.SyncTest do
  @moduledoc """
  Symbol timing recovery (clock_recovery_mm, symbol_sync with every timing
  error detector and interpolator, pfb_clock_sync, msk_timing_recovery)
  per case in test/dsp_cases/timing/sync, on shaped PSK and MSK signals
  with a timing and sample rate offset made by
  test/gnuradio/timing_golden.py:

    1. float instance vs GNU Radio: median error, rare larger errors
       (fraction and maximum), MER after lock,
    2. fixed-point variants vs float after lock: maximum and rms error,
       MER, output count,
    3. split invariance (bit-exact).

  Bounds and their reasoning are in the case files.
  """
  use ExUnit.Case, async: true

  alias ExSdr.Timing.TestKit, as: T
  alias NxFixed.Num

  @moduletag timeout: 900_000

  setup_all do
    runs =
      Map.new(T.sync_family(), fn {case, data} ->
        {float, _} = T.run(T.sync_block(case, data, Num.float()), data.input)
        {case["name"], %{case: case, data: data, float: float}}
      end)

    %{runs: runs}
  end

  # Modulation error ratio of the outputs after `skip`, against a per-axis
  # amplitude estimate (one point per axis and sign).
  defp mer(xs, skip) do
    comps =
      xs
      |> Enum.drop(skip)
      |> Enum.flat_map(fn
        {a, b} -> [a, b]
        a -> [a]
      end)

    amp = (comps |> Enum.map(&abs/1) |> Enum.sum()) / length(comps)
    err = comps |> Enum.map(&((abs(&1) - amp) ** 2)) |> Enum.sum()
    10 * :math.log10(amp * amp * length(comps) / err)
  end

  defp median(list), do: list |> Enum.sort() |> Enum.at(div(length(list), 2))

  for {case, _} <- T.sync_family() do
    @name case["name"]

    test "#{@name}: float instance vs GNU Radio", %{runs: runs} do
      %{case: case, data: data, float: float} = runs[@name]
      b = case["float_vs_gnuradio"]

      # GNU Radio stops early: it waits for input beyond its windows
      assert length(float) >= length(data.reference)
      assert length(float) - length(data.reference) <= 64

      {ref, got} = T.prefix(data.reference, float)
      e = T.errors(ref, got)
      assert median(e) <= b["median_error"]
      assert Enum.max(e) <= b["max_error"], "max #{Enum.max(e)}"
      fraction = Enum.count(e, &(&1 > 1.0e-4)) / length(e)
      assert fraction <= b["max_fraction_over_1e-4"], "fraction #{fraction}"

      d = abs(mer(ref, b["lock_after"]) - mer(got, b["lock_after"]))
      assert d <= b["mer_db_tolerance"], "MER differs by #{d} dB"
    end

    for variant <- case["fixed_vs_float"] do
      @variant variant

      test "#{@name}: #{variant["variant"]} vs float", %{runs: runs} do
        %{case: case, data: data, float: float} = runs[@name]
        skip = case["float_vs_gnuradio"]["lock_after"]
        {fixed, _} = T.run(T.sync_block(case, data, T.format(@variant["format"])), data.input)
        assert length(fixed) == length(float)

        e = T.errors(float, fixed) |> Enum.drop(skip)
        rms = :math.sqrt(Enum.sum(Enum.map(e, &(&1 * &1))) / length(e))
        assert Enum.max(e) <= @variant["max_error"], "max #{Enum.max(e)}"
        assert rms <= @variant["rms_error"], "rms #{rms}"
        d = abs(mer(float, skip) - mer(fixed, skip))
        assert d <= @variant["mer_db_tolerance"], "MER differs by #{d} dB"
      end
    end

    test "#{@name}: input in pieces gives the same output (Q15)", %{runs: runs} do
      %{case: case, data: data} = runs[@name]
      q = T.sync_block(case, data, Num.q(16, 15))
      {whole, _} = T.run(q, data.input)
      n = length(data.input)
      assert T.run_chunked(q, data.input, [div(n, 3), div(n, 3)]) == whole
    end
  end

  test "float instance: input in small pieces", %{runs: runs} do
    for name <- ~w(symbol_sync_cc_early_late pfb_clock_sync_ccf_osps2 msk_timing_recovery_cc) do
      %{case: case, data: data, float: float} = runs[name]
      input = Enum.take(data.input, 600)
      b = T.sync_block(case, data, Num.float())
      {whole, _} = T.run(b, input)
      assert T.run_chunked(b, input, [1, 2, 3, 7, 1, 100]) == whole
      assert Enum.take(float, length(whole)) == whole
    end
  end
end
