Code.require_file("test_kit.exs", __DIR__)

defmodule ExSdr.Timing.ResamplerTest do
  @moduledoc """
  mmse_resampler and pfb_arb_resampler per case in test/dsp_cases/timing:
  the float instance against GNU Radio, the fixed-point variants against
  the float instance (both with the bounds and reasoning of the case
  file), and split invariance.
  """
  use ExUnit.Case, async: true

  alias ExSdr.Timing.TestKit, as: T
  alias NxFixed.Num

  @moduletag timeout: 600_000

  setup_all do
    runs =
      Map.new(T.family(), fn {case, data} ->
        {float, _} = T.run(T.block(case, Num.float()), data.input)
        {case["name"], %{case: case, data: data, float: float}}
      end)

    %{runs: runs}
  end

  for {case, _data} <- T.family() do
    @name case["name"]

    test "#{@name}: float instance vs GNU Radio", %{runs: runs} do
      %{case: case, data: data, float: float} = runs[@name]
      bound = case["float_vs_gnuradio"]["max_error"]

      # GNU Radio stops a few outputs early (it needs a full window ahead)
      assert length(float) >= length(data.reference)
      assert length(float) - length(data.reference) <= 32

      {ref, got} = T.prefix(data.reference, float)
      worst = T.errors(ref, got) |> Enum.max()
      assert worst <= bound, "max error #{worst} > #{bound}"
    end

    for variant <- case["fixed_vs_float"] do
      @variant variant

      test "#{@name}: #{variant["variant"]} vs float", %{runs: runs} do
        %{data: data, float: float, case: case} = runs[@name]
        f = T.format(@variant["format"])
        {fixed, _} = T.run(T.block(case, f), data.input)
        assert length(fixed) == length(float)
        lsb = Enum.max(T.errors(fixed, float)) * 32768
        assert lsb <= @variant["max_error_lsb"], "#{lsb} LSB"
      end
    end

    test "#{@name}: input in pieces gives the same output (float and Q15)", %{runs: runs} do
      %{case: case, data: data, float: float} = runs[@name]
      assert T.run_chunked(T.block(case, Num.float()), data.input, [1, 7, 300, 2, 513]) == float

      q = T.block(case, Num.q(16, 15))
      {whole, _} = T.run(q, data.input)
      assert T.run_chunked(q, data.input, [5, 1, 640, 3]) == whole
    end
  end
end
