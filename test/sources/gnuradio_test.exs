defmodule ExSdr.Sources.GnuRadioTest do
  @moduledoc """
  The deterministic sources and channel parts against GNU Radio
  (test/dsp_cases/sources, output in test/fixtures/gnuradio/sources), then
  every fixed-point variant against the float instance. Noise and fading
  (random) are checked by their statistics in noise_test.exs and
  channel_test.exs. The fixed-point runs go through the width check of
  `NxFixed` (a violation raises).
  """
  use ExUnit.Case, async: true

  Code.require_file("support.exs", __DIR__)
  alias ExSdr.SourcesCase, as: C

  @moduletag timeout: 300_000

  @deterministic for {case, _} <- C.cases(), Map.has_key?(case, "float_vs_gnuradio"), do: case

  setup_all do
    runs =
      for {case, data} <- C.cases(), Map.has_key?(case, "float_vs_gnuradio"), into: %{} do
        {case["name"], %{case: case, data: data, float: C.run(case, data, NxFixed.Num.float())}}
      end

    %{runs: runs}
  end

  for case <- @deterministic do
    @name case["name"]
    @bound case["float_vs_gnuradio"]

    test "#{case["name"]}: float instance vs GNU Radio (#{case["description"]})", %{runs: runs} do
      %{data: data, float: float} = runs[@name]
      # GNU Radio's scheduler leaves up to a few more input samples unused at
      # the end of a finite stream than the exact "every output whose 8 tap
      # window is available" rule here; compare the common prefix
      extra = length(float) - length(data.reference)

      assert extra in 0..3,
             "#{@name}: #{length(float)} items against GNU Radio's #{length(data.reference)}"

      r = C.compare(data.reference, float)

      assert r.worst_lsb <= @bound["max_error_lsb"],
             "#{@name}: sample #{r.index} off by #{Float.round(r.worst_lsb, 3)} LSB (bound #{@bound["max_error_lsb"]}): #{@bound["rationale"]}"

      assert r.snr >= @bound["snr_db_min"],
             "#{@name}: SNR #{Float.round(r.snr, 1)} dB < #{@bound["snr_db_min"]}: #{@bound["rationale"]}"
    end

    for variant <- case["fixed_vs_float"] do
      @variant variant
      test "#{case["name"]}: #{variant["variant"]} vs float instance", %{runs: runs} do
        %{case: case, data: data, float: float} = runs[@name]
        fixed = C.run(case, data, C.format(@variant["format"]))
        assert length(fixed) == length(float)
        r = C.compare(float, fixed)

        assert r.worst_lsb <= @variant["max_error_lsb"],
               "#{@name} #{@variant["variant"]}: sample #{r.index} off by #{Float.round(r.worst_lsb, 2)} LSB (bound #{@variant["max_error_lsb"]}): #{@variant["rationale"]}"

        assert r.snr >= @variant["snr_db_min"],
               "#{@name} #{@variant["variant"]}: SNR #{Float.round(r.snr, 1)} dB < #{@variant["snr_db_min"]}: #{@variant["rationale"]}"
      end
    end
  end

  describe "bit sources" do
    for {case, _} <- C.cases(), case["output"]["type"] == "bytes" do
      @name case["name"]
      test "#{case["name"]}: same bits as GNU Radio (#{case["description"]})" do
        {case, data} = C.case_named(@name)
        block = C.build(case, nil)
        {bits, _} = ExSdr.Sources.generate(block, case["samples"])
        assert :binary.bin_to_list(bits) == data.reference
      end
    end
  end

  describe "design comparisons" do
    test "NCO table size: each two more table bits buy 12 dB (sig_source_f_sin)", %{runs: runs} do
      %{case: case, data: data, float: float} = runs["sig_source_f_sin"]

      snr = fn lut ->
        f = NxFixed.Num.q(16, 15, rounding: :truncate, lut_bits: lut)
        C.compare(float, C.run(case, data, f)).snr
      end

      assert_in_delta snr.(12) - snr.(10), 12.0, 1.5
      assert_in_delta snr.(14) - snr.(12), 12.0, 1.5
    end

    test "rounding the phase buys 6 dB at the same table size (sig_source_c_sin)", %{runs: runs} do
      %{case: case, data: data, float: float} = runs["sig_source_c_sin"]

      snr = fn rounding ->
        f = NxFixed.Num.q(16, 15, rounding: rounding, lut_bits: 10)
        C.compare(float, C.run(case, data, f)).snr
      end

      assert_in_delta snr.(:nearest) - snr.(:truncate), 6.0, 1.0
    end
  end
end
