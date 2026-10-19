Code.require_file("support/case_helper.exs", __DIR__)

defmodule ExSdr.Digital.GnuRadioCasesTest do
  @moduledoc """
  Every case of test/dsp_cases/digital against GNU Radio's committed output
  (test/fixtures/gnuradio/digital): bit and byte results must be equal,
  sample results within the bound of the case file. Fixed-point instances
  run through the width check (`NxFixed.run_scan_stacked/5` does it) and are
  compared with the float instance.
  """
  use ExUnit.Case, async: true

  alias ExSdr.Digital
  alias ExSdr.Digital.CaseHelper, as: H

  for case <- H.case_files() do
    @name case["name"]
    @case_name case["name"]

    test "#{case["name"]}: float instance vs GNU Radio (#{case["description"]})" do
      {case, data} = H.load(@case_name)
      type = H.output_type(case)
      {out, _} = Digital.run(H.block(case, H.format(nil)), H.input(case, data, H.format(nil)))
      got = H.items(out, type)
      assert length(got) == length(data.reference)

      case case["float_vs_gnuradio"] do
        nil ->
          assert got == data.reference

        bound ->
          worst = H.max_abs(got, data.reference)

          assert worst <= bound["max_abs"],
                 "#{@name}: #{worst} > #{bound["max_abs"]} (#{bound["rationale"]})"
      end
    end

    for variant <- case["fixed_vs_float"] || [] do
      @variant variant
      test "#{case["name"]}: #{variant["variant"]} vs float instance" do
        {case, data} = H.load(@case_name)
        type = H.output_type(case)
        fmt = H.format(@variant["format"])
        {ref, _} = Digital.run(H.block(case, H.format(nil)), H.input(case, data, H.format(nil)))
        {out, _} = Digital.run(H.block(case, fmt), H.input(case, data, fmt))
        {ref, got} = {H.items(ref, type), H.items(out, type)}

        case @variant do
          %{"max_mismatches" => n} ->
            assert H.mismatches(ref, got) <= n,
                   "#{@name}: #{H.mismatches(ref, got)} mismatches > #{n} (#{@variant["rationale"]})"

          %{"max_abs" => bound} ->
            worst = H.max_abs(ref, got)

            assert worst <= bound,
                   "#{@name} #{@variant["variant"]}: #{worst} > #{bound} (#{@variant["rationale"]})"
        end
      end
    end
  end
end
