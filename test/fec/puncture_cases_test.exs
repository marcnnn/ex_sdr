defmodule ExSdr.FEC.PunctureCasesTest do
  @moduledoc """
  The cases of test/dsp_cases/fec (puncture_bb, puncture_ff, depuncture_bb
  on random input) against GNU Radio's committed output in
  test/fixtures/gnuradio/fec: items must be equal.
  """
  use ExUnit.Case, async: true

  alias ExSdr.Test.GnuRadioFixture

  for {case, _} <- GnuRadioFixture.family("fec") do
    @case_name case["name"]

    test "#{case["name"]}: #{case["description"]}" do
      {case, data} =
        Enum.find(GnuRadioFixture.family("fec"), fn {c, _} -> c["name"] == @case_name end)

      name = String.to_atom(case["block"]["name"])
      opts = for {k, v} <- case["block"]["opts"], do: {String.to_atom(k), v}
      block = ExSdr.FEC.block(name, opts)

      input =
        case case["input"]["type"] do
          "bytes" -> Nx.tensor(data.input, type: :u8)
          "real" -> data.input |> Nx.tensor(type: :f32) |> Nx.divide(32768.0)
        end

      {out, _} = ExSdr.FEC.run(block, input)
      assert Nx.to_flat_list(out) == data.reference
    end
  end
end
