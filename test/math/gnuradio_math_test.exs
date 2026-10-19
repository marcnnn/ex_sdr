defmodule ExSdr.MathGnuRadioTest do
  @moduledoc """
  Every math case (test/dsp_cases/math/*.json) against GNU Radio's output
  (test/fixtures/gnuradio/math/, committed; regenerate with
  `scripts/gnuradio_golden.sh test/fixtures/gnuradio/math test/dsp_cases/math`).

  Per case:

    1. The float instance against GNU Radio: per-sample error and SNR.
    2. Each fixed-point variant against the float instance (the width check
       runs inside the kernel compilation, so a variant that can overflow
       fails here).
    3. Split invariance: the output of the chain run in three chunks with the
       state carried equals the output of one call, bit for bit.

  The bounds and their reasoning are in each case's JSON (`rationale`).
  Errors are absolute, in the units of the output (Q15 LSB = 2^-15).
  """
  use ExUnit.Case, async: true

  Code.require_file("case_helper.exs", __DIR__)

  alias ExSdr.MathTest.Case, as: C
  alias ExSdr.Test.GnuRadioFixture, as: F

  @moduletag timeout: 300_000

  setup_all do
    runs =
      for path <- C.paths(), into: %{} do
        {case, data} = C.load(path)
        format = C.format("float")
        {out, _} = C.run(C.blocks(case, format), C.container(data.input, case, format))
        {case["name"], %{case: case, data: data, float: C.to_list(out)}}
      end

    %{runs: runs}
  end

  for path <- C.paths() do
    @external_resource path
    case = F.read_case(path)
    @name case["name"]

    test "#{@name}: float instance vs GNU Radio (#{case["description"]})", %{runs: runs} do
      %{case: case, data: data, float: float} = runs[@name]

      assert_close(
        case,
        C.reference(case, data),
        float,
        case["float_vs_gnuradio"],
        "float vs GNU Radio"
      )
    end

    for variant <- case["fixed_vs_float"] || [] do
      @variant variant

      test "#{@name}: #{@variant["variant"]} vs float instance", %{runs: runs} do
        %{case: case, data: data, float: float} = runs[@name]
        format = C.format(@variant["format"])
        {out, _} = C.run(C.blocks(case, format), C.container(data.input, case, format))

        # both sides are ours: no GNU Radio flush to account for
        same_length = Map.delete(case, "reference_extra_items")

        assert_close(
          same_length,
          float,
          C.to_list(out),
          @variant,
          "#{@name} #{@variant["variant"]} vs float"
        )
      end

      test "#{@name}: #{@variant["variant"]} is split invariant", %{runs: runs} do
        %{case: case, data: data} = runs[@name]
        format = C.format(@variant["format"])
        {whole, _} = C.run(C.blocks(case, format), C.container(data.input, case, format))

        assert chunked(case, format, data.input) == C.to_list(whole)
      end
    end

    test "#{@name}: float instance is split invariant", %{runs: runs} do
      %{case: case, data: data, float: float} = runs[@name]
      assert chunked(case, C.format("float"), data.input) == float
    end
  end

  test "delay_ff: the final state holds the samples GNU Radio flushes at the end", %{runs: runs} do
    %{case: case, data: data} = runs["delay_ff"]
    format = C.format("float")
    {_out, [state]} = C.run(C.blocks(case, format), C.container(data.input, case, format))

    # the state is the delay line, newest first; GNU Radio's extra items are the
    # oldest-first tail of the input
    tail = Enum.take(C.reference(case, data), -case["reference_extra_items"])
    assert C.to_list(state) |> Enum.reverse() == tail
  end

  # Runs the chain in three chunks (1024, 1536 and the rest), carrying the state.
  # Chunk sizes are multiples of 8, so every rate of the family divides them.
  defp chunked(case, format, input) do
    blocks = C.blocks(case, format)

    chunks = [
      Enum.take(input, 1024),
      input |> Enum.drop(1024) |> Enum.take(1536),
      Enum.drop(input, 2560)
    ]

    {outs, _states} =
      Enum.reduce(chunks, {[], nil}, fn chunk, {acc, states} ->
        {out, next} = C.run(blocks, C.container(chunk, case, format), states)
        {[C.to_list(out) | acc], next}
      end)

    outs |> Enum.reverse() |> Enum.concat()
  end

  def assert_close(case, reference, signal, expect, label) do
    extra = case["reference_extra_items"] || 0

    assert length(reference) == length(signal) + extra,
           "#{label}: #{length(signal)} items, GNU Radio #{length(reference)}"

    ref = Enum.take(reference, length(signal))

    {worst, i} = C.worst(ref, signal)

    assert worst <= expect["max_error"], """
    #{label}: sample #{i} is off by #{worst} (bound #{expect["max_error"]})
      reference: #{inspect(Enum.at(ref, i))}
      got:       #{inspect(Enum.at(signal, i))}
      bound:     #{expect["rationale"]}
    """

    snr = F.snr_db(ref, signal)

    assert snr >= expect["snr_db_min"],
           "#{label}: SNR #{Float.round(snr, 1)} dB < #{expect["snr_db_min"]} dB (#{expect["rationale"]})"
  end
end
