defmodule ExSdr.FEC.LDPCGnuRadioTest do
  @moduledoc """
  The LDPC blocks against GNU Radio's gr-fec (reference data from
  test/gnuradio/fec_ldpc_golden.py, cases in test/dsp_cases/fec/ldpc/).
  Encoders: equal codewords. Bit flip and sum-product decoders: equal
  decoded bits.
  """
  use ExUnit.Case, async: true
  @moduletag :ldpc_alist

  alias ExSdr.FEC.LDPC

  @cases_dir Path.expand("../dsp_cases/fec/ldpc", __DIR__)
  @fixtures Path.expand("../fixtures/gnuradio/fec", __DIR__)
  @alist_dir Path.join(@fixtures, "ldpc_alist")

  @cases @cases_dir
         |> Path.join("ldpc_*.json")
         |> Path.wildcard()
         |> Enum.sort()
         |> Enum.map(&{&1, JSON.decode!(File.read!(&1))})

  defp load(path, case) do
    dir = Path.join(@fixtures, case["name"])
    manifest = dir |> Path.join("manifest.json") |> File.read!() |> JSON.decode!()
    sha = :crypto.hash(:sha256, File.read!(path)) |> Base.encode16(case: :lower)

    assert manifest["case_sha256"] == sha,
           "GNU Radio data for #{case["name"]} is stale: run test/gnuradio/fec_ldpc_golden.py"

    input =
      if File.exists?(Path.join(dir, "input.u8")),
        do: Nx.from_binary(File.read!(Path.join(dir, "input.u8")), :u8),
        else: Nx.from_binary(File.read!(Path.join(dir, "input.f32le")), :f32)

    %{manifest: manifest, input: input, reference: File.read!(Path.join(dir, "gnuradio.u8"))}
  end

  defp block(case) do
    alist = Path.join(@alist_dir, case["alist"])

    case case["block"] do
      "ldpc_par_mtrx_encoder" ->
        LDPC.block(:ldpc_par_mtrx_encoder, code: LDPC.h_matrix(alist, case["gap"]))

      "ldpc_gen_mtrx_encoder" ->
        LDPC.block(:ldpc_gen_mtrx_encoder, code: LDPC.g_matrix(alist))

      "ldpc_encoder" ->
        LDPC.block(:ldpc_encoder, code: LDPC.cldpc(alist))

      "ldpc_bit_flip_decoder" ->
        code =
          if case["matrix"] == "H",
            do: LDPC.h_matrix(alist, case["gap"]),
            else: LDPC.g_matrix(alist)

        LDPC.block(:ldpc_bit_flip_decoder, code: code, max_iter: case["max_iter"])

      "ldpc_decoder" ->
        LDPC.block(:ldpc_decoder, code: LDPC.cldpc(alist), max_iter: case["max_iter"])
    end
  end

  for {path, case} <- @cases do
    @path path
    @case case
    test "#{case["name"]}: equal to GNU Radio's #{case["block"]}" do
      data = load(@path, @case)
      b = block(@case)
      assert b.gnuradio == "fec.#{@case["block"]}"

      assert {data.manifest["k"], data.manifest["n"]} in [
               b.rate,
               {elem(b.rate, 1), elem(b.rate, 0)}
             ]

      {out, _} = LDPC.run(b, data.input)
      assert Nx.to_binary(out) == data.reference
    end
  end

  test "the case files cover every GNU Radio LDPC block ported" do
    blocks = @cases |> Enum.map(fn {_, c} -> c["block"] end) |> Enum.uniq() |> Enum.sort()

    assert blocks ==
             ~w(ldpc_bit_flip_decoder ldpc_decoder ldpc_encoder ldpc_gen_mtrx_encoder ldpc_par_mtrx_encoder)
  end
end
