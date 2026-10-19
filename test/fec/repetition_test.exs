defmodule ExSdr.FEC.RepetitionTest do
  @moduledoc """
  Repetition and dummy codes against GNU Radio (test/fixtures/gnuradio/fec/misc.json,
  made by test/gnuradio/fec_misc_golden.py): bits must be equal.
  """
  use ExUnit.Case, async: true

  alias ExSdr.FEC.Dummy
  alias ExSdr.FEC.Generic
  alias ExSdr.FEC.Repetition

  @fixture Path.expand("../fixtures/gnuradio/fec/misc.json", __DIR__)

  setup_all do
    %{gr: @fixture |> File.read!() |> JSON.decode!()}
  end

  defp bits(s), do: for(<<c <- s>>, do: c - ?0)
  defp bit_string(t), do: t |> Generic.to_binary() |> :binary.bin_to_list() |> Enum.join()
  defp hex(s), do: Base.decode16!(s, case: :lower)

  test "repetition encoder and decoder equal GNU Radio", %{gr: gr} do
    for c <- gr["repetition"] do
      enc = Repetition.encoder(frame_size: c["frame_size"], rep: c["rep"])
      dec = Repetition.decoder(frame_size: c["frame_size"], rep: c["rep"], ap_prob: c["ap_prob"])

      assert Generic.input_size(enc) == c["enc_input_size"]
      assert Generic.output_size(enc) == c["enc_output_size"]
      assert Generic.rate(enc) == c["enc_rate"]
      assert Generic.input_size(dec) == c["dec_input_size"]
      assert Generic.output_size(dec) == c["dec_output_size"]
      assert_in_delta Generic.rate(dec), c["dec_rate"], 1.0e-12

      for {f, e} <- Enum.zip(c["frames"], c["encoded"]) do
        {out, _} = Generic.work(enc, Nx.tensor(bits(f), type: :u8), nil)
        assert bit_string(out) == e
      end

      for {s, d} <- Enum.zip(c["soft"], c["decoded"]) do
        {out, _} = Generic.work(dec, Nx.from_binary(hex(s), :f32), nil)
        assert bit_string(out) == d, "rep #{c["rep"]} ap #{c["ap_prob"]}"
      end
    end
  end

  test "repetition as blocks, split invariance of the frame stream", %{gr: gr} do
    c = hd(gr["repetition"])
    enc = ExSdr.FEC.block(:repetition_encoder, frame_size: c["frame_size"], rep: c["rep"])
    all = c["frames"] |> Enum.flat_map(&bits/1) |> Nx.tensor(type: :u8)
    {whole, _} = ExSdr.FEC.run(enc, all)
    assert bit_string(whole) == Enum.join(c["encoded"])

    f = c["frame_size"]
    {a, st} = ExSdr.FEC.run(enc, Nx.slice_along_axis(all, 0, f))
    {b, _} = ExSdr.FEC.run(enc, Nx.slice_along_axis(all, f, 2 * f), st)
    assert bit_string(Nx.concatenate([a, b])) == bit_string(whole)

    dec = ExSdr.FEC.block(:repetition_decoder, frame_size: c["frame_size"], rep: c["rep"], ap_prob: c["ap_prob"])
    assert dec.rate == {f * c["rep"], f}
    assert dec.hardware == :stateless
  end

  test "dummy encoder and decoder equal GNU Radio", %{gr: gr} do
    for c <- gr["dummy_encoders"] do
      enc = Dummy.encoder(frame_size: c["frame_size"], pack: c["pack"], packed_bits: c["packed_bits"])
      assert Generic.input_size(enc) == c["input_size"]
      assert Generic.output_size(enc) == c["output_size"]
      assert Generic.input_conversion(enc) == c["input_conversion"]
      assert Generic.output_conversion(enc) == c["output_conversion"]
      assert Generic.rate(enc) == c["rate"]
      {out, _} = Generic.work(enc, Nx.from_binary(hex(c["input"]), :u8), nil)
      assert out |> Generic.to_binary() |> Base.encode16(case: :lower) == c["output"]
    end

    d = gr["dummy_decoder"]
    dec = Dummy.decoder(frame_size: 16)
    assert Generic.input_conversion(dec) == d["input_conversion"]
    {out, _} = Generic.work(dec, Nx.from_binary(hex(d["input"]), :f32), nil)
    assert Nx.to_flat_list(out) == d["output"]
  end
end
