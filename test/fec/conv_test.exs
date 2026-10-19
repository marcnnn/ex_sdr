defmodule ExSdr.FEC.ConvTest do
  @moduledoc """
  gr-fec's convolutional encoder and decoder against GNU Radio's output
  (test/fixtures/gnuradio/fec/conv.json, made by test/gnuradio/fec_conv_golden.py):
  encoders and decoders are driven frame by frame through the generic
  interface, bits must be equal.
  """
  use ExUnit.Case, async: true

  alias ExSdr.FEC.Conv
  alias ExSdr.FEC.Generic

  @fixture Path.expand("../fixtures/gnuradio/fec/conv.json", __DIR__)

  setup_all do
    %{gr: @fixture |> File.read!() |> JSON.decode!()}
  end

  defp bits(s), do: for(<<c <- s>>, do: c - ?0)
  defp bin_bits(t), do: t |> Generic.to_binary() |> :binary.bin_to_list()
  defp mode(m), do: String.to_atom(m)

  describe "cc_encoder" do
    test "equals GNU Radio in all modes, including carried state", %{gr: gr} do
      for c <- gr["encoders"] do
        enc =
          Conv.encoder(
            frame_size: c["frame_size"],
            k: c["k"],
            polys: c["polys"],
            start_state: c["start_state"],
            mode: mode(c["mode"]),
            padded: c["padded"]
          )

        assert Generic.input_size(enc) == c["input_size"]
        assert Generic.output_size(enc) == c["output_size"]
        assert Generic.rate(enc) == c["rate"]

        {outs, _} =
          Enum.map_reduce(c["frames"], Generic.init(enc), fn f, st ->
            {o, st} = Generic.work(enc, Nx.tensor(bits(f), type: :u8), st)
            {bin_bits(o), st}
          end)

        assert outs == Enum.map(c["outputs"], &bits/1),
               "k #{c["k"]} #{inspect(c["polys"])} #{c["mode"]} start #{c["start_state"]} padded #{c["padded"]}"
      end
    end

    test "as a block over several frames equals frame by frame", %{gr: gr} do
      c = Enum.find(gr["encoders"], &(&1["mode"] == "streaming" and &1["k"] == 7))

      b =
        ExSdr.FEC.block(:cc_encoder,
          frame_size: c["frame_size"],
          k: 7,
          polys: c["polys"],
          start_state: c["start_state"],
          mode: :streaming
        )

      input = c["frames"] |> Enum.flat_map(&bits/1) |> Nx.tensor(type: :u8)
      {out, _} = NxFixed.Block.run(b, input)
      assert bin_bits(out) == Enum.flat_map(c["outputs"], &bits/1)
    end
  end

  describe "cc_decoder" do
    test "equals GNU Radio's decisions on the same 8 bit soft input", %{gr: gr} do
      for c <- gr["decoders"] do
        dec =
          Conv.decoder(
            frame_size: c["frame_size"],
            k: c["k"],
            polys: c["polys"],
            start_state: c["start_state"],
            end_state: c["end_state"],
            mode: mode(c["mode"]),
            padded: c["padded"]
          )

        assert Generic.input_size(dec) == c["input_size"]
        assert Generic.output_size(dec) == c["output_size"]
        assert Generic.history(dec) == c["history"]
        assert Generic.shift(dec) == c["shift"]
        assert Generic.input_conversion(dec) == c["input_conversion"]

        {outs, _} =
          Enum.map_reduce(c["inputs"], Generic.init(dec), fn hex, st ->
            input = hex |> Base.decode16!(case: :lower) |> Nx.from_binary(:u8)
            {o, st} = Generic.work(dec, input, st)
            {bin_bits(o), st}
          end)

        assert outs == Enum.map(c["outputs"], &bits/1),
               "#{c["mode"]} sigma #{c["sigma"]} start #{c["start_state"]} end #{c["end_state"]} padded #{c["padded"]}"
      end
    end
  end

  test "float to 8 bit conversion equals blocks.float_to_uchar", %{gr: gr} do
    %{"input" => x, "output" => y} = gr["float_to_uchar"]
    assert Generic.to_u8(x) |> Nx.to_flat_list() == y
  end
end
