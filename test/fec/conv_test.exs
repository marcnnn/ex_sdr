defmodule ExSdr.FEC.ConvTest do
  @moduledoc """
  gr-fec's convolutional encoder and decoder against GNU Radio's output
  (test/fixtures/gnuradio/fec/conv.json, made by test/gnuradio/fec_conv_golden.py):
  encoders and decoders are driven frame by frame through the generic
  interface, bits must be equal.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias ExSdr.FEC.Conv
  alias ExSdr.FEC.ConvViterbi
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

  @codes [
    {3, [7, 5]},
    {4, [11, 13]},
    {5, [25, 27]},
    {7, [79, 109]},
    {7, [-79, 109]},
    {7, [79, 109, 89]},
    {9, [491, 369]},
    # no tap on the newest input in the first polynomial: metrics are computed directly
    {4, [14, 13]}
  ]

  describe "properties" do
    property "noiseless frames decode to the input in every mode, for any K and rate" do
      check all(
              {k, polys} <- StreamData.member_of(@codes),
              mode <- StreamData.member_of([:streaming, :terminated, :truncated, :tailbiting]),
              f <- StreamData.integer(24..60),
              start <- StreamData.integer(0..7),
              seed <- StreamData.integer(),
              max_runs: 40
            ) do
        start = if mode == :tailbiting, do: 0, else: rem(start, Bitwise.<<<(1, k - 1))
        o = [frame_size: f, k: k, polys: polys, mode: mode, start_state: start]
        e = Conv.encoder(o)
        d = Conv.decoder(o)
        :rand.seed(:exsss, {seed, 7, 8})
        data = for _ <- 1..f, do: :rand.uniform(2) - 1
        {coded, _} = Generic.work(e, Nx.tensor(data, type: :u8), Generic.init(e))

        soft = coded |> Nx.multiply(255) |> Nx.as_type(:u8)

        # streaming frames read look-ahead items: give the next frame's start
        soft =
          if Generic.history(d) > 0,
            do:
              Nx.concatenate([
                soft,
                Nx.broadcast(Nx.tensor(128, type: :u8), {Generic.history(d)})
              ]),
            else: soft

        {out, _} = Generic.work(d, soft, Generic.init(d))
        assert Nx.to_flat_list(out) == data
      end
    end

    property "the encoder is linear (mod 2) from the zero state" do
      check all(
              {k, polys} <-
                StreamData.member_of(
                  Enum.reject(@codes, fn {_, p} -> Enum.any?(p, &(&1 < 0)) end)
                ),
              seed <- StreamData.integer(),
              max_runs: 30
            ) do
        o = [frame_size: 40, k: k, polys: polys, mode: :truncated]
        e = Conv.encoder(o)
        :rand.seed(:exsss, {seed, 1, 5})
        a = for _ <- 1..40, do: :rand.uniform(2) - 1
        b = for _ <- 1..40, do: :rand.uniform(2) - 1
        ab = Enum.zip_with(a, b, &Bitwise.bxor/2)

        enc = fn x ->
          Generic.work(e, Nx.tensor(x, type: :u8), Generic.init(e))
          |> elem(0)
          |> Nx.to_flat_list()
        end

        assert enc.(ab) == Enum.zip_with(enc.(a), enc.(b), &Bitwise.bxor/2)
      end
    end
  end

  describe "integer metrics" do
    test "path metrics stay within u8 and the minimum is zero after every step" do
      t = ConvViterbi.tables(7, [79, 109])
      :rand.seed(:exsss, {3, 3, 3})

      for sigma <- [30, 90, 255] do
        syms =
          for _ <- 1..400,
              do: min(255, max(0, round(:rand.uniform(2) * 255.0 - 255 + :rand.normal() * sigma)))

        metrics = ConvViterbi.init_metrics(t, :unbiased, 31)

        {_, final} = ConvViterbi.run(t, syms, metrics)
        values = Tuple.to_list(final)
        assert Enum.min(values) == 0
        assert Enum.max(values) <= 255
      end
    end

    test "the branch metric is a distance in 0..max with max = 63 for rate 1/2" do
      t = ConvViterbi.tables(7, [79, 109])
      assert t.max == 63
      assert ConvViterbi.distance([0, 0], [0, 0]) == 0
      assert ConvViterbi.distance([255, 255], [0, 0]) == 63
      assert ConvViterbi.distance([255, 0], [255, 255]) == 32
    end

    test "polynomials match the Comm trellis after reversing the register" do
      # GNU Radio's 79 and 109 are 0o171 and 0o133 read the other way round
      assert ConvViterbi.reverse_bits(79, 7) == 0o171
      assert ConvViterbi.reverse_bits(109, 7) == 0o133
    end
  end
end
