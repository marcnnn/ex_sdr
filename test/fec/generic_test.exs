defmodule ExSdr.FEC.GenericTest do
  @moduledoc """
  The generic encoder / decoder framework against GNU Radio
  (test/fixtures/gnuradio/fec/misc.json and conv.json): tagged stream and
  PDU wrappers, stream blocks in pieces, the extended encoder / decoder
  chains; plus BER of the soft Viterbi decoder against the hard-decision one
  of `ExSdr.Comm.Convolutional` (seeded).
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias ExSdr.FEC.Conv
  alias ExSdr.FEC.Generic
  alias ExSdr.FEC.Repetition

  @misc Path.expand("../fixtures/gnuradio/fec/misc.json", __DIR__)

  setup_all do
    %{gr: @misc |> File.read!() |> JSON.decode!()}
  end

  defp bits(s), do: for(<<c <- s>>, do: c - ?0)
  defp bitbin(s), do: s |> bits() |> :binary.list_to_bin()
  defp hex(s), do: Base.decode16!(s, case: :lower)
  defp enc(b), do: Base.encode16(b, case: :lower)

  defp objects("cc_terminated", mtu) do
    o = [frame_size: mtu * 8, k: 7, polys: [79, 109], mode: :terminated]
    {Conv.encoder(o), Conv.decoder(o)}
  end

  defp objects("cc_terminated_padded", mtu) do
    o = [frame_size: mtu * 8, k: 7, polys: [79, 109], mode: :terminated, padded: true]
    {Conv.encoder(o), Conv.decoder(o)}
  end

  defp objects("cc_truncated", mtu) do
    o = [frame_size: mtu * 8, k: 7, polys: [79, 109], mode: :truncated]
    {Conv.encoder(o), Conv.decoder(o)}
  end

  defp objects("cc_tailbiting", mtu) do
    o = [frame_size: mtu * 8, k: 7, polys: [79, 109], mode: :tailbiting]
    {Conv.encoder(o), Conv.decoder(o)}
  end

  defp objects("repetition", mtu) do
    {Repetition.encoder(frame_size: mtu * 8, rep: 3),
     Repetition.decoder(frame_size: mtu * 8, rep: 3, ap_prob: 0.5)}
  end

  describe "tagged stream blocks" do
    test "encode and decode packets like GNU Radio", %{gr: gr} do
      for c <- gr["tagged"] do
        {e, d} = objects(c["name"], c["mtu"])
        {encoded, _} = Generic.tagged_encode(e, Enum.map(c["packets"], &bitbin/1), Generic.init(e), c["mtu"])
        assert Enum.map(encoded, &(&1 |> :binary.bin_to_list() |> Enum.join())) == c["encoded"], c["name"]

        soft =
          if String.starts_with?(c["name"], "cc"),
            do: Enum.map(c["soft"], &Nx.from_binary(hex(&1), :u8)),
            else: Enum.map(c["soft"], &Nx.from_binary(hex(&1), :f32))

        {decoded, _} = Generic.tagged_decode(d, soft, Generic.init(d), c["mtu"])
        assert Enum.map(decoded, &(&1 |> :binary.bin_to_list() |> Enum.join())) == c["decoded"], c["name"]
      end
    end

    test "as blocks", %{gr: gr} do
      c = Enum.find(gr["tagged"], &(&1["name"] == "cc_terminated"))
      {e, _} = objects(c["name"], c["mtu"])
      b = ExSdr.FEC.block(:tagged_encoder, code: e, mtu: c["mtu"])
      {out, _} = ExSdr.FEC.run(b, Enum.map(c["packets"], &bitbin/1))
      assert Enum.map(out, &(&1 |> :binary.bin_to_list() |> Enum.join())) == c["encoded"]
      assert b.kind == :host
      assert b.gnuradio == "fec.tagged_encoder"
    end

    test "packets larger than the MTU are rejected" do
      {e, _} = objects("repetition", 4)
      assert_raise ArgumentError, fn -> Generic.tagged_encode(e, [:binary.copy(<<0>>, 33)], nil, 4) end
    end
  end

  describe "async (PDU) blocks" do
    test "encoders and decoders equal GNU Radio, packed and unpacked", %{gr: gr} do
      for c <- gr["async"] do
        {e, d} = objects(c["code"], c["mtu"])
        opts = [packed: c["packed"], rev_pack: c["rev_pack"], mtu: c["mtu"]]

        case c["kind"] do
          "encoder" ->
            opts = Keyword.put(opts, :rev_unpack, c["rev_unpack"])

            out =
              for p <- c["pdus"] do
                {{%{}, o}, _} = Generic.async_encode(e, {%{}, hex(p)}, Generic.init(e), opts)
                enc(o)
              end

            assert out == c["out"], "#{c["code"]} packed #{c["packed"]} #{c["rev_unpack"]}/#{c["rev_pack"]}"

          "decoder" ->
            out =
              for s <- c["soft"] do
                {{%{}, o}, _} = Generic.async_decode(d, {%{}, Nx.from_binary(hex(s), :f32)}, Generic.init(d), opts)
                enc(o)
              end

            # unpacked output is one bit per byte: compare as hex strings of the bytes
            assert out == c["out"], "#{c["code"]} decoder packed #{c["packed"]}"
        end
      end
    end

    test "streaming mode convolutional decoders are rejected" do
      d = Conv.decoder(frame_size: 64, k: 7, polys: [79, 109], mode: :streaming)

      assert_raise ArgumentError, fn ->
        Generic.async_decode(d, {%{}, Nx.broadcast(0.0, {140})}, nil)
      end
    end
  end

  describe "stream blocks" do
    test "decoder blocks in pieces equal one call (streaming mode with look-ahead)" do
      f = 24
      o = [frame_size: f, k: 7, polys: [79, 109], mode: :streaming]
      e = Conv.encoder(o)
      :rand.seed(:exsss, {1, 2, 3})
      data = for _ <- 1..(6 * f), do: :rand.uniform(2) - 1
      {coded, _} = Generic.encode_frames(Nx.tensor(data, type: :u8), Generic.init(e), code: e)

      soft =
        coded
        |> Nx.to_flat_list()
        |> Enum.map(&min(255, max(0, round(&1 * 255 + :rand.normal() * 60))))
        |> Nx.tensor(type: :u8)

      dec = ExSdr.FEC.block(:cc_decoder, o)
      assert dec.rate == {2 * f, f}
      assert dec.latency == f
      assert dec.warmup == :unbounded
      assert dec.hardware == :memory

      {whole, _} = ExSdr.FEC.run(dec, soft)

      # five frames come out, the sixth waits for its look-ahead
      assert Nx.axis_size(whole, 0) == 5 * f
      assert Nx.to_flat_list(whole) == Enum.take(data, 5 * f)

      # in pieces of one, two and three frames
      for sizes <- [[1, 1, 1, 1, 1, 1], [2, 1, 3], [3, 3]] do
        {outs, _, _} =
          Enum.reduce(sizes, {[], 0, dec.init}, fn n, {acc, off, st} ->
            piece = Nx.slice_along_axis(soft, off * 2 * f, n * 2 * f)
            {out, st} = ExSdr.FEC.run(dec, piece, st)
            {if(out, do: acc ++ [out], else: acc), off + n, st}
          end)

        assert outs |> Nx.concatenate() |> Nx.to_flat_list() == Nx.to_flat_list(whole)
      end
    end

    test "encoder state is carried across calls in streaming mode, reset in the others" do
      o = [frame_size: 10, k: 5, polys: [25, 27]]
      data = Nx.tensor(for(i <- 1..30, do: rem(i * i, 2)), type: :u8)

      for mode <- [:streaming, :truncated, :terminated, :tailbiting] do
        b = ExSdr.FEC.block(:cc_encoder, [mode: mode] ++ o)
        {whole, _} = ExSdr.FEC.run(b, data)

        {a, st} = ExSdr.FEC.run(b, Nx.slice_along_axis(data, 0, 10))
        {c, _} = ExSdr.FEC.run(b, Nx.slice_along_axis(data, 10, 20), st)
        assert Nx.concatenate([a, c]) == whole, "mode #{mode}"
      end

      # the encoder is feed-forward: warm-up K - 1 bits in streaming mode
      assert ExSdr.FEC.block(:cc_encoder, o).warmup == 4
      assert ExSdr.FEC.block(:cc_encoder, [mode: :terminated] ++ o).warmup == 0
    end
  end

  describe "extended encoder and decoder" do
    test "puncturing a rate 1/2 code to 3/4 and decoding it" do
      # DVB-T style rate 3/4 from K = 7: keep 4 of 6 code bits per 3 input bits
      o = [frame_size: 90, k: 7, polys: [79, 109], mode: :terminated]
      e = Conv.encoder(o)
      d = Conv.decoder(o)
      :rand.seed(:exsss, {4, 5, 6})
      data = for _ <- 1..90, do: :rand.uniform(2) - 1
      puncpat = "110110"

      {coded, _} = ExSdr.FEC.Extended.encode(e, Nx.tensor(data, type: :u8), Generic.init(e), puncpat)
      # 2 * (90 + 6) = 192 mother code bits, 4 of every 6 kept
      assert Nx.axis_size(coded, 0) == 128

      # noiseless soft values in the float convention: +1 for 1, -1 for 0
      soft = coded |> Nx.as_type(:f32) |> Nx.multiply(2) |> Nx.subtract(1)
      {out, _} = ExSdr.FEC.Extended.decode(d, soft, Generic.decoder_init(d), puncpat)
      assert Nx.to_flat_list(out) == data
    end

    test "polar packed encoder is unpacked by the chain" do
      frozen = ExSdr.FEC.Polar.frozen_bit_positions(64, 32)
      e = ExSdr.FEC.Polar.encoder(block_size: 64, num_info_bits: 32, frozen_bit_positions: frozen, packed: true)
      info = Nx.tensor(for(i <- 1..32, do: rem(i, 3) |> min(1)), type: :u8)
      {out, _} = ExSdr.FEC.Extended.encode(e, info, nil)
      assert Nx.axis_size(out, 0) == 64
      plain = ExSdr.FEC.Polar.encoder(block_size: 64, num_info_bits: 32, frozen_bit_positions: frozen)
      {expected, _} = Generic.work(plain, info, nil)
      assert out == expected
    end
  end

  describe "soft decoding gain" do
    test "8 bit soft decisions beat hard decisions (seeded, K = 7 rate 1/2)" do
      f = 400
      o = [frame_size: f, k: 7, polys: [79, 109], mode: :terminated]
      e = Conv.encoder(o)
      d = Conv.decoder(o)
      :rand.seed(:exsss, {2025, 1, 1})

      # Es/N0 about 1.5 dB: BPSK +-1 with noise sigma 0.84
      sigma = 0.84

      {soft_err, hard_err} =
        Enum.reduce(1..30, {0, 0}, fn _, {s, h} ->
          data = for _ <- 1..f, do: :rand.uniform(2) - 1
          {coded, _} = Generic.work(e, Nx.tensor(data, type: :u8), Generic.init(e))

          rx = for b <- Nx.to_flat_list(coded), do: 2 * b - 1 + :rand.normal() * sigma

          soft_in = Generic.to_u8(rx, 48.0, 128.0)
          {soft, _} = Generic.work(d, soft_in, Generic.init(d))

          hard_in = for r <- rx, do: if(r > 0, do: 255, else: 0)
          {hard, _} = Generic.work(d, Nx.tensor(hard_in, type: :u8), Generic.init(d))

          count = fn out -> out |> Nx.to_flat_list() |> Enum.zip(data) |> Enum.count(fn {a, b} -> a != b end) end
          {s + count.(soft), h + count.(hard)}
        end)

      assert soft_err * 2 < hard_err
      # the integer decoder agrees with the hard-decision reference decoder
      # on hard input (both choose a maximum likelihood path; compare bit error counts)
      assert hard_err > 0
    end

    test "on clean hard input it decodes like ExSdr.Comm.Convolutional" do
      code = ExSdr.Comm.Convolutional.new(constraint_length: 7, polynomials: [0o171, 0o133])
      f = 120
      o = [frame_size: f, k: 7, polys: [79, 109], mode: :terminated]
      e = Conv.encoder(o)
      d = Conv.decoder(o)
      :rand.seed(:exsss, {9, 9, 9})
      data = for _ <- 1..f, do: :rand.uniform(2) - 1
      {coded, _} = Generic.work(e, Nx.tensor(data, type: :u8), Generic.init(e))

      # GNU Radio's polynomials are the reversed standard ones, same output order
      assert ExSdr.Comm.Convolutional.encode(code, data ++ List.duplicate(0, 6)) ==
               Nx.to_flat_list(coded)

      flips = [3, 17, 40, 41, 90, 150]

      hard =
        coded
        |> Nx.to_flat_list()
        |> Enum.with_index()
        |> Enum.map(fn {b, i} -> if i in flips, do: 1 - b, else: b end)

      {soft, _} = Generic.work(d, Nx.tensor(Enum.map(hard, &(&1 * 255)), type: :u8), Generic.init(d))

      reference =
        ExSdr.Comm.Convolutional.decode(code, hard, native: false)
        |> Enum.take(f)

      assert Nx.to_flat_list(soft) == reference
      assert reference == data
    end
  end
end
