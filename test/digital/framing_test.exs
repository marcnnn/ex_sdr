defmodule ExSdr.Digital.FramingTest do
  @moduledoc """
  Framing, CRC and packet blocks against GNU Radio's output for the same
  inputs (test/fixtures/gnuradio/digital/framing.json, made by
  test/gnuradio/digital_framing_golden.py): all results are bits and bytes
  and must be equal.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias ExSdr.Digital
  alias ExSdr.Digital.CRC
  alias ExSdr.Digital.Framing

  @fixture Path.expand("../fixtures/gnuradio/digital/framing.json", __DIR__)

  setup_all do
    %{gr: @fixture |> File.read!() |> JSON.decode!()}
  end

  defp hex(s), do: Base.decode16!(s, case: :lower)
  defp enc(b), do: Base.encode16(b, case: :lower)

  describe "CRC" do
    test "equals GNU Radio's digital.crc for widths 8, 16, 32 and 64", %{gr: gr} do
      for %{"params" => [w, poly, init, xorout, rin, rout], "messages" => msgs} <- gr["crc"],
          %{"data" => data, "crc" => crc} <- msgs do
        c =
          CRC.new(
            width: w,
            poly: poly,
            init: init,
            xorout: xorout,
            reflect_in: rin,
            reflect_out: rout
          )

        assert CRC.compute(c, hex(data)) == crc,
               "width #{w} poly #{Integer.to_string(poly, 16)} data #{data}"
      end
    end

    test "check value of CRC-32" do
      c = Framing.crc32()
      assert CRC.compute(c, "123456789") == 0xCBF43926
    end

    test "widths below a byte (CRC-5/USB, CRC-3/GSM)" do
      usb =
        CRC.new(
          width: 5,
          poly: 0x05,
          init: 0x1F,
          reflect_in: true,
          reflect_out: true,
          xorout: 0x1F
        )

      assert CRC.compute(usb, "123456789") == 0x19
      gsm = CRC.new(width: 3, poly: 0x3, init: 0, xorout: 0x7)
      assert CRC.compute(gsm, "123456789") == 0x4
    end

    property "data in pieces gives the same CRC (register carried)" do
      c = Framing.crc32()

      check all(data <- binary(max_length: 200), cut <- integer(0..200)) do
        cut = min(cut, byte_size(data))
        <<a::binary-size(cut), b::binary>> = data
        reg = c |> CRC.update(CRC.init_state(c), a) |> then(&CRC.update(c, &1, b))
        assert CRC.finish(reg, c) == CRC.compute(c, data)
      end
    end
  end

  describe "crc32_bb" do
    test "packed and unpacked, append and check, equal GNU Radio", %{gr: gr} do
      for %{"check" => check, "packed" => packed, "in" => ins, "out" => outs} <- gr["crc32_bb"] do
        dec = if packed, do: &hex/1, else: &bits/1
        out = &if(packed, do: enc(&1), else: unbits(&1))
        block = Digital.block(:crc32_bb, check: check, packed: packed)
        {got, nil} = Digital.run(block, Enum.map(ins, dec))
        assert Enum.map(got, out) == outs, "check #{check} packed #{packed}"
      end
    end
  end

  defp bits(s), do: for(<<c <- s>>, into: <<>>, do: <<c - ?0>>)
  defp unbits(b), do: for(<<x <- b>>, into: "", do: <<x + ?0>>)

  describe "PDU CRC blocks" do
    test "crc_append and crc_check equal GNU Radio", %{gr: gr} do
      for %{"config" => cfg, "in" => ins, "appended" => appended, "check" => checks} <-
            gr["crc_pdu"] do
        [n, poly, init, xor, rin, rout, swap, skip] = cfg

        opts = [
          num_bits: n,
          poly: poly,
          initial_value: init,
          final_xor: xor,
          input_reflected: rin,
          result_reflected: rout,
          swap_endianness: swap,
          skip_header_bytes: skip
        ]

        {got, _} = Digital.run(Digital.block(:crc_append, opts), Enum.map(ins, &hex/1))
        assert Enum.map(got, &enc/1) == appended, "append #{inspect(cfg)}"

        for %{"discard_crc" => discard, "in" => cin, "ok" => ok, "fail" => fail} <- checks do
          block = Digital.block(:crc_check, [discard_crc: discard] ++ opts)
          {got, _} = Digital.run(block, Enum.map(cin, &hex/1))
          assert Enum.map(got.ok, &enc/1) == ok, "ok #{inspect(cfg)} discard #{discard}"
          assert Enum.map(got.fail, &enc/1) == fail, "fail #{inspect(cfg)} discard #{discard}"
        end
      end
    end

    test "crc32_async_bb and crc16_async_bb equal GNU Radio", %{gr: gr} do
      for %{"block" => name, "in" => ins, "out" => outs} <- gr["crc_async"] do
        {got, _} = Digital.run(Digital.block(String.to_atom(name), []), Enum.map(ins, &hex/1))
        assert Enum.map(got, &enc/1) == outs, name
      end
    end

    test "check mode of the async blocks accepts what append made" do
      for name <- [:crc32_async_bb, :crc16_async_bb] do
        {with_crc, _} = Digital.run(Digital.block(name, []), ["abc", "defghij"])
        n = if name == :crc32_async_bb, do: 4, else: 2
        bad = "xyz12345"
        {res, _} = Digital.run(Digital.block(name, check: true), with_crc ++ [bad])
        assert res.ok == ["abc", "defghij"]
        assert res.fail == [binary_part(bad, 0, byte_size(bad) - n)]
      end
    end
  end

  describe "access code" do
    test "correlate_access_code_bb and _tag_bb equal GNU Radio", %{gr: gr} do
      for %{
            "access_code" => ac,
            "threshold" => thr,
            "in" => input,
            "bb" => bb,
            "tag_out" => tag_out,
            "tags" => tags
          } <-
            gr["correlate"] do
        input = bits(input)

        {out, _} =
          Digital.run(
            Digital.block(:correlate_access_code_bb, access_code: ac, threshold: thr),
            input
          )

        assert unbits(out) == bb, "bb #{ac} #{thr}"

        {{pass, got_tags}, _} =
          Digital.run(
            Digital.block(:correlate_access_code_tag_bb, access_code: ac, threshold: thr),
            input
          )

        assert unbits(pass) == tag_out
        assert Enum.map(got_tags, &Tuple.to_list/1) == tags, "tags #{ac} #{thr}"
      end
    end

    property "stream in pieces gives the same output" do
      check all(
              bits <- list_of(integer(0..1), min_length: 100, max_length: 400),
              cut <- integer(0..400),
              thr <- integer(0..3)
            ) do
        input = :binary.list_to_bin(bits)
        cut = min(cut, byte_size(input))
        <<a::binary-size(cut), b::binary>> = input
        block = Digital.block(:correlate_access_code_bb, access_code: "10110100", threshold: thr)
        {whole, _} = Digital.run(block, input)
        {o1, st} = Digital.run(block, a)
        {o2, _} = Digital.run(block, b, st)
        assert o1 <> o2 == whole
      end
    end
  end

  describe "packet headers" do
    test "protocol_formatter_bb with header_format_default and _counter equals GNU Radio", %{
      gr: gr
    } do
      for %{
            "kind" => kind,
            "access_code" => ac,
            "threshold" => thr,
            "bps" => bps,
            "payload_lengths" => lens,
            "headers" => headers
          } <- gr["formatter"] do
        block =
          Digital.block(:protocol_formatter_bb,
            kind: String.to_atom(kind),
            access_code: ac,
            threshold: thr,
            bps: bps
          )

        {got, _} = Digital.run(block, Enum.map(lens, &:binary.copy(<<7>>, &1)))
        assert Enum.map(got, &enc/1) == headers, "#{kind} #{ac} bps #{bps}"
      end
    end

    test "the length of a default header can be read back" do
      f = Framing.header_format(:default, "1010101011110000", 2)
      {<<_ac::16, rest::binary>>, _} = Framing.header(f, 1234)
      assert Framing.parse_length(rest) == {:ok, 1234}
      assert Framing.parse_length(<<0, 5, 0, 6>>) == :error
    end
  end

  describe "HDLC" do
    test "hdlc_framer_pb equals GNU Radio", %{gr: gr} do
      %{"payloads" => payloads, "frames" => frames} = gr["hdlc"]["framer"]
      {got, _} = Digital.run(Digital.block(:hdlc_framer_pb, []), Enum.map(payloads, &hex/1))
      assert Enum.map(got, &unbits/1) == frames
    end

    test "hdlc_deframer_bp equals GNU Radio on streams with noise, errors, aborts and shared flags",
         %{gr: gr} do
      for %{
            "stream" => name,
            "bits" => stream,
            "length_min" => lmin,
            "length_max" => lmax,
            "frames" => frames
          } <-
            gr["hdlc"]["deframer"] do
        block = Digital.block(:hdlc_deframer_bp, length_min: lmin, length_max: lmax)
        # GNU Radio's scheduler needs trailing input to flush; give both the same
        {got, _} = Digital.run(block, bits(stream) <> :binary.copy(<<0>>, 4000))
        assert Enum.map(got, &enc/1) == frames, "#{name} min #{lmin} max #{lmax}"
      end
    end

    property "frame then deframe returns the payloads, also in pieces" do
      check all(
              payloads <-
                list_of(binary(min_length: 4, max_length: 60), min_length: 1, max_length: 5),
              cut <- integer(0..2000)
            ) do
        stream = payloads |> Enum.map(&Framing.hdlc_frame/1) |> IO.iodata_to_binary()
        cut = min(cut, byte_size(stream))
        <<a::binary-size(cut), b::binary>> = stream
        block = Digital.block(:hdlc_deframer_bp, length_min: 1, length_max: 500)
        {p1, st} = Digital.run(block, a)
        {p2, _} = Digital.run(block, b, st)
        assert p1 ++ p2 == payloads
      end
    end
  end
end
