defmodule ExSdr.Comm.NifDispatchTest do
  # The regular-scheduler `_small` variants run the same code as the dirty
  # ones, and the callers pick by size without changing results.
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias ExSdr.Comm.{Bits, Convolutional, ConvInterleaver, ReedSolomon, Scrambler}
  alias ExSdr.NIF

  @moduletag :native

  setup_all do
    if ExSdr.SIMD.available?(), do: :ok, else: {:skip, "NIF not loaded"}
  end

  defp bits(max), do: StreamData.map(StreamData.binary(max_length: max), &to_bits/1)
  defp to_bits(bin), do: for(<<b::1 <- bin>>, into: <<>>, do: <<b>>)

  property "bits_pack: regular and dirty agree with the Elixir reference" do
    check all(bits <- bits(40)) do
      ref = Bits.pack(bits, native: false)
      assert NIF.bits_pack(bits) == ref
      assert NIF.bits_pack_small(bits) == ref
      assert Bits.pack(bits, native: true) == ref
    end
  end

  property "bits_pack: arbitrary byte values OR-ed like the reference" do
    check all(raw <- StreamData.binary(max_length: 64)) do
      assert NIF.bits_pack_small(raw) == Bits.pack(raw, native: false)
    end
  end

  test "bits_pack drops a trailing partial byte" do
    assert Bits.pack(<<1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1>>) == <<0b10100001>>
    assert Bits.pack(<<1, 1, 1>>) == <<>>
  end

  test "bits_pack above the regular threshold" do
    bits = to_bits(:crypto.strong_rand_bytes(300_000))
    assert byte_size(bits) > 2_000_000
    assert Bits.pack(bits, native: true) == Bits.pack(bits, native: false)
  end

  property "byte_affine: regular equals dirty" do
    check all(
            data <- StreamData.binary(max_length: 200),
            keep <- StreamData.binary(min_length: 1, max_length: 20),
            phase <- StreamData.integer(0..40)
          ) do
      mask = :binary.copy(<<0x5A>>, byte_size(keep))

      assert NIF.byte_affine_small(data, keep, mask, phase) ==
               NIF.byte_affine(data, keep, mask, phase)
    end
  end

  property "conv_interleave_stream: regular equals dirty" do
    check all(
            data <- StreamData.binary(max_length: 400),
            i <- StreamData.integer(1..6),
            m <- StreamData.integer(0..5),
            inverse <- StreamData.boolean(),
            phase <- StreamData.integer(0..30)
          ) do
      history = :crypto.strong_rand_bytes(i * (i - 1) * m)

      assert NIF.conv_interleave_stream_small(history, data, i, m, inverse, phase) ==
               NIF.conv_interleave_stream(history, data, i, m, inverse, phase)
    end
  end

  test "rs_decode: regular equals dirty on clean, corrupted and garbage codewords" do
    code = ExSdr.DVBT.ReedSolomon.code()
    args = [code.field.poly, code.n, code.k, code.fcr, code.prim, code.shortening]

    clean =
      for _ <- 1..20, into: <<>> do
        ExSdr.DVBT.ReedSolomon.encode(:crypto.strong_rand_bytes(188))
      end

    corrupt =
      for <<cw::binary-204 <- clean>>, reduce: <<>> do
        acc ->
          pos = Enum.take_random(0..203, :rand.uniform(14) - 1)

          bad =
            Enum.reduce(pos, cw, fn p, c ->
              <<a::binary-size(p), b, r::binary>> = c
              <<a::binary, Bitwise.bxor(b, 1 + :rand.uniform(254))::8, r::binary>>
            end)

          acc <> bad
      end

    for data <- [clean, corrupt, :crypto.strong_rand_bytes(204 * 8)] do
      assert apply(NIF, :rs_decode_small, [data | args]) == apply(NIF, :rs_decode, [data | args])
    end

    assert {_, 0, 0} = apply(NIF, :rs_decode_small, [clean | args])
    assert {_, c, _} = apply(NIF, :rs_decode_small, [corrupt | args])
    assert c > 0
  end

  test "rs_decode: syndrome tables are per code (alternating codes stay correct)" do
    a = ReedSolomon.new(m: 8, field_poly: 0x11D, n: 255, k: 239, fcr: 0, prim: 1, shortening: 51)

    b =
      ReedSolomon.new(m: 8, field_poly: 0x11D, n: 255, k: 223, fcr: 112, prim: 11, shortening: 0)

    for code <- [a, b, a, b], native <- [true, false] do
      msg = :crypto.strong_rand_bytes(ReedSolomon.message_length(code))
      cw = ReedSolomon.encode_bytes(code, msg)
      <<x, rest::binary>> = cw
      bad = <<Bitwise.bxor(x, 7), rest::binary>>
      assert ReedSolomon.decode_all_bytes(code, bad, native: native) == {msg, 1, 0}
    end
  end

  property "viterbi_stream: regular equals dirty, state included" do
    code = Convolutional.new(constraint_length: 7, polynomials: [0o171, 0o133])

    check all(
            steps <- StreamData.list_of(StreamData.integer(0..300), min_length: 1, max_length: 4),
            flush <- StreamData.boolean()
          ) do
      st = NIF.viterbi_stream_new(7, code.polynomials, false, true, 32)

      last = length(steps) - 1

      steps
      |> Enum.with_index()
      |> Enum.reduce({st, st}, fn {n, idx}, {a, b} ->
        flush = flush and idx == last
        rx = for _ <- 1..(2 * n)//1, into: <<>>, do: <<:rand.uniform(3) - 1>>
        {oa, sa} = NIF.viterbi_stream(a, rx, 7, code.polynomials, flush)
        {ob, sb} = NIF.viterbi_stream_small(b, rx, 7, code.polynomials, flush)
        assert {oa, sa} == {ob, sb}
        {sa, sb}
      end)
    end
  end

  test "Convolutional streaming gives the same bits on both sides of the size threshold" do
    code = Convolutional.new(constraint_length: 7, polynomials: [0o171, 0o133])
    rx = for _ <- 1..(2 * 20_000), into: <<>>, do: <<:rand.uniform(2) - 1>>

    run = fn pieces ->
      {outs, dec} =
        Enum.map_reduce(pieces, Convolutional.decoder(code), fn p, d ->
          Convolutional.decode_stream(d, p)
        end)

      {tail, _} = Convolutional.flush(dec)
      IO.iodata_to_binary(outs ++ [tail])
    end

    big = run.([rx])
    small = run.(for <<p::binary-size(2000) <- rx>>, do: p)
    assert big == small
  end

  test "Scrambler and ConvInterleaver agree across the size threshold" do
    s = Scrambler.new(:crypto.strong_rand_bytes(1503), :crypto.strong_rand_bytes(1503))
    data = :crypto.strong_rand_bytes(1_100_000)
    pieces = for <<p::binary-size(1000) <- data>>, do: p

    {outs, _} =
      Enum.map_reduce(pieces, 0, fn p, pos ->
        {Scrambler.apply_at(s, p, pos, native: true), pos + byte_size(p)}
      end)

    assert byte_size(data) > 1_048_576
    assert Scrambler.apply_at(s, data, 0, native: true) == IO.iodata_to_binary(outs)

    c = ConvInterleaver.new(branches: 12, depth: 17)
    data = :crypto.strong_rand_bytes(250_000)
    {big, _} = ConvInterleaver.stream(c, :deinterleave, data, ConvInterleaver.stream_state(c))
    pieces = for <<p::binary-size(1000) <- data>>, do: p

    {outs, _} =
      Enum.map_reduce(pieces, ConvInterleaver.stream_state(c), fn p, st ->
        ConvInterleaver.stream(c, :deinterleave, p, st)
      end)

    assert big == IO.iodata_to_binary(outs)
  end
end
