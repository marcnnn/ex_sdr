defmodule ExSdr.Comm.ConvolutionalTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias ExSdr.Comm.Convolutional, as: CC

  # Codes of several standards: K, generators, puncturing
  @codes [
    # DVB-T / 802.11a/g, rate 3/4
    {7, [0o171, 0o133], [[1, 0, 1], [1, 1, 0]]},
    # 802.11 rate 2/3
    {7, [0o133, 0o171], [[1, 1], [1, 0]]},
    # GSM TCH/FS K = 5
    {5, [0o23, 0o33], nil},
    # CDMA IS-95 forward link K = 9, rate 1/2
    {9, [0o753, 0o561], nil},
    # rate 1/3 LTE-style K = 7 (tail-biting not modelled)
    {7, [0o133, 0o171, 0o165], nil},
    # rate 1/4, K = 3
    {3, [0o7, 0o5, 0o7, 0o6], nil}
  ]

  test "rate" do
    assert CC.rate(
             CC.new(
               constraint_length: 7,
               polynomials: [0o171, 0o133],
               puncture: [[1, 0, 1], [1, 1, 0]]
             )
           ) ==
             {3, 4}
  end

  for {k, polys, punct} <- @codes do
    property "K=#{k} #{inspect(polys, base: :octal)} #{inspect(punct)}: native, SIMD and reference agree; clean input decodes" do
      code =
        CC.new(
          constraint_length: unquote(k),
          polynomials: unquote(polys),
          puncture: unquote(punct)
        )

      check all msg <- list_of(integer(0..1), min_length: 1, max_length: 300),
                errors <- list_of(integer(0..10_000), max_length: 12),
                start <- member_of([:zero, :unknown]),
                max_runs: 25 do
        coded = CC.encode(code, msg)
        # an incomplete last puncturing period decodes to a whole period
        if start == :zero,
          do: assert(code |> CC.decode_reference(coded) |> Enum.take(length(msg)) == msg)

        noisy =
          errors
          |> Enum.map(&rem(&1, length(coded)))
          |> Enum.reduce(coded, fn p, acc -> List.update_at(acc, p, &(1 - &1)) end)

        ref = CC.decode_reference(code, noisy, start: start)
        {period, _} = CC.rate(code)
        assert length(ref) == div(length(msg) + period - 1, period) * period

        if ExSdr.SIMD.available?() do
          bin = :binary.list_to_bin(noisy)

          assert CC.decode_binary(code, bin, start: start, simd: false) ==
                   :binary.list_to_bin(ref)

          assert CC.decode_binary(code, bin, start: start, simd: true) == :binary.list_to_bin(ref)
        end
      end
    end

    property "K=#{k} #{inspect(polys, base: :octal)} #{inspect(punct)}: streaming decoder equals the windowed reference for any split" do
      code =
        CC.new(
          constraint_length: unquote(k),
          polynomials: unquote(polys),
          puncture: unquote(punct)
        )

      check all msg <- list_of(integer(0..1), min_length: 1, max_length: 400),
                errors <- list_of(integer(0..10_000), max_length: 20),
                start <- member_of([:zero, :unknown]),
                depth <- integer(1..48),
                cuts <- list_of(integer(0..200), max_length: 8),
                simd <- boolean(),
                max_runs: 25 do
        coded = CC.encode(code, msg)

        noisy =
          errors
          |> Enum.map(&rem(&1, length(coded)))
          |> Enum.reduce(coded, fn p, acc -> List.update_at(acc, p, &(1 - &1)) end)

        ref = CC.decode_reference(code, noisy, start: start, depth: depth)
        # the streaming decoder is native only
        if ExSdr.SIMD.available?(), do: assert_stream(code, noisy, start, depth, cuts, simd, ref)
      end
    end
  end

  defp assert_stream(code, noisy, start, depth, cuts, simd, ref) do
    dec = CC.decoder(code, start: start, depth: depth, simd: simd)
    assert stream(dec, :binary.list_to_bin(noisy), cuts) == :binary.list_to_bin(ref)

    # a window longer than the input: one traceback, as the block decoder
    long = CC.decoder(code, start: start, depth: 10_000, simd: simd)

    assert stream(long, :binary.list_to_bin(noisy), cuts) ==
             :binary.list_to_bin(CC.decode_reference(code, noisy, start: start))
  end

  # feeds `bits` in pieces of the given sizes (then the rest) and flushes
  defp stream(dec, bits, cuts) do
    {pieces, rest} =
      Enum.reduce(cuts, {[], bits}, fn c, {acc, rest} ->
        c = min(c, byte_size(rest))
        <<piece::binary-size(c), rest::binary>> = rest
        {[piece | acc], rest}
      end)

    {out, dec} =
      Enum.reduce(Enum.reverse([rest | pieces]), {[], dec}, fn piece, {out, dec} ->
        {bits, dec} = CC.decode_stream(dec, piece)
        {[out, bits], dec}
      end)

    {tail, _} = CC.flush(dec)
    IO.iodata_to_binary([out, tail])
  end
end
