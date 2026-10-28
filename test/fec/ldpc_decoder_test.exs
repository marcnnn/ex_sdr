defmodule ExSdr.FEC.LDPCDecoderTest do
  @moduledoc """
  The soft and bit flip decoders beyond the GNU Radio comparison:
  the dataflow form (`block.fun`, unrolled) equal to the compiled loop,
  frames independent of batching, fixed point against float, and a seeded BER/FER test
  against the float sum-product reference.
  """
  use ExUnit.Case, async: true
  @moduletag :ldpc_alist

  alias ExSdr.FEC.LDPC
  alias ExSdr.FEC.LDPC.Kernels
  alias NxFixed.{Fixed, Num}

  @alist_dir Path.expand("../fixtures/gnuradio/fec/ldpc_alist", __DIR__)

  defp code_100, do: LDPC.h_matrix(Path.join(@alist_dir, "n_0100_k_0042_gap_02.alist"), 2)
  defp code_300, do: LDPC.h_matrix(Path.join(@alist_dir, "n_0300_k_0152_gap_03.alist"), 3)

  # BPSK (bit 1 -> +1) in AWGN at Eb/N0 (dB): LLRs 2y/sigma^2 (positive for 1),
  # with the information bits. Noise from :rand (exsss, seeded).
  defp channel(code, frames, ebn0_db, seed) do
    :rand.seed(:exsss, seed)
    bits = for _ <- 1..(frames * code.k), do: :rand.uniform(2) - 1
    enc = LDPC.block(:ldpc_par_mtrx_encoder, code: code)
    {cw, _} = LDPC.run(enc, Nx.tensor(bits, type: :u8))
    sigma = :math.sqrt(1 / (2 * code.k / code.n * :math.pow(10, ebn0_db / 10)))

    llr =
      for c <- Nx.to_flat_list(cw) do
        y = 2 * c - 1 + sigma * :rand.normal()
        2 * y / (sigma * sigma)
      end

    %{bits: bits, codewords: cw, llr: Nx.tensor(llr, type: :f32), frames: frames}
  end

  defp frame_errors(out, bits, k) do
    Enum.zip(Nx.to_flat_list(out), bits)
    |> Enum.chunk_every(k)
    |> Enum.count(fn f -> Enum.any?(f, fn {a, b} -> a != b end) end)
  end

  defp bit_errors(out, bits),
    do: Enum.zip(Nx.to_flat_list(out), bits) |> Enum.count(fn {a, b} -> a != b end)

  defp unrolled(b, input) do
    compiler = if Code.ensure_loaded?(EXLA), do: EXLA, else: Nx.Defn.Evaluator
    # index tables as arguments: same values as the constants, faster XLA compile
    f =
      Nx.Defn.jit(fn x, t -> b.fun.(x, {}, Keyword.put(b.opts, :tables, t)) |> elem(0) end,
        compiler: compiler
      )

    f.(input, Kernels.tables(b.opts)) |> Nx.backend_transfer(Nx.BinaryBackend)
  end

  defp input(llr, nil), do: llr
  defp input(llr, format), do: LDPC.quantize_llr(llr, format)

  @q6 Num.q(6, 1)

  defp decoders(code, max_iter) do
    [
      {"bit flip", LDPC.block(:ldpc_bit_flip_decoder, code: code, max_iter: max_iter), nil},
      {"sum-product flooding", LDPC.block(:ldpc_bp_decoder, code: code, max_iter: max_iter), nil},
      {"sum-product layered",
       LDPC.block(:ldpc_bp_decoder, code: code, max_iter: max_iter, schedule: :layered), nil},
      {"normalised min-sum layered, float",
       LDPC.block(:ldpc_min_sum_decoder, code: code, max_iter: max_iter, scale: 0.75), nil},
      {"min-sum flooding, q6",
       LDPC.block(:ldpc_min_sum_decoder,
         code: code,
         max_iter: max_iter,
         schedule: :flooding,
         format: @q6
       ), @q6},
      {"normalised min-sum layered, q6",
       LDPC.block(:ldpc_min_sum_decoder,
         code: code,
         max_iter: max_iter,
         scale: 0.75,
         format: @q6
       ), @q6},
      {"offset min-sum layered, q6",
       LDPC.block(:ldpc_min_sum_decoder,
         code: code,
         max_iter: max_iter,
         offset: 0.5,
         format: @q6
       ), @q6}
    ]
  end

  test "the unrolled dataflow (block.fun) equals the loop with early exit" do
    code = code_100()
    ch = channel(code, 24, 2.5, 1)

    for {name, b, format} <- decoders(code, 6) do
      x = input(ch.llr, format)
      {loop, _} = LDPC.run(b, x)
      assert unrolled(b, x) == loop, name
    end
  end

  test "frames decode independently of how they are batched" do
    code = code_100()
    ch = channel(code, 20, 2.0, 2)

    for {name, b, format} <- decoders(code, 20) do
      x = input(ch.llr, format)
      {whole, _} = LDPC.run(b, x)

      parts =
        for {start, len} <- [{0, 10}, {10, 10}] do
          part =
            case x do
              %Fixed{t: t} = f -> %{f | t: Nx.slice(t, [start * code.n], [len * code.n])}
              t -> Nx.slice(t, [start * code.n], [len * code.n])
            end

          {out, _} = LDPC.run(b, part)
          Nx.to_flat_list(out)
        end

      assert Enum.concat(parts) == Nx.to_flat_list(whole), name
    end
  end

  test "error-free input decodes to the information bits; output: :codeword gives the codeword" do
    code = code_100()
    ch = channel(code, 10, 30.0, 3)

    for {name, b, format} <- decoders(code, 5) do
      {out, _} = LDPC.run(b, input(ch.llr, format))
      assert Nx.to_flat_list(out) == ch.bits, name
    end

    b = LDPC.block(:ldpc_min_sum_decoder, code: code, output: :codeword, format: @q6)
    assert b.rate == {100, 100}
    {out, _} = LDPC.run(b, LDPC.quantize_llr(ch.llr, @q6))
    assert out == ch.codewords
  end

  test "fixed point (6-bit messages, 8-bit posteriors) against float min-sum" do
    # Same algorithm, same scale 0.75 (exact in 1/16). Differences come from
    # quantising the channel LLRs to 0.5 steps and saturating at 15.5 (31.5
    # for posteriors). At 2.5 dB most frames converge to the same codeword in
    # both; the frames that differ are near the decoding threshold.
    code = code_300()
    ch = channel(code, 300, 2.5, 4)
    float = LDPC.block(:ldpc_min_sum_decoder, code: code, scale: 0.75)
    fixed = LDPC.block(:ldpc_min_sum_decoder, code: code, scale: 0.75, format: @q6)
    {a, _} = LDPC.run(float, ch.llr)
    {b, _} = LDPC.run(fixed, LDPC.quantize_llr(ch.llr, @q6))

    same =
      Enum.zip(
        Nx.to_flat_list(a) |> Enum.chunk_every(code.k),
        Nx.to_flat_list(b) |> Enum.chunk_every(code.k)
      )
      |> Enum.count(fn {x, y} -> x == y end)

    assert same >= 0.97 * 300
    fe_float = frame_errors(a, ch.bits, code.k)
    fe_fixed = frame_errors(b, ch.bits, code.k)
    # 0.1 dB at this code's slope (FER x12 per dB around 2 dB: x1.3), plus
    # two standard deviations of a Poisson count of about 10.
    assert fe_fixed <= 1.3 * fe_float + 6
  end

  @tag timeout: 300_000
  test "BER/FER at 1.5 and 2.5 dB against the float sum-product reference" do
    # n = 300, k = 152 (gr-fec's n_0300_k_0152_gap_03), 400 frames per point,
    # 50 iterations. The reference is gr-fec's algorithm (flooding sum-product,
    # f64). Measured with these seeds (FER): sum-product 136 and 11 of 400,
    # normalised min-sum 151 and 10, 6-bit 147 and 10, plain min-sum 174 and
    # 15. Bounds: the FER of this code falls about 12x per dB here, so
    # 0.2 dB is a factor 1.65 and 0.4 dB a factor 2.7; plus 8 frames (about
    # two standard deviations of a Poisson count of 10-15).
    code = code_300()
    frames = 400
    bp = LDPC.block(:ldpc_bp_decoder, code: code, max_iter: 50)

    within = [
      {"sum-product layered",
       LDPC.block(:ldpc_bp_decoder, code: code, max_iter: 50, schedule: :layered), nil, 1.65},
      {"normalised min-sum, float",
       LDPC.block(:ldpc_min_sum_decoder, code: code, max_iter: 50, scale: 0.75), nil, 1.65},
      {"normalised min-sum, q6",
       LDPC.block(:ldpc_min_sum_decoder, code: code, max_iter: 50, scale: 0.75, format: @q6), @q6,
       1.65},
      {"offset min-sum, q6",
       LDPC.block(:ldpc_min_sum_decoder, code: code, max_iter: 50, offset: 0.5, format: @q6), @q6,
       1.65},
      {"min-sum, float", LDPC.block(:ldpc_min_sum_decoder, code: code, max_iter: 50), nil, 2.7}
    ]

    results =
      for {ebn0, seed} <- [{1.5, 11}, {2.5, 12}] do
        ch = channel(code, frames, ebn0, seed)
        uncoded = bit_errors(Nx.greater(ch.llr, 0), Nx.to_flat_list(ch.codewords))
        {ref, _} = LDPC.run(bp, ch.llr)
        fe_ref = frame_errors(ref, ch.bits, code.k)
        be_ref = bit_errors(ref, ch.bits)

        # coding gain: fewer bit errors than uncoded transmission
        assert be_ref / (frames * code.k) < 0.5 * uncoded / (frames * code.n)

        for {name, b, format, factor} <- within do
          {out, _} = LDPC.run(b, input(ch.llr, format))
          fe = frame_errors(out, ch.bits, code.k)
          assert fe <= factor * fe_ref + 8, "#{name} at #{ebn0} dB: #{fe} vs #{fe_ref}"
          assert bit_errors(out, ch.bits) < uncoded * code.k / code.n
        end

        {ebn0, fe_ref}
      end

    [{_, fe_low}, {_, fe_high}] = results
    # waterfall: the reference improves by a large factor over 1 dB
    assert fe_high * 4 < fe_low
    assert fe_high <= 0.06 * frames
  end
end
