defmodule ExSdr.FEC.LDPCWidthTest do
  @moduledoc """
  `NxFixed.analyze/3` on the integer instances of the LDPC blocks
  (`block.fun`, the unrolled dataflow): no value can leave its type, so
  every backend and the hardware agree bit for bit.
  """
  use ExUnit.Case, async: true

  alias ExSdr.FEC.LDPC
  alias NxFixed.{Fixed, Num}

  @alist_dir Path.expand("../fixtures/gnuradio/fec/ldpc_alist", __DIR__)

  defp code_100, do: LDPC.h_matrix(Path.join(@alist_dir, "n_0100_k_0042_gap_02.alist"), 2)

  # Two iterations are analysed: every value carried from one iteration to
  # the next (posteriors, messages, decisions) is saturated to its declared
  # range at the end of an iteration, so the ranges entering iteration 2
  # are the ranges entering every later one.
  defp template(%Fixed{t: t, frac: f}), do: %Fixed{t: Nx.to_template(t), frac: f}
  defp template(t), do: Nx.to_template(t)

  defp width_check(b, item) do
    NxFixed.analyze(b.fun, [template(item), {}, b.opts])
  end

  test "integer min-sum decoders" do
    code = code_100()

    combos =
      for(
        s <- [:layered, :flooding],
        c <- [[], [scale: 0.75], [offset: 0.5]],
        do: {Num.q(6, 1), s, c}
      ) ++
        [{Num.q(5, 0), :layered, [scale: 0.75]}, {Num.q(8, 2), :flooding, [offset: 1.0]}]

    for {format, schedule, correction} <- combos do
      b =
        LDPC.block(
          :ldpc_min_sum_decoder,
          [code: code, max_iter: 2, schedule: schedule, format: format] ++ correction
        )

      item = LDPC.quantize_llr(Nx.broadcast(0.0, {2 * code.n}), format)

      assert {:ok, _} = width_check(b, item),
             "#{inspect(format)} #{schedule} #{inspect(correction)}"
    end
  end

  test "state bits are honoured" do
    b =
      LDPC.block(:ldpc_min_sum_decoder,
        code: code_100(),
        max_iter: 2,
        format: Num.q(6, 1),
        state_bits: 16
      )

    assert b.opts[:state_type] == :s16
    item = LDPC.quantize_llr(Nx.broadcast(0.0, {200}), Num.q(6, 1))
    assert {:ok, _} = width_check(b, item)
  end

  test "bit flip decoder and encoders" do
    code = code_100()
    b = LDPC.block(:ldpc_bit_flip_decoder, code: code, max_iter: 2)
    assert {:ok, _} = width_check(b, Nx.broadcast(Nx.tensor(0, type: :s8), {200}))

    for name <- [:ldpc_par_mtrx_encoder, :ldpc_encoder] do
      b = LDPC.block(name, code: code)
      assert {:ok, _} = width_check(b, Nx.broadcast(Nx.tensor(0, type: :u8), {2 * code.k}))
    end
  end
end
