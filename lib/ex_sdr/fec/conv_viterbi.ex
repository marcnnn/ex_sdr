defmodule ExSdr.FEC.ConvViterbi do
  @moduledoc """
  Soft-decision Viterbi decoding of rate 1/n convolutional codes in the
  integer format of GNU Radio's `cc_decoder` (volk `volk_8u_x4_conv_k7_r2_8u`),
  generalised to any constraint length and number of outputs.

  The trellis is the one of `ExSdr.Comm.Convolutional` (its output table
  gives the code bits of every register value); this module only adds the
  state numbering of GNU Radio and the soft branch metrics.

  Conventions (GNU Radio's):

    * polynomials are bit masks over the K-bit register `reg = state * 2 + input`,
      bit 0 is the newest input, bit K - 1 the oldest. A negative polynomial
      inverts that output. GNU Radio's `[79, 109]` for K = 7 are the usual
      `0o171, 0o133` with the bit order reversed.
    * state `s` (K - 1 bits) holds the last inputs, bit 0 newest. Predecessors
      of state `s` are `s >>> 1` (choice 0) and `(s >>> 1) + 2^(K-2)` (choice 1).
    * soft input: one `u8` per code bit, 0 is a sure 0, 255 a sure 1,
      128 is "no information" (a punctured bit).

  Integer format (what hardware would hold):

    * path metrics are `u8` and saturate at 255 (the SIMD kernels of volk use
      saturating adds; its portable C kernel wraps, the two differ only when
      the spread between states exceeds 255, which clean input does reach);
      after every step the minimum is subtracted.
    * the branch metric of a transition is
      `(1 + sum_j (expected_j xor symbol_j)) >>> 3` where `expected_j` is 0 or
      255; this is a distance, 0 for a perfect match. The largest metric is
      `max = (n * 127) >>> 2` (63 for n = 2).
    * when the transition's expected bits are the complement of the butterfly's
      reference branch (the case when the first and last tap of every
      polynomial are set, as for all usual codes), its metric is `max - metric`
      like in GNU Radio (this differs from recomputing the distance by up to
      one count); otherwise the distance is computed directly.
    * the survivor is the smaller metric; on a tie the predecessor with
      choice 1 (the larger state number) wins, as in GNU Radio.
    * one decision bit per state and step is stored for the traceback
      (`2^(K-1)` bits per step, the block RAM of a hardware decoder). The
      decision at step `t` yields the input bit `t - (K - 1)`.
    * start: known start state gets metric 0, all others `bias` (63); without
      a known state all metrics are equal.
  """
  import Bitwise

  @doc """
  Tables for constant `k` and `polys` (GNU Radio polynomials, may be
  negative). Returns a map; cached in `:persistent_term`.
  """
  def tables(k, polys, opts \\ []) do
    key = {__MODULE__, k, polys, Keyword.get(opts, :metrics, :gr)}

    case :persistent_term.get(key, nil) do
      nil ->
        t = build(k, polys, Keyword.get(opts, :metrics, :gr))
        :persistent_term.put(key, t)
        t

      t ->
        t
    end
  end

  defp build(k, polys, metrics) do
    n = length(polys)
    states = 1 <<< (k - 1)
    half = states >>> 1
    mask = (1 <<< k) - 1

    # the Comm trellis wants standard polynomials: newest input as the most
    # significant bit, so reverse the K bits; inversion is applied here
    comm =
      ExSdr.Comm.Convolutional.new(
        constraint_length: k,
        polynomials: Enum.map(polys, &reverse_bits(abs(&1) &&& mask, k))
      )

    invert = Enum.map(polys, &(&1 < 0))

    # expected code bits (0 / 255) of the register value, GNU Radio bit order
    expected = fn reg ->
      comm_reg = reverse_bits(reg, k)

      comm.outputs
      |> elem(comm_reg)
      |> Tuple.to_list()
      |> Enum.zip(invert)
      |> Enum.map(fn {bit, inv} -> if bit == 1 != inv, do: 255, else: 0 end)
    end

    complement = fn e -> Enum.map(e, &(255 - &1)) end

    # per new state s and choice c: how its metric follows from the
    # butterfly's reference branch (predecessor i, input 0 -> state 2i)
    kinds =
      for s <- 0..(states - 1) do
        i = s >>> 1
        ref = expected.(2 * i)

        for c <- 0..1 do
          e = expected.((i + c * half) * 2 + (s &&& 1))

          cond do
            metrics == :direct -> {:direct, e}
            e == ref -> :ref
            e == complement.(ref) -> :comp
            true -> {:direct, e}
          end
        end
        |> List.to_tuple()
      end
      |> List.to_tuple()

    refs = for i <- 0..(half - 1), do: expected.(2 * i)

    %{
      k: k,
      n: n,
      states: states,
      half: half,
      kinds: kinds,
      refs: List.to_tuple(refs),
      max: n * 127 >>> 2
    }
  end

  def reverse_bits(v, width) do
    Enum.reduce(0..(width - 1), 0, fn i, acc -> acc ||| (v >>> i &&& 1) <<< (width - 1 - i) end)
  end

  @doc "Distance metric of expected 0/255 bits against symbols (GNU Radio's branch metric)."
  def distance(expected, syms) do
    sum = 1 + (Enum.zip_reduce(expected, syms, 0, fn e, s, acc -> acc + bxor(e, s) end))
    sum >>> 3
  end

  @doc """
  Initial metrics: `{:state, s}` biases a known start state, `:unbiased`
  gives all states the same metric.
  """
  def init_metrics(t, {:state, s}, bias),
    do: for(i <- 0..(t.states - 1), do: if(i == s, do: 0, else: bias)) |> List.to_tuple()

  def init_metrics(t, :unbiased, bias), do: List.duplicate(bias, t.states) |> List.to_tuple()

  @doc """
  Runs the add-compare-select recursion over `syms` (a list of `u8`, `n` per
  step) from `metrics`. Returns `{decisions, metrics}`: `decisions` is a
  tuple with one integer per step (bit `s` is the decision of state `s`:
  1 if the survivor came from predecessor choice 1).
  """
  def run(t, syms, metrics) do
    {decs, metrics} =
      syms
      |> Enum.chunk_every(t.n)
      |> Enum.reduce({[], metrics}, fn step_syms, {decs, metrics} ->
        {dec, metrics} = step(t, step_syms, metrics)
        {[dec | decs], metrics}
      end)

    {decs |> Enum.reverse() |> List.to_tuple(), metrics}
  end

  defp step(t, syms, x) do
    states = t.states
    half = t.half

    mref =
      for r <- Tuple.to_list(t.refs), do: distance(r, syms)

    mref = List.to_tuple(mref)

    {y, dec} =
      Enum.reduce((states - 1)..0//-1, {[], 0}, fn s, {acc, dec} ->
        i = s >>> 1
        {k0, k1} = elem(t.kinds, s)
        m0 = min(elem(x, i) + bm(t, k0, elem(mref, i), syms), 255)
        m1 = min(elem(x, i + half) + bm(t, k1, elem(mref, i), syms), 255)

        if m0 >= m1,
          do: {[m1 | acc], dec ||| 1 <<< s},
          else: {[m0 | acc], dec}
      end)

    min = Enum.min(y)
    {dec, y |> Enum.map(&(&1 - min)) |> List.to_tuple()}
  end

  defp bm(_t, :ref, m, _syms), do: m
  defp bm(t, :comp, m, _syms), do: t.max - m
  defp bm(_t, {:direct, e}, _m, syms), do: distance(e, syms)

  @doc "The state with the smallest metric (the first on ties)."
  def best_state(metrics) do
    {_, s} = metrics |> Tuple.to_list() |> Enum.with_index() |> Enum.min()
    s
  end

  @doc """
  Traceback from `state` after step `last` (the decisions of steps
  `last, last - 1, ..., first`): returns `{bits, state}` where `bits` has the
  decoded bit of every step (`bit_t` is the input `t - (K - 1)`), the first
  entry belonging to step `first`, and `state` is the state before step `first`.
  """
  def traceback(t, decisions, state, last, first) do
    top = t.k - 2

    Enum.reduce(last..first//-1, {[], state}, fn step, {bits, s} ->
      d = elem(decisions, step) >>> s &&& 1
      {[d | bits], (s >>> 1) ||| d <<< top}
    end)
  end
end
