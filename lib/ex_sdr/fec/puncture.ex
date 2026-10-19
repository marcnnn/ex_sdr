defmodule ExSdr.FEC.Puncture do
  @moduledoc """
  gr-fec's `puncture_bb`, `puncture_ff` and `depuncture_bb`: drop (or put
  back) the items of a repeating pattern.

  `puncsize` items form a period, `puncpat` is an integer whose
  `puncsize` low bits say which items of a period are kept, the most
  significant of them for the first item (`0b110` keeps the first two of
  three). `delay` rotates the pattern right by that many places before it is
  used, as GNU Radio does (the rotation is done on the unmasked integer, so
  bits of `puncpat` above `puncsize` take part in it, just like there).

  `puncture` keeps the marked items, `depuncture` puts `symbol` (default 127,
  the middle of the 8 bit soft range) in place of the dropped ones. Both work
  on whole periods; input is a tensor of any type.
  """
  import Bitwise

  @doc """
  The effective pattern as a list of 0/1, first item of the period first.
  """
  def pattern(puncsize, puncpat, delay \\ 0) do
    mask = (1 <<< puncsize) - 1

    rotated =
      Enum.reduce(1..delay//1, puncpat, fn _, p ->
        ((p &&& 1) <<< (puncsize - 1)) + (p >>> 1)
      end)

    pat = rotated &&& mask
    for j <- 0..(puncsize - 1), do: pat >>> (puncsize - 1 - j) &&& 1
  end

  @doc "Kept items per period."
  def kept(pattern), do: Enum.sum(pattern)

  @doc "Drops the items marked 0; `items` has a whole number of periods."
  def puncture(items, puncsize, puncpat, delay \\ 0) do
    pat = pattern(puncsize, puncpat, delay)
    n = Nx.axis_size(items, 0)
    if rem(n, puncsize) != 0, do: raise(ArgumentError, "input is not a whole number of periods")
    idx = for {1, j} <- Enum.with_index(pat), do: j

    items
    |> Nx.reshape({div(n, puncsize), puncsize})
    |> Nx.take(Nx.tensor(idx), axis: 1)
    |> Nx.reshape({:auto})
  end

  @doc "Reinserts `symbol` for the dropped items; `items` has a whole number of kept groups."
  def depuncture(items, puncsize, puncpat, delay \\ 0, symbol \\ 127) do
    pat = pattern(puncsize, puncpat, delay)
    keep = kept(pat)
    n = Nx.axis_size(items, 0)
    if rem(n, keep) != 0, do: raise(ArgumentError, "input is not a whole number of kept groups")
    groups = div(n, keep)

    {idx, _} =
      Enum.map_reduce(pat, 0, fn
        1, i -> {i, i + 1}
        0, i -> {keep, i}
      end)

    fill = Nx.broadcast(Nx.tensor(symbol, type: Nx.type(items)), {groups, 1})

    [Nx.reshape(items, {groups, keep}), fill]
    |> Nx.concatenate(axis: 1)
    |> Nx.take(Nx.tensor(idx), axis: 1)
    |> Nx.reshape({:auto})
  end

  @doc "Frame function of the `puncture_*` blocks (`opts`: `puncsize`, `puncpat`, `delay`)."
  def puncture_frames(items, state, opts),
    do: {puncture(items, opts[:puncsize], opts[:puncpat], opts[:delay] || 0), state}

  @doc "Frame function of `depuncture_bb`."
  def depuncture_frames(items, state, opts),
    do:
      {depuncture(
         items,
         opts[:puncsize],
         opts[:puncpat],
         opts[:delay] || 0,
         opts[:symbol] || 127
       ), state}
end
