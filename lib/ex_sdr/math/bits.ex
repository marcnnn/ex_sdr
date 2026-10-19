defmodule ExSdr.Math.Bits do
  @moduledoc """
  Bit and byte packing of the math family, on integer Nx tensors (`u8`).

  A bit is an item holding 0 or 1. The k-bit packing blocks treat a byte's
  low `k` bits as the item.

    * `unpack_k_bits/3` (frame, rate `{1, k}`): each byte gives its low `k`
      bits, most significant first (GNU Radio `unpack_k_bits_bb`).
    * `pack_k_bits/3` (frame, rate `{k, 1}`): `k` bits, most significant
      first, make one byte (GNU Radio `pack_k_bits_bb`).
    * `repack_bits/3` (host, rate `{k, l}`): a stream of `k`-bit items
      becomes a stream of `l`-bit bytes, least significant bit first (GNU
      Radio `repack_bits_bb`, default endianness). Bits that do not yet fill
      an output byte wait in the state, so any split of the input gives the
      same output.
  """
  import Bitwise

  @doc false
  # Frame kernels return `{output, state}`; these have no state.
  def unpack_k_bits(input, _state, opts) do
    k = Keyword.fetch!(opts, :k)

    shifts =
      if Keyword.get(opts, :endianness, :msb) == :msb,
        do: Enum.to_list((k - 1)..0//-1),
        else: Enum.to_list(0..(k - 1))

    input
    |> Nx.as_type(:s32)
    |> Nx.new_axis(1)
    |> Nx.right_shift(Nx.tensor(shifts, type: :s32))
    |> Nx.bitwise_and(1)
    |> Nx.as_type(:u8)
    |> Nx.reshape({:auto})
    |> then(&{&1, nil})
  end

  @doc false
  def pack_k_bits(input, _state, opts) do
    k = Keyword.fetch!(opts, :k)
    weights = weights(k, Keyword.get(opts, :endianness, :msb))

    input
    |> Nx.as_type(:s32)
    |> Nx.bitwise_and(1)
    |> Nx.reshape({:auto, k})
    |> Nx.multiply(Nx.tensor(weights, type: :s32))
    |> Nx.sum(axes: [1])
    |> Nx.as_type(:u8)
    |> then(&{&1, nil})
  end

  @doc """
  Repacking kernel for `k`-bit items (`k` at most 8) into `l`-bit bytes
  (`l` at most 8). Input: a `u8` tensor; output: a binary of bytes, which
  may be empty (the host contract allows binaries; Nx cannot hold empty
  tensors). The state is the bitstring of bits not yet forming an output
  byte.

  Options `:in_order` and `:out_order` (`:lsb` or `:msb`, default `:lsb`) say
  which bit of an input item comes first in the bit stream and which bit of
  an output item the first stream bit becomes. `repack_bits_bb` is
  `:lsb` on both sides; `packed_to_unpacked_bb` and `unpacked_to_packed_bb`
  take the endianness for the byte side and are `:msb` on the chunk side.
  """
  def repack_bits(k, l, opts \\ []) when k in 1..8 and l in 1..8 do
    in_order = Keyword.get(opts, :in_order, :lsb)
    out_order = Keyword.get(opts, :out_order, :lsb)

    in_shifts =
      if in_order == :lsb, do: Enum.to_list(0..(k - 1)), else: Enum.to_list((k - 1)..0//-1)

    fn input, leftover ->
      bits =
        for <<b <- Nx.to_binary(input)>>, j <- in_shifts, into: <<>>, do: <<b >>> j &&& 1::1>>

      all = <<leftover::bitstring, bits::bitstring>>
      count = div(bit_size(all), l)
      used = count * l
      <<head::bitstring-size(used), rest::bitstring>> = all

      out = for <<group::size(l) <- head>>, into: <<>>, do: <<group_value(group, l, out_order)>>
      {out, rest}
    end
  end

  # A group of l stream bits (first bit first) as a value: the first bit is
  # the most significant for :msb, the least significant for :lsb.
  defp group_value(group, _l, :msb), do: group

  defp group_value(group, l, :lsb) do
    for(<<(bit::1 <- <<group::size(l)>>)>>, do: bit)
    |> Enum.with_index()
    |> Enum.reduce(0, fn {bit, j}, acc -> acc ||| bit <<< j end)
  end

  defp weights(k, :msb), do: for(j <- (k - 1)..0//-1, do: 1 <<< j)
  defp weights(k, :lsb), do: for(j <- 0..(k - 1), do: 1 <<< j)
end
