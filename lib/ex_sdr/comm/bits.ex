defmodule ExSdr.Comm.Bits do
  @moduledoc "Bit packing helpers (binaries of 0/1 bytes <-> bytes)."
  import Bitwise

  # Native packs up to this many bits run on a regular scheduler: measured
  # on the dev host (aarch64) at about 0.05 ns per bit, so 2 Mbit is about
  # 0.1 ms, under the BEAM's 1 ms rule for regular NIFs.
  @small_bits 2_000_000

  @doc """
  Packs 0/1 bytes into bytes, MSB first (a trailing partial byte is
  dropped). Option `:native` (default: when the NIF is loaded) selects the
  native kernel or the Elixir reference.
  """
  def pack(bits, opts \\ []) do
    cond do
      not Keyword.get_lazy(opts, :native, &ExSdr.SIMD.available?/0) -> pack_elixir(bits)
      byte_size(bits) <= @small_bits -> ExSdr.NIF.bits_pack_small(bits)
      true -> ExSdr.NIF.bits_pack(bits)
    end
  end

  defp pack_elixir(bits) do
    for <<b0, b1, b2, b3, b4, b5, b6, b7 <- binary_part(bits, 0, div(byte_size(bits), 8) * 8)>>,
      into: <<>>,
      do:
        <<b0 <<< 7 ||| b1 <<< 6 ||| b2 <<< 5 ||| b3 <<< 4 ||| b4 <<< 3 ||| b5 <<< 2 ||| b6 <<< 1 |||
            b7>>
  end

  @doc "Unpacks bytes into 0/1 bytes, MSB first."
  def unpack(bytes), do: for(<<b::1 <- bytes>>, into: <<>>, do: <<b>>)
end
