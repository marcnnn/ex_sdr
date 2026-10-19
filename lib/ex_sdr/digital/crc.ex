defmodule ExSdr.Digital.CRC do
  @moduledoc """
  Generic CRC in the Rocksoft parameter model, the same parameters as
  GNU Radio's `digital.crc(num_bits, poly, initial_value, final_xor,
  input_reflected, result_reflected)`: `width`, `poly` (MSB-first notation,
  without the leading term), `init`, `reflect_in`, `reflect_out`, `xorout`.

  The register is carried explicitly (`update/3`), so data in pieces gives
  the same CRC as the whole.

      iex> crc = ExSdr.Digital.CRC.new(width: 32, poly: 0x04C11DB7, init: 0xFFFFFFFF,
      ...>   reflect_in: true, reflect_out: true, xorout: 0xFFFFFFFF)
      iex> Integer.to_string(ExSdr.Digital.CRC.compute(crc, "123456789"), 16)
      "CBF43926"
  """
  import Bitwise

  defstruct [:width, :poly, :init, :reflect_in, :reflect_out, :xorout, :table]

  @doc "Options: `:width`, `:poly`, `:init` (0), `:reflect_in` (false), `:reflect_out` (false), `:xorout` (0)."
  def new(opts) do
    width = Keyword.fetch!(opts, :width)
    poly = Keyword.fetch!(opts, :poly)
    true = width in 1..64

    crc = %__MODULE__{
      width: width,
      poly: poly,
      init: Keyword.get(opts, :init, 0),
      reflect_in: Keyword.get(opts, :reflect_in, false),
      reflect_out: Keyword.get(opts, :reflect_out, false),
      xorout: Keyword.get(opts, :xorout, 0)
    }

    %{crc | table: if(width >= 8, do: table(crc))}
  end

  @doc "The CRC of `data` (a binary of bytes)."
  def compute(%__MODULE__{} = crc, data), do: crc |> update(init_state(crc), data) |> finish(crc)

  @doc "The register before any data."
  def init_state(%__MODULE__{init: init, width: w}), do: init &&& mask(w)

  @doc "Feeds `data` (bytes) into the register `state`; returns the new register."
  def update(%__MODULE__{table: nil} = crc, state, data) do
    for <<byte <- data>>, reduce: state do
      reg -> bit_by_bit(crc, reg, byte)
    end
  end

  def update(%__MODULE__{table: table, width: w} = crc, state, data) do
    m = mask(w)

    for <<byte <- data>>, reduce: state do
      reg ->
        byte = if crc.reflect_in, do: reverse(byte, 8), else: byte
        idx = bxor(reg >>> (w - 8), byte) &&& 0xFF
        bxor(reg <<< 8 &&& m, elem(table, idx))
    end
  end

  @doc "The CRC for register `state` (reflection and final XOR applied)."
  def finish(state, %__MODULE__{width: w} = crc) do
    out = if crc.reflect_out, do: reverse(state, w), else: state
    bxor(out, crc.xorout) &&& mask(w)
  end

  @doc "The CRC as `width / 8` bytes, most significant first (`little?`: least significant first)."
  def to_bytes(value, width, little? \\ false) do
    0 = rem(width, 8)
    big = <<value::size(width)>>

    if little?,
      do: big |> :binary.bin_to_list() |> Enum.reverse() |> :binary.list_to_bin(),
      else: big
  end

  defp bit_by_bit(crc, reg, byte) do
    w = crc.width
    byte = if crc.reflect_in, do: reverse(byte, 8), else: byte

    Enum.reduce(7..0//-1, reg, fn i, reg ->
      top = bxor(reg >>> (w - 1), byte >>> i) &&& 1
      reg = reg <<< 1 &&& mask(w)
      if top == 1, do: bxor(reg, crc.poly), else: reg
    end)
  end

  defp table(crc) do
    w = crc.width

    for n <- 0..255 do
      Enum.reduce(1..8, n <<< (w - 8), fn _, reg ->
        top = reg >>> (w - 1) &&& 1
        reg = reg <<< 1 &&& mask(w)
        if top == 1, do: bxor(reg, crc.poly), else: reg
      end)
    end
    |> List.to_tuple()
  end

  defp mask(w), do: (1 <<< w) - 1

  @doc false
  def reverse(x, bits) do
    Enum.reduce(0..(bits - 1), 0, fn i, acc -> acc ||| (x >>> i &&& 1) <<< (bits - 1 - i) end)
  end
end
