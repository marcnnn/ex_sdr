defmodule ExSdr.FEC.Extended do
  @moduledoc """
  gr-fec's `extended_encoder` / `extended_decoder` hierarchical blocks for a
  code object: the conversions around the code and the puncturing.

  Encoder: pack the input bits to bytes if the code wants them (`"pack"`),
  encode, unpack the output if it is packed bits, puncture with `puncpat`
  (a string of 0 and 1, the first character for the first item of a period;
  `"11"` is no puncturing). Decoder: floats (positive: bit 1) are converted to
  the 8 bit input (`ExSdr.FEC.Generic.to_u8/3` with the decoder's shift) when the
  decoder asks for `uchar`, otherwise shifted by it, depunctured (symbol 127,
  uchar decoders only) and decoded by `Generic.decode_frames/3`.
  """
  alias ExSdr.FEC.Generic
  alias ExSdr.FEC.Puncture

  @doc "GNU Radio's `read_bitlist`: `\"1101\"` is 13."
  def read_bitlist(s),
    do: s |> String.to_charlist() |> Enum.reduce(0, fn c, acc -> acc * 2 + (c - ?0) end)

  @doc "`{encoded bits (u8 tensor), state}` for a whole number of frames."
  def encode(g, bits, state, puncpat \\ "11") do
    bits = Generic.to_tensor(bits, :u8)

    bits =
      if Generic.input_conversion(g) == "pack",
        do: bits |> Nx.to_binary() |> ExSdr.Comm.Bits.pack() |> Generic.to_tensor(:u8),
        else: bits

    {out, state} = Generic.encode_frames(bits, state, code: g)

    out =
      if Generic.output_conversion(g) == "packed_bits",
        do: out |> Nx.to_binary() |> ExSdr.Comm.Bits.unpack() |> Generic.to_tensor(:u8),
        else: out

    {if(puncpat == "11",
       do: out,
       else: Puncture.puncture(out, String.length(puncpat), read_bitlist(puncpat), 0)
     ), state}
  end

  @doc "`{decoded bits | nil, state}`; `state` as made by `Generic.decoder_init/1`."
  def decode(g, soft, state, puncpat \\ "11") do
    soft = Generic.to_tensor(soft, :f32)
    shift = Generic.shift(g)

    input =
      case Generic.input_conversion(g) do
        "uchar" -> Generic.to_u8(soft, 48.0, shift)
        _ -> if shift == 0.0, do: soft, else: Nx.add(soft, shift)
      end

    input =
      if puncpat == "11",
        do: input,
        else: Puncture.depuncture(input, String.length(puncpat), read_bitlist(puncpat), 0, 127)

    Generic.decode_frames(input, state, code: g)
  end
end
