defmodule ExSdr.Digital.Framing do
  @moduledoc """
  Framing on bits and bytes (plain Elixir, state explicit so a stream in
  pieces gives the same result as one call): access code correlation,
  GNU Radio's default packet headers, CRC wrappers for packets and PDUs and
  HDLC framing.

  Bits are binaries with one 0/1 per byte (GNU Radio's "unpacked" stream,
  data in the LSB); bytes are binaries of packed bytes.
  """
  import Bitwise

  alias ExSdr.Digital.CRC

  # -- access code -------------------------------------------------------------------------

  @doc """
  Access code correlator state. `access_code` is a string or list of 0/1
  (at most 64); `threshold` the number of wrong bits still accepted.
  """
  def correlator(access_code, threshold) do
    bits =
      if is_binary(access_code),
        do: for(<<c <- access_code>>, do: c - ?0),
        else: access_code

    len = length(bits)
    true = len in 1..64

    %{
      code: Enum.reduce(bits, 0, &(&2 <<< 1 ||| &1)),
      mask: (1 <<< len) - 1,
      threshold: threshold,
      data: 0,
      flags: 0,
      n: 0
    }
  end

  @doc """
  `correlate_access_code_bb`: the input delayed by 64 bits (zeros first),
  in the LSB of each output byte, with bit 1 set on the first bit after an
  access code (up to `threshold` bits wrong). Output and input have the same
  length. Returns `{output_bits, state}`.
  """
  def correlate_access_code(c, bits) when is_binary(bits) do
    {out, c} =
      for <<b <- bits>>, reduce: {[], c} do
        {acc, c} ->
          {_, c2, hit} = step(c, b)
          out = (c.data >>> 63 &&& 1) ||| (c.flags >>> 64 &&& 1) <<< 1
          {[out | acc], %{c2 | flags: (c.flags <<< 1 ||| hit) &&& (1 <<< 65) - 1}}
      end

    {out |> Enum.reverse() |> :binary.list_to_bin(), c}
  end

  @doc """
  `correlate_access_code_tag_bb`: the input unchanged and the tags it adds:
  `{offset, wrong_bits}` where `offset` is the index of the first bit after
  the access code. Returns `{{bits, tags}, state}`.
  """
  def correlate_access_code_tag(c, bits) when is_binary(bits) do
    {tags, c} =
      for <<b <- bits>>, reduce: {[], c} do
        {acc, c} ->
          {nwrong, c2, hit} = step(c, b)
          acc = if hit == 1, do: [{c.n + 1, nwrong} | acc], else: acc
          {acc, c2}
      end

    {{bits, Enum.reverse(tags)}, c}
  end

  # shift one bit in; {wrong bits, state, 1 if it completes an access code}
  defp step(c, b) do
    data = (c.data <<< 1 ||| (b &&& 1)) &&& (1 <<< 64) - 1
    nwrong = popcount(bxor(data, c.code) &&& c.mask)
    {nwrong, %{c | data: data, n: c.n + 1}, if(nwrong <= c.threshold, do: 1, else: 0)}
  end

  defp popcount(x) do
    bin = <<x::64>>
    for <<b::1 <- bin>>, reduce: 0, do: (acc -> acc + b)
  end

  # -- packet headers ---------------------------------------------------------------------

  @doc """
  GNU Radio's `header_format_default` / `header_format_counter` /
  `header_format_crc` as data. `kind` is `:default`, `:counter` or `:crc`
  (which ignores the access code); `bps` only matters for `:counter` (it is a
  header field). The state of `header/3` is the packet counter.
  """
  def header_format(kind, access_code, threshold, bps \\ 1)
      when kind in [:default, :counter, :crc] do
    bits = if is_binary(access_code), do: for(<<c <- access_code>>, do: c - ?0), else: access_code
    %{kind: kind, access_code: bits, threshold: threshold, bps: bps}
  end

  @doc """
  The header bytes for a payload of `nbytes` bytes: the access code (as
  GNU Radio does, only its last `8 * div(len, 8)` bits are written), the
  payload length as 16 bits twice, and for `:counter` the bits per symbol
  and the packet counter (16 bits each). `:crc` gives 4 bytes: length (12
  bits) and packet number (12 bits) least significant bit first, and a CRC-8
  (polynomial 7, initial value 0xFF) of both as 16 bit little-endian words.
  Returns `{header, counter}`.
  """
  def header(f, nbytes, counter \\ 0)

  def header(%{kind: :crc}, nbytes, counter) do
    num = counter &&& 0xFFF
    len = nbytes &&& 0xFFF
    crc = CRC.new(width: 8, poly: 0x07, init: 0xFF)
    check = CRC.compute(crc, <<len::little-16, num::little-16>>)
    {<<len &&& 255, (num &&& 15) <<< 4 ||| len >>> 8, num >>> 4, check>>, counter + 1}
  end

  def header(%{kind: kind} = f, nbytes, counter) do
    code = Enum.reduce(f.access_code, 0, &(&2 <<< 1 ||| &1))
    n = 8 * div(length(f.access_code), 8)
    base = <<code::size(n), nbytes::16, nbytes::16>>

    case kind do
      :default -> {base, counter}
      :counter -> {<<base::binary, f.bps::16, counter &&& 0xFFFF::16>>, counter + 1}
    end
  end

  @doc """
  The payload length from a received `:default` header (the bytes after
  the access code): `{:ok, length}` when the two length fields agree.
  """
  def parse_length(<<len::16, len::16, _::binary>>), do: {:ok, len}
  def parse_length(_), do: :error

  # -- CRC on packets ----------------------------------------------------------------------

  @doc "The CRC-32 `crc32_bb` uses (Ethernet): reflected, init and final XOR all ones."
  def crc32, do: crc32_params() |> CRC.new()

  defp crc32_params,
    do: [
      width: 32,
      poly: 0x04C11DB7,
      init: 0xFFFFFFFF,
      reflect_in: true,
      reflect_out: true,
      xorout: 0xFFFFFFFF
    ]

  @doc """
  `crc32_bb` on one packet. Packed: the bytes plus the CRC-32 as four bytes,
  least significant first; with `check: true` the CRC is verified and
  removed (`:error` if wrong or the packet is shorter than four bytes).
  Unpacked (`packed: false`, one bit per byte, 8 bits per data byte LSB
  first): the same on bits, the CRC appended as 32 bits LSB first.
  """
  def crc32_packet(packet, opts \\ []) do
    check = Keyword.get(opts, :check, false)
    packed = Keyword.get(opts, :packed, true)
    crc = crc32()
    bytes = if packed, do: packet, else: pack_lsb_first(packet)

    if check do
      n = byte_size(bytes) - 4

      with true <- n >= 0,
           <<body::binary-size(n), tail::little-32>> <- bytes,
           true <- CRC.compute(crc, body) == tail do
        {:ok, if(packed, do: body, else: binary_part(packet, 0, n * 8))}
      else
        _ -> :error
      end
    else
      c = CRC.compute(crc, bytes)
      tail = <<c::little-32>>
      {:ok, if(packed, do: packet <> tail, else: packet <> unpack_lsb_first(tail))}
    end
  end

  defp pack_lsb_first(bits) do
    for <<b0, b1, b2, b3, b4, b5, b6, b7 <- binary_part(bits, 0, div(byte_size(bits), 8) * 8)>>,
      into: <<>>,
      do:
        <<(b0 ||| b1 <<< 1 ||| b2 <<< 2 ||| b3 <<< 3 ||| b4 <<< 4 ||| b5 <<< 5 ||| b6 <<< 6 |||
             b7 <<< 7) &&& 0xFF>>
  end

  defp unpack_lsb_first(bytes),
    do: for(<<b <- bytes>>, k <- 0..7, into: <<>>, do: <<b >>> k &&& 1>>)

  @doc """
  `crc_append` on one PDU: the CRC of `data` (without the first
  `skip_header_bytes`) appended, most significant byte first, or least
  significant first with `swap_endianness`. `opts` are the block's: `:num_bits`,
  `:poly`, `:initial_value`, `:final_xor`, `:input_reflected`,
  `:result_reflected`, `:swap_endianness`, `:skip_header_bytes`.
  """
  def crc_append(pdu, opts) do
    {crc, skip, swap} = pdu_crc(opts)
    <<_::binary-size(skip), body::binary>> = pdu
    pdu <> CRC.to_bytes(CRC.compute(crc, body), crc.width, swap)
  end

  @doc """
  `crc_check` on one PDU: `{:ok, pdu}` or `{:fail, pdu}`; with
  `discard_crc: true` the CRC bytes are removed from the PDU either way
  (as GNU Radio does).
  """
  def crc_check(pdu, opts) do
    {crc, skip, swap} = pdu_crc(opts)
    n = div(crc.width, 8)
    discard = Keyword.get(opts, :discard_crc, true)
    size = byte_size(pdu) - n

    if size < skip do
      {:fail, pdu}
    else
      <<body::binary-size(size), tail::binary-size(n)>> = pdu
      <<_::binary-size(skip), checked::binary>> = body
      ok = CRC.to_bytes(CRC.compute(crc, checked), crc.width, swap) == tail
      {if(ok, do: :ok, else: :fail), if(discard, do: body, else: pdu)}
    end
  end

  @doc "`crc_append` options of GNU Radio's `crc32_async_bb` (CRC-32, least significant byte first)."
  def crc32_pdu_opts,
    do: [
      num_bits: 32,
      poly: 0x04C11DB7,
      initial_value: 0xFFFFFFFF,
      final_xor: 0xFFFFFFFF,
      input_reflected: true,
      result_reflected: true,
      swap_endianness: true
    ]

  @doc "`crc_append` options of GNU Radio's `crc16_async_bb` (CRC-16/CCITT-FALSE, most significant byte first)."
  def crc16_pdu_opts,
    do: [
      num_bits: 16,
      poly: 0x1021,
      initial_value: 0xFFFF,
      final_xor: 0,
      input_reflected: false,
      result_reflected: false,
      swap_endianness: false
    ]

  defp pdu_crc(opts) do
    crc =
      CRC.new(
        width: Keyword.fetch!(opts, :num_bits),
        poly: Keyword.fetch!(opts, :poly),
        init: Keyword.fetch!(opts, :initial_value),
        xorout: Keyword.fetch!(opts, :final_xor),
        reflect_in: Keyword.fetch!(opts, :input_reflected),
        reflect_out: Keyword.fetch!(opts, :result_reflected)
      )

    {crc, Keyword.get(opts, :skip_header_bytes, 0), Keyword.get(opts, :swap_endianness, false)}
  end

  # -- HDLC ---------------------------------------------------------------------------------

  @flag <<0, 1, 1, 1, 1, 1, 1, 0>>

  defp fcs,
    do:
      CRC.new(
        width: 16,
        poly: 0x1021,
        init: 0xFFFF,
        reflect_in: true,
        reflect_out: true,
        xorout: 0xFFFF
      )

  @doc """
  `hdlc_framer_pb` on one payload: unpacked bits of flag (0x7E), the
  payload and its CRC-16 (X.25) least significant bit and byte first with a
  0 stuffed after every five 1s, flag.
  """
  def hdlc_frame(payload) do
    crc = CRC.compute(fcs(), payload)
    data = unpack_lsb_first(payload <> <<crc::little-16>>)
    @flag <> stuff(data, 0, <<>>) <> @flag
  end

  # a 0 follows five 1s only if more data follows (GNU Radio does not stuff
  # at the end, the flag's own 0 comes next)
  defp stuff(<<>>, _, acc), do: acc
  defp stuff(bits, 5, acc), do: stuff(bits, 0, <<acc::binary, 0>>)
  defp stuff(<<1, rest::binary>>, ones, acc), do: stuff(rest, ones + 1, <<acc::binary, 1>>)
  defp stuff(<<0, rest::binary>>, _, acc), do: stuff(rest, 0, <<acc::binary, 0>>)

  @doc "Initial state of the HDLC deframer."
  def hdlc_deframer(length_min \\ 32, length_max \\ 500),
    do: %{min: length_min, max: length_max, sync: 0, bits: nil, ones: 0}

  @doc """
  `hdlc_deframer_bp`: finds frames in a stream of unpacked bits. Frames
  that fail the CRC, are not whole bytes, contain an abort (seven 1s) or
  whose length with the two CRC bytes is outside `length_min..length_max`
  are dropped.
  Returns `{payloads, state}`.
  """
  def hdlc_deframe(d, bits) when is_binary(bits) do
    {frames, d} =
      for <<b <- bits>>, reduce: {[], d} do
        {acc, d} ->
          {frame, d} = deframe_bit(d, b)
          {if(frame, do: [frame | acc], else: acc), d}
      end

    {Enum.reverse(frames), d}
  end

  defp deframe_bit(%{bits: nil} = d, b) do
    sync = (d.sync <<< 1 ||| b) &&& 0xFF
    if sync == 0x7E, do: {nil, %{d | sync: 0, bits: [], ones: 0}}, else: {nil, %{d | sync: sync}}
  end

  defp deframe_bit(d, b) do
    bits = [b | d.bits]
    sync = (d.sync <<< 1 ||| b) &&& 0xFF
    ones = if b == 1, do: d.ones + 1, else: 0

    cond do
      sync == 0x7E -> {finish_frame(d, Enum.drop(bits, 8)), %{d | bits: [], sync: 0, ones: 0}}
      ones >= 7 -> {nil, %{d | bits: nil, sync: 0, ones: 0}}
      true -> {nil, %{d | bits: bits, sync: sync, ones: ones}}
    end
  end

  # `raw` is the stuffed frame (newest bit first), without the flags
  defp finish_frame(d, raw) do
    with {:ok, body} <- unstuff(Enum.reverse(raw), 0, []),
         n = length(body),
         true <- n >= 24 and rem(n, 8) == 0,
         bytes = pack_lsb_first(:binary.list_to_bin(body)),
         size = byte_size(bytes) - 2,
         <<payload::binary-size(size), crc::little-16>> <- bytes,
         true <- size + 2 >= d.min and size + 2 <= d.max,
         true <- CRC.compute(fcs(), payload) == crc do
      payload
    else
      _ -> nil
    end
  end

  defp unstuff([], _, acc), do: {:ok, Enum.reverse(acc)}
  defp unstuff([0 | rest], 5, acc), do: unstuff(rest, 0, acc)
  defp unstuff([1 | _], 5, _), do: :error
  defp unstuff([1 | rest], ones, acc), do: unstuff(rest, ones + 1, [1 | acc])
  defp unstuff([0 | rest], _, acc), do: unstuff(rest, 0, [0 | acc])
end
