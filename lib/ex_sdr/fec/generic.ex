defmodule ExSdr.FEC.Generic do
  @moduledoc """
  GNU Radio's `generic_encoder` / `generic_decoder` interface and the blocks
  that run such objects: `fec.encoder`, `fec.decoder` (frames of a stream),
  `fec.tagged_encoder` / `fec.tagged_decoder` (packets of varying length) and
  `fec.async_encoder` / `fec.async_decoder` (PDUs).

  A code object is a `%ExSdr.FEC.Generic{}`: an implementation module (a
  `ExSdr.FEC.Generic` behaviour: `ExSdr.FEC.Conv`, `Repetition`, `Dummy`,
  `Polar`, ...), its configuration and the current frame size. Everything is
  plain data; `set_frame_size/2` returns a new object.

  The core of every code is `work/3`, one frame: `input` (a tensor with
  `input_size + history` items) to `{output, state}`. `:frame` blocks
  (`encode_frames/3`, `decode_frames/3`) run it over a stream of frames, the
  `:host` wrappers over packets and PDUs. The state is explicit: encoders of
  convolutional codes in streaming mode carry the register, streaming
  decoders carry the look-ahead items and the chained start state.

  Items: bits are `u8` tensors holding 0/1. Soft decoder input is what the
  decoder says (`input_item/1`): `:u8` for the Viterbi decoder (0 is a sure
  0, 255 a sure 1, see `to_u8/3`), `:f32` for the others (positive: bit 1,
  as GNU Radio's decoders). Output bits are `u8` tensors.
  """
  defstruct [:impl, :name, :kind, :conf, :frame_size, :max_frame_size]

  @type t :: %__MODULE__{}

  @doc "Items consumed per frame (without history)."
  @callback input_size(t) :: pos_integer
  @doc "Items produced per frame."
  @callback output_size(t) :: pos_integer
  @doc "GNU Radio's `rate()`: output over input items for an encoder, its inverse for a decoder."
  @callback rate(t) :: float
  @doc "One frame: `input` has `input_size + history` items."
  @callback work(t, input :: Nx.Tensor.t(), state :: term) :: {Nx.Tensor.t(), term}
  @doc "Initial state (nil if the code has none)."
  @callback init(t) :: term
  @doc "`:u8` or `:f32` (decoders)."
  @callback input_item(t) :: :u8 | :f32
  @doc "Look-ahead items a decoder needs after the frame (GNU Radio `get_history`)."
  @callback history(t) :: non_neg_integer
  @doc "Offset GNU Radio adds before the 8 bit conversion (`get_shift`)."
  @callback shift(t) :: float
  @doc "Hardware structure (`NxFixed.Block`'s `:hardware`)."
  @callback hardware(t) :: atom
  @doc "Override of the frame size rule; returns `{ok?, object}`."
  @callback set_frame_size(t, non_neg_integer) :: {boolean, t}
  @doc "Items of delay (`:latency` of the block)."
  @callback latency(t) :: non_neg_integer
  @doc "Input items after which the output no longer depends on the initial state (`:warmup`)."
  @callback warmup(t) :: non_neg_integer | :unbounded
  @doc "GNU Radio's input conversion: \"none\", \"pack\", \"uchar\" or \"packed_bits\"."
  @callback input_conversion(t) :: String.t()
  @doc "GNU Radio's output conversion: \"none\" or \"packed_bits\"."
  @callback output_conversion(t) :: String.t()

  @optional_callbacks init: 1,
                      input_item: 1,
                      history: 1,
                      shift: 1,
                      hardware: 1,
                      set_frame_size: 2,
                      latency: 1,
                      warmup: 1,
                      input_conversion: 1,
                      output_conversion: 1

  @doc "A code object of `impl` (used by the implementations' constructors)."
  def new(impl, name, kind, conf, frame_size) do
    %__MODULE__{
      impl: impl,
      name: name,
      kind: kind,
      conf: conf,
      frame_size: frame_size,
      max_frame_size: frame_size
    }
  end

  def input_size(%{impl: m} = g), do: m.input_size(g)
  def output_size(%{impl: m} = g), do: m.output_size(g)
  def rate(%{impl: m} = g), do: m.rate(g)
  def work(%{impl: m} = g, input, state), do: m.work(g, input, state)

  def init(%{impl: m} = g), do: optional(m, :init, g, nil)
  def input_item(%{impl: m} = g), do: optional(m, :input_item, g, :u8)
  def history(%{impl: m} = g), do: optional(m, :history, g, 0)
  def shift(%{impl: m} = g), do: optional(m, :shift, g, 0.0)
  def hardware(%{impl: m} = g), do: optional(m, :hardware, g, :transform)
  def latency(%{impl: m} = g), do: optional(m, :latency, g, 0)
  def warmup(%{impl: m} = g), do: optional(m, :warmup, g, 0)
  def input_conversion(%{impl: m} = g), do: optional(m, :input_conversion, g, "none")
  def output_conversion(%{impl: m} = g), do: optional(m, :output_conversion, g, "none")

  defp optional(m, f, g, default) do
    Code.ensure_loaded(m)
    if function_exported?(m, f, 1), do: apply(m, f, [g]), else: default
  end

  @doc """
  `set_frame_size`: frames larger than the one the object was made for are
  clamped to it and reported (`false`), as in GNU Radio.
  """
  def set_frame_size(%{impl: m} = g, n) do
    Code.ensure_loaded(m)

    if function_exported?(m, :set_frame_size, 2) do
      m.set_frame_size(g, n)
    else
      default_set_frame_size(g, n)
    end
  end

  @doc false
  def default_set_frame_size(g, n) when n > g.max_frame_size,
    do: {false, %{g | frame_size: g.max_frame_size}}

  def default_set_frame_size(g, n), do: {true, %{g | frame_size: n}}

  # -- conversions --------------------------------------------------------------------

  @doc """
  The 8 bit soft input of GNU Radio's `uchar` decoders from float values
  (positive: bit 1): `clamp(round(x * scale + bias), 0, 255)` with round half
  to even, as `blocks.float_to_uchar(scale=48, bias=128)` in
  `extended_decoder` and volk's `volk_32f_s32f_x2_convert_8u` in
  `async_decoder`. Takes a list or tensor, returns a `u8` tensor.
  """
  def to_u8(values, scale \\ 48.0, bias \\ 128.0) do
    values
    |> to_tensor(:f32)
    |> Nx.multiply(scale)
    |> Nx.add(bias)
    |> round_half_even()
    |> Nx.clip(0, 255)
    |> Nx.as_type(:u8)
  end

  defp round_half_even(x) do
    fl = Nx.floor(x)
    d = Nx.subtract(x, fl)
    odd = Nx.equal(Nx.remainder(fl, 2), 1)
    up = Nx.logical_or(Nx.greater(d, 0.5), Nx.logical_and(Nx.equal(d, 0.5), odd))
    Nx.add(fl, Nx.as_type(up, :f32))
  end

  @doc "Items as a tensor of `type` from a list, a binary of u8 or a tensor."
  def to_tensor(%Nx.Tensor{} = t, type), do: Nx.as_type(t, type)
  def to_tensor(list, type) when is_list(list), do: Nx.tensor(list, type: type)

  def to_tensor(bin, :u8) when is_binary(bin), do: Nx.from_binary(bin, :u8)
  def to_tensor(bin, type) when is_binary(bin), do: Nx.from_binary(bin, type)

  @doc "Tensor of 0/1 or u8 as a binary of bytes."
  def to_binary(nil), do: <<>>
  def to_binary(%Nx.Tensor{} = t), do: t |> Nx.as_type(:u8) |> Nx.to_binary()

  # -- stream blocks ----------------------------------------------------------------------

  @doc """
  `fec.encoder`: `items` is a whole number of frames; `opts[:code]` the
  encoder object. Returns `{output, state}`.
  """
  def encode_frames(items, state, opts) do
    g = Keyword.fetch!(opts, :code)
    size = input_size(g)
    n = Nx.axis_size(items, 0)
    if rem(n, size) != 0, do: raise(ArgumentError, "input is not a whole number of frames")

    {outs, state} =
      Enum.map_reduce(0..(div(n, size) - 1), state, fn i, st ->
        work(g, Nx.slice_along_axis(items, i * size, size), st)
      end)

    {Nx.concatenate(outs), state}
  end

  @doc """
  `fec.decoder`: runs the decoder object `opts[:code]` over frames. A decoder
  with history (`Conv` in streaming mode) reads `history` items past the
  frame, so the last frames wait in the state until the next call; the output
  is `nil` while no frame is complete. Returns `{output | nil, state}`.
  """
  def decode_frames(items, state, opts) do
    g = Keyword.fetch!(opts, :code)
    size = input_size(g)
    hist = history(g)

    buf =
      case state.carry do
        nil -> items
        carry -> Nx.concatenate([carry, items])
      end

    n = Nx.axis_size(buf, 0)
    frames = if n >= size + hist, do: div(n - hist, size), else: 0

    {outs, impl_state} =
      Enum.map_reduce(List.duplicate(nil, frames) |> Enum.with_index(), state.impl, fn {_, i},
                                                                                       st ->
        work(g, Nx.slice_along_axis(buf, i * size, size + hist), st)
      end)

    used = frames * size

    carry =
      if used < n, do: Nx.slice_along_axis(buf, used, n - used), else: nil

    out = if outs == [], do: nil, else: Nx.concatenate(outs)
    {out, %{carry: carry, impl: impl_state}}
  end

  @doc "Initial state of `decode_frames/3`."
  def decoder_init(g), do: %{carry: nil, impl: init(g)}

  # -- tagged streams ------------------------------------------------------------------

  @doc """
  `fec.tagged_encoder`: `packets` is a list of packets (tensors or binaries of
  0/1 bytes), each as long as its length tag says; every packet is encoded
  with the frame size set to its length. `mtu` (bytes) limits the packet
  length (`8 * mtu` bits). Returns `{packets, state}` (binaries).
  """
  def tagged_encode(g, packets, state, mtu \\ 1500) do
    Enum.map_reduce(packets, state, fn p, st ->
      bits = to_tensor(p, :u8)
      n = Nx.axis_size(bits, 0)

      if n > mtu * 8,
        do: raise(ArgumentError, "tagged_encoder: received frame is larger than MTU")

      {_, gp} = set_frame_size(g, n)
      {out, st} = work(gp, bits, st)
      {to_binary(out), st}
    end)
  end

  @doc """
  `fec.tagged_decoder`: soft packets (tensors, lists or binaries of the
  decoder's input type); the frame size follows from the packet length as in
  GNU Radio (`round(n * rate) - diff`, where `diff` accounts for the
  termination bits of a terminated convolutional code). Decoders with history
  are not supported (as in GNU Radio's async decoder).
  """
  def tagged_decode(g, packets, state, mtu \\ 1500) do
    type = input_item(g)

    Enum.map_reduce(packets, state, fn p, st ->
      soft = to_tensor(p, type)
      n = Nx.axis_size(soft, 0)
      rate = rate(g)

      if round(n * rate) > mtu * 8,
        do: raise(ArgumentError, "tagged_decoder: received frame is larger than MTU")

      {_, gp} = set_frame_size(g, mtu * 8)
      diff = round(rate * input_size(gp)) - output_size(gp)
      {_, gp} = set_frame_size(gp, round(n * rate) - diff)
      {out, st} = work(gp, soft, st)
      {to_binary(out), st}
    end)
  end

  # -- PDUs ------------------------------------------------------------------------------

  @doc """
  `fec.async_encoder`: a PDU `{meta, data}` in, `{meta, data}` out. With
  `packed: false` data is a binary of 0/1 bytes (several code blocks if it is
  longer than the frame size), with `packed: true` packed bytes that are
  unpacked, encoded as one frame and packed again. Options as in GNU Radio:
  `:packed`, `:rev_unpack` and `:rev_pack` (both default true: least
  significant bit first, as the tagged stream encoders; false: MSB first),
  `:mtu` in bytes.
  """
  def async_encode(g, {meta, data}, state, opts \\ []) do
    packed = Keyword.get(opts, :packed, false)
    mtu = Keyword.get(opts, :mtu, 1500)

    if packed do
      if byte_size(data) > mtu, do: raise(ArgumentError, "message larger than the MTU")
      bits = unpack(data, Keyword.get(opts, :rev_unpack, true))
      {_, gp} = set_frame_size(g, byte_size(bits))
      {out, state} = work(gp, to_tensor(bits, :u8), state)
      out_bits = to_binary(out)
      nbytes = div(output_size(gp), 8)
      packed_out = pack(binary_part(out_bits, 0, nbytes * 8), Keyword.get(opts, :rev_pack, true))
      {{meta, packed_out}, state}
    else
      nbits = byte_size(data)
      if nbits > mtu * 8, do: raise(ArgumentError, "message larger than the MTU")
      {variable, gp} = set_frame_size(g, nbits)
      size = input_size(gp)

      blocks =
        if variable do
          [to_tensor(data, :u8)]
        else
          if rem(nbits, size) != 0,
            do: raise(ArgumentError, "input does not divide into code block size")

          for <<b::binary-size(size) <- data>>, do: to_tensor(b, :u8)
        end

      {outs, state} = Enum.map_reduce(blocks, state, &work(gp, &1, &2))
      {{meta, outs |> Enum.map(&to_binary/1) |> IO.iodata_to_binary()}, state}
    end
  end

  @doc """
  `fec.async_decoder`: a PDU `{meta, soft}` in (floats as a list or tensor),
  `{meta, bits}` out. Floats are converted to the decoder's 8 bit input
  (`to_u8/3` with `shift` as bias) when it asks for `uchar`. With
  `packed: true` the bits are packed into bytes. Decoders with history are
  rejected, as in GNU Radio.
  """
  def async_decode(g, {meta, soft}, state, opts \\ []) do
    if history(g) > 0,
      do: raise(ArgumentError, "async_decoder does not support decoders with history")

    packed = Keyword.get(opts, :packed, false)
    mtu = Keyword.get(opts, :mtu, 1500)
    soft = to_tensor(soft, :f32)
    nbits_in = Nx.axis_size(soft, 0)
    rate = rate(g)
    max_in = round(mtu * 8 * 1.0 / rate)
    if nbits_in > max_in, do: raise(ArgumentError, "received frame larger than max frame size")

    {_, gm} = set_frame_size(g, mtu * 8)
    diff = round(rate * input_size(gm)) - output_size(gm)
    {variable, gp} = set_frame_size(gm, round(nbits_in * rate) - diff)

    input =
      case input_conversion(gp) do
        "uchar" -> to_u8(soft, 48.0, shift(gp))
        _ -> Nx.add(soft, shift(gp))
      end

    size = input_size(gp)

    blocks =
      if variable do
        [input]
      else
        if rem(nbits_in, size) != 0, do: raise(ArgumentError, "bad block multiple in")
        for i <- 0..(div(nbits_in, size) - 1), do: Nx.slice_along_axis(input, i * size, size)
      end

    {outs, state} = Enum.map_reduce(blocks, state, &work(gp, &1, &2))
    bits = outs |> Enum.map(&to_binary/1) |> IO.iodata_to_binary()

    # the PDU has `round(n * rate) - diff` items whatever the decoder writes
    # (a padded terminated cc_decoder writes fewer, GNU Radio leaves the rest
    # as it was; here it is zero)
    want = if variable, do: round(nbits_in * rate) - diff, else: byte_size(bits)
    bits = if byte_size(bits) < want, do: bits <> :binary.copy(<<0>>, want - byte_size(bits)), else: bits
    bits = if packed, do: pack(bits, Keyword.get(opts, :rev_pack, true)), else: bits
    {{meta, bits}, state}
  end

  # bit packing as gr::blocks::kernel::unpack_k_bits / pack_k_bits (k = 8)
  defp unpack(bytes, false), do: ExSdr.Comm.Bits.unpack(bytes)

  defp unpack(bytes, true),
    do:
      for(<<b <- bytes>>,
        into: <<>>,
        do: for(<<(x::1 <- <<b>>)>>, into: <<>>, do: <<x>>) |> reverse_bin()
      )

  defp pack(bits, false), do: ExSdr.Comm.Bits.pack(bits)

  defp pack(bits, true) do
    for <<chunk::binary-size(8) <- bits>>, into: <<>> do
      chunk |> reverse_bin() |> ExSdr.Comm.Bits.pack()
    end
  end

  defp reverse_bin(b), do: b |> :binary.bin_to_list() |> Enum.reverse() |> :binary.list_to_bin()
end
