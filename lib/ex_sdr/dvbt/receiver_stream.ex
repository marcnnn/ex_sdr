defmodule ExSdr.DVBT.Receiver.Stream do
  @moduledoc """
  Streaming DVB-T receiver: baseband samples in pieces of any size,
  transport stream packets out as soon as they are decoded. The same
  stages as `ExSdr.DVBT.Receiver.receive_sync/2`, each with its state
  carried from one piece to the next:

    * acquisition (`ExSdr.DVBT.Sync.acquire/2`): buffers samples until it
      succeeds, once per stream (no tracking yet: timing and frequency
      offset stay fixed afterwards)
    * front end (`ExSdr.Comm.OFDM.FrontEnd`): fixed chunks of
      `:chunk_symbols` OFDM symbols from the superframe start, so the
      computation is compiled once; the phase of the frequency correction
      continues across chunks
    * Viterbi: `ExSdr.Comm.Convolutional.decoder/2` with a sliding-window
      traceback of `:traceback` steps
    * byte packing, Forney deinterleaver (`ConvInterleaver.stream/5`),
      Reed-Solomon (incomplete codewords wait), descrambler (aligned once
      on the first inverted sync byte, then its position in the 8 packet
      group)

  The output does not depend on how the input is split into pieces. It is
  the batch receiver's output up to the Viterbi traceback (sliding window
  instead of one per block: a few decisions near the noise threshold may
  differ, the Reed-Solomon decoder usually hides them), continued past the
  last whole group of 8 packets.

  Latency after acquisition: one front end chunk (2112 samples per symbol,
  231 us at 9.14 Msamples/s), the traceback window, and the Forney
  deinterleaver's fixed 2244 bytes, plus processing time.
  """
  alias ExSdr.Comm.{Bits, ConvInterleaver}
  alias ExSdr.Comm.Convolutional, as: CC
  alias ExSdr.Comm.OFDM.FrontEnd
  alias ExSdr.DVBT.{Convolutional, EnergyDispersal, OuterInterleaver, Receiver, ReedSolomon, Sync}

  @sym 2112
  # samples ExSdr.DVBT.Sync.acquire/2 needs with its default options
  @acquire_samples (32 + 1) * @sym + 2 * 272 * @sym

  defstruct [
    :opts,
    :spec,
    :chunk,
    :sync,
    :viterbi,
    # compiled front end per chunk size
    front_ends: %{},
    buffer: <<>>,
    # absolute index of the first buffered sample
    position: 0,
    # samples after the superframe start that went through the front end
    processed: 0,
    bit_rest: <<>>,
    deinterleaver: nil,
    skip: 0,
    rs_rest: <<>>,
    # bytes since the start of the current 8 packet group, nil until aligned
    group_phase: nil,
    rs_corrected: 0,
    rs_failed: 0
  ]

  @doc """
  Options: `:constellation`, `:code_rate` (as the batch receiver),
  `:chunk_symbols` (front end chunk, a multiple of 4, default 8),
  `:traceback` (Viterbi traceback depth in steps, default 128), and the
  options of `ExSdr.DVBT.Sync.acquire/2`. With `:freq_offset`
  (subcarrier spacings) the synchronisation is known: the first sample
  pushed starts a superframe and there is no acquisition.
  """
  def new(opts) do
    chunk = Keyword.get(opts, :chunk_symbols, 8)
    spec = Receiver.front_end_spec(Keyword.fetch!(opts, :constellation))

    unless chunk > 0 and rem(chunk, FrontEnd.group(spec)) == 0 do
      raise ArgumentError, ":chunk_symbols must be a positive multiple of #{FrontEnd.group(spec)}"
    end

    sync =
      if Keyword.has_key?(opts, :freq_offset),
        do: %{start: 0, freq_offset: Keyword.fetch!(opts, :freq_offset)}

    %__MODULE__{
      opts: opts,
      spec: spec,
      chunk: chunk,
      sync: sync,
      viterbi:
        CC.decoder(Convolutional.code(Keyword.fetch!(opts, :code_rate)),
          start: :unknown,
          depth: Keyword.get(opts, :traceback, 128)
        ),
      deinterleaver: ConvInterleaver.stream_state(OuterInterleaver.interleaver()),
      skip: OuterInterleaver.delay()
    }
  end

  @doc "Feeds samples (complex tensor); returns `{transport stream bytes, stream}`."
  def push(%__MODULE__{} = st, samples) do
    st = %{st | buffer: st.buffer <> (samples |> Nx.as_type(:c64) |> Nx.to_binary())}
    st = if st.sync, do: st, else: acquire(st)
    if st.sync, do: run_chunks(st, st.chunk), else: {<<>>, st}
  end

  @doc """
  Ends the stream: the remaining whole groups of 4 symbols go through the
  front end and the Viterbi decoder is flushed. Returns `{bytes, stream}`.
  """
  def flush(%__MODULE__{sync: nil} = st), do: {<<>>, st}

  def flush(%__MODULE__{} = st) do
    {ts, st} = run_chunks(st, st.chunk)
    whole = div(div(byte_size(st.buffer), 8 * @sym), FrontEnd.group(st.spec))

    {rest, st} =
      if whole > 0, do: run_chunks(st, whole * FrontEnd.group(st.spec)), else: {<<>>, st}

    {bits, viterbi} = CC.flush(st.viterbi)
    {tail, st} = decode(%{st | viterbi: viterbi}, bits)
    {IO.iodata_to_binary([ts, rest, tail]), st}
  end

  defp acquire(st) do
    if byte_size(st.buffer) < 8 * @acquire_samples do
      st
    else
      case Sync.acquire(Nx.from_binary(st.buffer, :c64), st.opts) do
        {:ok, sync} ->
          start = sync.start

          %{
            st
            | sync: %{sync | start: st.position + start},
              buffer: binary_part(st.buffer, 8 * start, byte_size(st.buffer) - 8 * start),
              position: st.position + start
          }

        {:error, _} ->
          # drop one frame and try again when more samples arrived
          drop = 68 * @sym
          size = byte_size(st.buffer) - 8 * drop

          %{st | buffer: binary_part(st.buffer, 8 * drop, size), position: st.position + drop}
      end
    end
  end

  # all whole chunks of `symbols` symbols in the buffer
  defp run_chunks(st, symbols, acc \\ []) do
    samples = symbols * @sym

    if byte_size(st.buffer) < 8 * samples do
      {IO.iodata_to_binary(Enum.reverse(acc)), st}
    else
      <<chunk::binary-size(8 * samples), buffer::binary>> = st.buffer
      fe = Map.get_lazy(st.front_ends, symbols, fn -> FrontEnd.prepare(st.spec, symbols) end)
      offset = st.sync.freq_offset
      phase = FrontEnd.start_phase(st.spec, offset, st.processed)
      bits = FrontEnd.run_prepared(fe, Nx.from_binary(chunk, :c64), offset, phase)
      {decoded, viterbi} = CC.decode_stream(st.viterbi, bits)

      st = %{
        st
        | buffer: buffer,
          front_ends: Map.put(st.front_ends, symbols, fe),
          position: st.position + samples,
          processed: st.processed + samples,
          viterbi: viterbi
      }

      {ts, st} = decode(st, decoded)
      run_chunks(st, symbols, [ts | acc])
    end
  end

  # decoded bits -> bytes -> deinterleaver -> RS -> descrambler
  defp decode(st, bits) do
    bits = st.bit_rest <> bits
    whole = div(byte_size(bits), 8) * 8
    <<body::binary-size(whole), bit_rest::binary>> = bits

    {bytes, deinterleaver} =
      ConvInterleaver.stream(
        OuterInterleaver.interleaver(),
        :deinterleave,
        Bits.pack(body),
        st.deinterleaver
      )

    # the first `delay` bytes are the zeros the delay lines started with
    drop = min(st.skip, byte_size(bytes))
    bytes = st.rs_rest <> binary_part(bytes, drop, byte_size(bytes) - drop)
    whole = div(byte_size(bytes), 204) * 204
    <<codewords::binary-size(whole), rs_rest::binary>> = bytes
    {packets, corrected, failed} = ReedSolomon.decode_all(codewords)

    st = %{
      st
      | bit_rest: bit_rest,
        deinterleaver: deinterleaver,
        skip: st.skip - drop,
        rs_rest: rs_rest,
        rs_corrected: st.rs_corrected + corrected,
        rs_failed: st.rs_failed + failed
    }

    descramble(st, align(st, packets))
  end

  defp align(%{group_phase: nil}, packets) do
    start =
      Enum.find(
        0..(div(byte_size(packets), 188) - 1)//1,
        &(:binary.at(packets, &1 * 188) == 0xB8)
      )

    if start, do: {:aligned, binary_part(packets, start * 188, byte_size(packets) - start * 188)}
  end

  defp align(_, packets), do: packets

  defp descramble(st, nil), do: {<<>>, st}
  defp descramble(st, {:aligned, packets}), do: descramble(%{st | group_phase: 0}, packets)

  defp descramble(st, packets) do
    ts = EnergyDispersal.descramble_at(packets, st.group_phase)
    {ts, %{st | group_phase: rem(st.group_phase + byte_size(packets), 8 * 188)}}
  end
end
