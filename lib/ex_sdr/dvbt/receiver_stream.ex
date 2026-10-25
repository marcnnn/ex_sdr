defmodule ExSdr.DVBT.Receiver.Stream do
  @moduledoc """
  Streaming DVB-T receiver: baseband samples in pieces of any size,
  transport stream packets out as soon as they are decoded. The same
  stages as `ExSdr.DVBT.Receiver.receive_sync/2`, each with its state
  carried from one piece to the next:

    * acquisition (`ExSdr.DVBT.Sync.acquire/2`): buffers samples until it
      succeeds, once per stream (no tracking yet: timing and frequency
      offset stay fixed afterwards)
    * front end: fixed chunks of `:chunk_symbols` OFDM symbols from the
      superframe start, so the computation is compiled once; the phase of
      the frequency correction continues across chunks (`:fixed`, the
      default when nx_simd is available: `ExSdr.Comm.OFDM.FixedFrontEnd`,
      integer arithmetic on `sc16`; `:float`: `ExSdr.Comm.OFDM.FrontEnd`,
      EXLA)
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

  Input: `:input` selects what `push/2` takes. With `:complex` (default)
  a complex tensor; with `:sc16` interleaved s16 I/Q as a binary or an s16
  tensor, which is what radios (e.g. a B210) deliver, full scale = 1.0.
  The buffer holds the samples in the front end's own format (`sc16` for
  `:fixed`, `c64` for `:float`), converted on entry: complex to `sc16`
  with `:gain` (default 0.47: the s16 range is saturated beyond it, and the
  fixed-point front end wants the signal well inside it), `sc16` to complex
  for `:float` and for acquisition divided by 32768 times `:gain`, so the
  scale is the same as for complex input. Use `input: :sc16` with a
  radio's samples and `:gain` 1.0 when they are already scaled.

  Compiled front ends are cached across streams (`:persistent_term`, keyed
  on the spec, the chunk size and the front end options), and `new/1`
  prepares the one for `:chunk_symbols` eagerly, so no push pays for it.

  Two stages, each a pure function of its state, compose into `push/2`
  and `flush/1`: the front stage (`push_front/2`, `flush_front/1`:
  buffering, acquisition, front end; yields one `{index, code bits}` per
  chunk, `index` being the absolute index of the chunk's first sample) and
  the back stage (`push_back/2`, `flush_back/1`: Viterbi, bytes,
  deinterleaver, Reed-Solomon, descrambler). A state used for one stage
  leaves the other stage's fields alone, so `ExSdr.DVBT.Receiver.Pipeline`
  runs them in two processes.

  Latency after acquisition: one front end chunk (2112 samples per symbol,
  231 us at 9.14 Msamples/s), the traceback window, and the Forney
  deinterleaver's fixed 2244 bytes, plus processing time.
  """
  alias ExSdr.Comm.{Bits, ConvInterleaver}
  alias ExSdr.Comm.Convolutional, as: CC
  alias ExSdr.Comm.OFDM.{FixedFrontEnd, FrontEnd}
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
    # :fixed or :float
    :front_end,
    :input,
    :gain,
    # bytes per buffered sample: 4 (sc16) or 8 (c64)
    :bps,
    # compiled front end per chunk size
    front_ends: %{},
    # share compiled front ends between streams
    cache: true,
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
  options of `ExSdr.DVBT.Sync.acquire/2`. `:front_end` is `:fixed` or
  `:float` (default `:fixed` if nx_simd is available, else `:float`);
  `:fixed_opts` go to `ExSdr.Comm.OFDM.FixedFrontEnd.prepare/3` (`:format`,
  `:shifts`, `:compiler`, ...); `:input` (`:complex` or `:sc16`) and `:gain`
  as in the moduledoc. With `:freq_offset`
  (subcarrier spacings) the synchronisation is known: the first sample
  pushed starts a superframe and there is no acquisition. `:prepare`
  (default true) compiles the front end for the chunk size in `new/1`;
  `:cache` (default true) shares compiled front ends between streams.
  """
  def new(opts) do
    chunk = Keyword.get(opts, :chunk_symbols, 8)
    spec = Receiver.front_end_spec(Keyword.fetch!(opts, :constellation))

    unless chunk > 0 and rem(chunk, FrontEnd.group(spec)) == 0 do
      raise ArgumentError, ":chunk_symbols must be a positive multiple of #{FrontEnd.group(spec)}"
    end

    front_end =
      Keyword.get_lazy(opts, :front_end, fn ->
        if Code.ensure_loaded?(NxSimd) and NxSimd.available?(), do: :fixed, else: :float
      end)

    input = Keyword.get(opts, :input, :complex)

    unless front_end in [:fixed, :float] and input in [:complex, :sc16] do
      raise ArgumentError, ":front_end is :fixed or :float, :input is :complex or :sc16"
    end

    sync =
      if Keyword.has_key?(opts, :freq_offset),
        do: %{start: 0, freq_offset: Keyword.fetch!(opts, :freq_offset)}

    st = %__MODULE__{
      cache: Keyword.get(opts, :cache, true),
      opts: opts,
      spec: spec,
      chunk: chunk,
      sync: sync,
      front_end: front_end,
      input: input,
      gain: Keyword.get(opts, :gain, 0.47),
      bps: if(front_end == :fixed, do: 4, else: 8),
      viterbi:
        CC.decoder(Convolutional.code(Keyword.fetch!(opts, :code_rate)),
          start: :unknown,
          depth: Keyword.get(opts, :traceback, 128)
        ),
      deinterleaver: ConvInterleaver.stream_state(OuterInterleaver.interleaver()),
      skip: OuterInterleaver.delay()
    }

    if Keyword.get(opts, :prepare, true), do: elem(front_end_for(st, chunk), 1), else: st
  end

  @doc """
  Loads the front end for the chunk size into the state (compiled once per
  VM, see the moduledoc): what `new/1` does, for a state that came without
  it. A state holding compiled front ends must not be sent to another
  process (spawn, message): copying one loses the sharing inside the
  compiled graph and can take minutes. `ExSdr.DVBT.Receiver.Pipeline`
  strips them and calls this in the stage's process.
  """
  def prepare_front(%__MODULE__{} = st), do: elem(front_end_for(st, st.chunk), 1)

  @doc "Drops the cached compiled front ends."
  def clear_cache do
    for {{__MODULE__, :front_end, _, _} = key, _} <- :persistent_term.get(),
        do: :persistent_term.erase(key)

    :ok
  end

  @doc """
  Feeds samples (a complex tensor, or `sc16` with `input: :sc16`); returns
  `{transport stream bytes, stream}`.
  """
  def push(%__MODULE__{} = st, samples) do
    {chunks, st} = push_front(st, samples)
    push_back(st, chunks)
  end

  @doc """
  Ends the stream: the remaining whole groups of 4 symbols go through the
  front end and the Viterbi decoder is flushed. Returns `{bytes, stream}`.
  """
  def flush(%__MODULE__{sync: nil} = st), do: {<<>>, st}

  def flush(%__MODULE__{} = st) do
    {chunks, st} = flush_front(st)
    {ts, st} = push_back(st, chunks)
    {tail, st} = flush_back(st)
    {IO.iodata_to_binary([ts, tail]), st}
  end

  @doc """
  Front stage: buffers `samples`, acquires if needed, and runs every whole
  chunk through the front end. Returns `{[{index, code_bits}], stream}`,
  `index` the absolute sample index (in the pushed samples, from 0) of the
  chunk's first sample.
  """
  def push_front(%__MODULE__{} = st, samples), do: next_chunks(ingest(st, samples), [])

  @doc """
  Front stage at the end of the stream: the chunks of the remaining whole
  groups of 4 symbols, as in `push_front/2`.
  """
  def flush_front(%__MODULE__{sync: nil} = st), do: {[], st}

  def flush_front(%__MODULE__{} = st) do
    {chunks, st} = next_chunks(st, [])
    whole = div(div(byte_size(st.buffer), st.bps * @sym), FrontEnd.group(st.spec))

    if whole > 0 do
      {rest, st} = next_chunks(st, [], whole * FrontEnd.group(st.spec))
      {chunks ++ rest, st}
    else
      {chunks, st}
    end
  end

  @doc """
  Front stage, first half of `push_front/2`: converts `samples` into the
  buffer and acquires. Follow with `next_chunk/1` until `:none`.
  """
  def ingest(%__MODULE__{} = st, samples) do
    new = to_buffer(st, samples)
    # no copy when nothing is waiting
    buffer = if st.buffer == <<>>, do: new, else: st.buffer <> new
    st = %{st | buffer: buffer}
    if st.sync, do: st, else: acquire(st)
  end

  @doc """
  Front stage, second half: the next whole chunk in the buffer through the
  front end, `{:ok, {index, code_bits}, stream}`, or `{:none, stream}`.
  """
  def next_chunk(%__MODULE__{sync: nil} = st), do: {:none, st}
  def next_chunk(%__MODULE__{} = st), do: next_chunk(st, st.chunk)

  @doc """
  Back stage: code bits of one chunk, or a list of `{index, code_bits}`,
  through Viterbi, deinterleaver, Reed-Solomon and descrambler. Returns
  `{bytes, stream}`.
  """
  def push_back(%__MODULE__{} = st, chunks) when is_list(chunks) do
    {ts, st} =
      Enum.reduce(chunks, {[], st}, fn {_, bits}, {acc, st} ->
        {ts, st} = push_back(st, bits)
        {[acc, ts], st}
      end)

    {IO.iodata_to_binary(ts), st}
  end

  def push_back(%__MODULE__{} = st, bits) when is_binary(bits) do
    {decoded, viterbi} = CC.decode_stream(st.viterbi, bits)
    decode(%{st | viterbi: viterbi}, decoded)
  end

  @doc "Back stage at the end of the stream: flushes the Viterbi decoder. Returns `{bytes, stream}`."
  def flush_back(%__MODULE__{} = st) do
    {bits, viterbi} = CC.flush(st.viterbi)
    decode(%{st | viterbi: viterbi}, bits)
  end

  # input -> the buffer's format
  defp to_buffer(%{input: :complex, front_end: :float}, x),
    do: x |> Nx.as_type(:c64) |> Nx.to_binary()

  defp to_buffer(%{input: :complex, front_end: :fixed, gain: gain}, x),
    do: x |> Nx.as_type(:c64) |> FixedFrontEnd.to_sc16(gain) |> Nx.to_binary()

  defp to_buffer(%{input: :sc16, front_end: :fixed}, x), do: sc16_binary(x)

  defp to_buffer(%{input: :sc16, front_end: :float, gain: gain}, x),
    do: x |> sc16_binary() |> to_complex(gain) |> Nx.to_binary()

  defp sc16_binary(x) when is_binary(x), do: x
  defp sc16_binary(x), do: x |> Nx.as_type(:s16) |> Nx.to_binary()

  # sc16 binary -> complex tensor, scale 1 / (32768 * gain)
  defp to_complex(binary, gain) do
    pairs = binary |> Nx.from_binary(:s16) |> Nx.reshape({:auto, 2})
    scale = 1 / (32768 * gain)
    Nx.complex(Nx.multiply(pairs[[.., 0]], scale), Nx.multiply(pairs[[.., 1]], scale))
  end

  defp buffer_to_complex(%{front_end: :float, buffer: buffer}), do: Nx.from_binary(buffer, :c64)
  defp buffer_to_complex(%{buffer: buffer, gain: gain}), do: to_complex(buffer, gain)

  defp acquire(st) do
    if byte_size(st.buffer) < st.bps * @acquire_samples do
      st
    else
      case Sync.acquire(buffer_to_complex(st), st.opts) do
        {:ok, sync} ->
          start = sync.start

          %{
            st
            | sync: %{sync | start: st.position + start},
              buffer:
                binary_part(st.buffer, st.bps * start, byte_size(st.buffer) - st.bps * start),
              position: st.position + start
          }

        {:error, _} ->
          # drop one frame and try again when more samples arrived
          drop = 68 * @sym
          size = byte_size(st.buffer) - st.bps * drop

          %{
            st
            | buffer: binary_part(st.buffer, st.bps * drop, size),
              position: st.position + drop
          }
      end
    end
  end

  defp next_chunks(st, acc), do: next_chunks(st, acc, st.chunk)

  # all whole chunks of `symbols` symbols in the buffer
  defp next_chunks(%{sync: nil} = st, acc, _), do: {Enum.reverse(acc), st}

  defp next_chunks(st, acc, symbols) do
    case next_chunk(st, symbols) do
      {:ok, chunk, st} -> next_chunks(st, [chunk | acc], symbols)
      {:none, st} -> {Enum.reverse(acc), st}
    end
  end

  defp next_chunk(st, symbols) do
    samples = symbols * @sym

    if byte_size(st.buffer) < st.bps * samples do
      {:none, st}
    else
      <<chunk::binary-size(st.bps * samples), buffer::binary>> = st.buffer
      {fe, st} = front_end_for(st, symbols)
      bits = run_front_end(st, fe, chunk)
      index = st.position

      st = %{
        st
        | buffer: buffer,
          position: st.position + samples,
          processed: st.processed + samples
      }

      {:ok, {index, bits}, st}
    end
  end

  defp front_end_for(st, symbols) do
    case st.front_ends do
      %{^symbols => fe} ->
        {fe, st}

      _ ->
        fe = if st.cache, do: cached(st, symbols), else: prepare(st, symbols)
        {fe, %{st | front_ends: Map.put(st.front_ends, symbols, fe)}}
    end
  end

  # one compiled front end per {spec, symbols, options} in the VM; the lock
  # keeps concurrent streams from compiling the same one twice
  defp cached(st, symbols) do
    fixed_opts = if st.front_end == :fixed, do: Keyword.get(st.opts, :fixed_opts, [])
    digest = :erlang.md5(:erlang.term_to_binary({st.spec, symbols, fixed_opts}))
    key = {__MODULE__, :front_end, st.front_end, digest}

    case :persistent_term.get(key, nil) do
      nil ->
        :global.trans({key, self()}, fn ->
          with nil <- :persistent_term.get(key, nil) do
            fe = prepare(st, symbols)
            :persistent_term.put(key, fe)
            fe
          end
        end)

      fe ->
        fe
    end
  end

  defp prepare(%{front_end: :fixed} = st, symbols),
    do: FixedFrontEnd.prepare(st.spec, symbols, Keyword.get(st.opts, :fixed_opts, []))

  defp prepare(st, symbols), do: FrontEnd.prepare(st.spec, symbols)

  # the phase of the frequency correction continues at `processed`
  defp run_front_end(%{front_end: :fixed} = st, fe, chunk),
    do: FixedFrontEnd.run(fe, chunk, st.sync.freq_offset, st.processed)

  defp run_front_end(st, fe, chunk) do
    offset = st.sync.freq_offset
    phase = FrontEnd.start_phase(st.spec, offset, st.processed)
    FrontEnd.run_prepared(fe, Nx.from_binary(chunk, :c64), offset, phase)
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
