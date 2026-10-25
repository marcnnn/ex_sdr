defmodule ExSdr.DVBT.EndToEndTest do
  @moduledoc """
  DVB-T end to end (2k, QPSK, 1/2, GI 1/32): ExSdr's transmitter, a channel
  with unknown start, carrier frequency offset and AWGN, and ExSdr's
  receiver with its own acquisition. The live test runs gr-dtv's receiver
  on the identical samples (podman image exhdl-gnuradio-dtv).

  `mix test --only dvbt` / `mix test --only dvbt_live`.
  """
  use ExUnit.Case, async: false
  use ExUnitProperties

  alias ExSdr.DVBT.{Channel, Convolutional, Inner, OFDM, Receiver, TestSignal, Transmitter}

  @moduletag timeout: 900_000

  @opts [constellation: :qpsk, code_rate: {1, 2}, guard_interval: {1, 32}, cell_id: 0]
  @packets 1008
  @delay 3333
  @cfo 1.37

  setup_all do
    ts = TestSignal.ts_packets(@packets)
    clean = Transmitter.transmit(ts, @opts)
    # C/N 8.8 dB: comfortably above both receivers' threshold
    impaired =
      clean |> Channel.delay(@delay, 3) |> Channel.frequency_offset(@cfo) |> Channel.awgn(8.0, 42)

    %{ts: ts, clean: clean, impaired: impaired}
  end

  defp assert_source(ts, out_ts) do
    assert byte_size(out_ts) >= 188 * 600, "only #{div(byte_size(out_ts), 188)} packets"
    assert {start, _} = :binary.match(ts, binary_part(out_ts, 0, 188 * 4))
    assert binary_part(ts, start, byte_size(out_ts)) == out_ts
    start
  end

  @tag :dvbt
  test "fast path makes the same hard decisions as the reference front end", %{impaired: x} do
    {:ok, sync} = ExSdr.DVBT.Sync.acquire(x)
    symbols = Nx.slice(x, [sync.start], [68 * 2112])
    shifted = Channel.frequency_offset(symbols, -sync.freq_offset)

    reference =
      shifted
      |> OFDM.demodulate(0)
      |> Enum.flat_map(fn {d, _} -> Enum.map(d, &Inner.demap_symbol(&1, :qpsk)) end)

    generic = OFDM.ofdm()
    constellation = Inner.constellation(:qpsk)

    fast =
      generic
      |> ExSdr.Comm.OFDM.carriers(symbols, 68, sync.freq_offset)
      |> then(&ExSdr.Comm.OFDM.equalise(generic, &1, 0))
      |> then(
        &ExSdr.Comm.Constellation.demap_tensor(
          constellation,
          &1,
          ExSdr.Comm.Constellation.tensor_tables(constellation)
        )
      )
      |> Nx.to_flat_list()

    # the frequency correction starts at a different sample, which only
    # rotates all carriers by a constant phase; allow a handful of decisions
    # right at a boundary to flip
    differing = Enum.zip(reference, fast) |> Enum.count(fn {a, b} -> a != b end)
    assert differing <= 5, "#{differing} of #{length(fast)} decisions differ"
  end

  @tag :dvbt
  property "native Viterbi is bit-identical to the Elixir reference" do
    check all rate <- member_of([{1, 2}, {2, 3}, {3, 4}, {5, 6}, {7, 8}]),
              start <- member_of([:zero, :unknown]),
              bits <- list_of(integer(0..1), length: 600),
              flips <- list_of(integer(0..900), max_length: 30),
              max_runs: 40 do
      coded = Convolutional.encode(bits, rate)
      flips = MapSet.new(flips)

      noisy =
        coded |> Enum.with_index() |> Enum.map(fn {b, i} -> if i in flips, do: 1 - b, else: b end)

      assert Convolutional.decode(noisy, rate, start: start) ==
               Convolutional.decode_reference(noisy, rate, start: start)
    end
  end

  @tag :dvbt
  property "SIMD Viterbi is bit-identical to the scalar kernel and the Elixir reference" do
    check all rate <- member_of([{1, 2}, {2, 3}, {3, 4}, {5, 6}, {7, 8}]),
              start <- member_of([:zero, :unknown]),
              len <- integer(1..1500),
              seed <- integer(1..10_000),
              max_runs: 40 do
      :rand.seed(:exsss, {seed, 2, 3})
      bits = for _ <- 1..len, do: :rand.uniform(2) - 1
      # a few percent bit errors; partial puncturing periods at the end
      noisy =
        bits
        |> Convolutional.encode(rate)
        |> Enum.map(fn b -> if :rand.uniform() < 0.04, do: 1 - b, else: b end)

      rx = noisy |> :binary.list_to_bin() |> Convolutional.depunctured_binary(rate)
      unknown = start == :unknown

      simd = ExSdr.NIF.viterbi(rx, 7, [0o171, 0o133], unknown, true)
      assert simd == ExSdr.NIF.viterbi(rx, 7, [0o171, 0o133], unknown, false)

      assert :binary.bin_to_list(simd) ==
               Convolutional.decode_reference(noisy, rate, start: start)
    end
  end

  @tag :dvbt
  test "native RS decoder equals the Elixir reference on DVB codewords" do
    alias ExSdr.DVBT.ReedSolomon
    :rand.seed(:exsss, {5, 6, 7})

    # 0..12 byte errors per codeword: corrected, and failed ones
    codewords =
      for _ <- 1..200 do
        cw = ReedSolomon.encode(:crypto.strong_rand_bytes(188))

        Enum.reduce(Enum.take_random(0..203, :rand.uniform(13) - 1), cw, fn p, acc ->
          <<a::binary-size(p), b, c::binary>> = acc
          <<a::binary, Bitwise.bxor(b, :rand.uniform(255)), c::binary>>
        end)
      end

    {ref, corrected, failed} =
      Enum.reduce(codewords, {[], 0, 0}, fn cw, {o, c, f} ->
        case ReedSolomon.decode(cw) do
          {:ok, p, n} -> {[p | o], c + n, f}
          {:error, _, p} -> {[p | o], c, f + 1}
        end
      end)

    assert failed > 0 and corrected > 0

    assert ReedSolomon.decode_all(IO.iodata_to_binary(codewords)) ==
             {ref |> Enum.reverse() |> IO.iodata_to_binary(), corrected, failed}
  end

  @tag :dvbt
  test "noise-free: receiver decodes the source TS from the first superframe", %{
    ts: ts,
    clean: clean
  } do
    out = Receiver.receive_sync(clean, @opts)
    assert out.sync.start == 0
    assert assert_source(ts, out.ts) == 0
    assert {out.rs_corrected, out.rs_failed} == {0, 0}
  end

  @tag :dvbt
  test "unknown start, frequency offset, AWGN (C/N 8.8 dB): acquisition and error-free TS", %{
    ts: ts,
    impaired: x
  } do
    out = Receiver.receive_sync(x, @opts)
    assert out.sync.start == @delay
    assert_in_delta out.sync.freq_offset, @cfo, 0.01
    assert out.sync.scattered_phase_ok
    assert_source(ts, out.ts)
    assert out.rs_failed == 0
  end

  @tag :dvbt
  test "streaming receiver (float front end): the batch receiver's TS, whatever the piece sizes",
       %{
         ts: ts,
         impaired: x
       } do
    batch = Receiver.receive_sync(x, @opts)

    outs =
      for sizes <- [[2112 * 8], [1, 7777, 300_001, 45_000, 2111]] do
        out = stream(x, sizes, front_end: :float)
        assert out.sync.start == batch.sync.start
        assert out.rs_failed == 0
        assert binary_part(out.ts, 0, byte_size(batch.ts)) == batch.ts
        # the stream also outputs the packets after the last whole group of 8
        assert byte_size(out.ts) - byte_size(batch.ts) < 8 * 188
        assert_source(ts, out.ts)
        out
      end

    assert Enum.uniq(outs) |> length() == 1
  end

  @tag :dvbt
  test "streaming receiver (fixed-point front end): the batch receiver's TS, whatever the piece sizes",
       %{ts: ts, impaired: x} do
    batch = Receiver.receive_sync(x, @opts)
    fixed = [front_end: :fixed, fixed_opts: [fallback: false]]

    outs =
      for sizes <- [[2112 * 8], [1, 7777, 300_001, 45_000, 2111]] do
        out = stream(x, sizes, fixed)
        assert out.sync.start == batch.sync.start
        assert out.rs_failed == 0
        # same Reed-Solomon corrected output as the float batch receiver
        assert binary_part(out.ts, 0, byte_size(batch.ts)) == batch.ts
        assert byte_size(out.ts) - byte_size(batch.ts) < 8 * 188
        assert_source(ts, out.ts)
        out
      end

    assert Enum.uniq(outs) |> length() == 1

    # sc16 input (as a radio delivers it) is the same as complex input
    # converted with the default gain
    sc16 = ExSdr.Comm.OFDM.FixedFrontEnd.to_sc16(x, 0.47)
    raw = stream(x, [1, 7777, 300_001, 45_000, 2111], fixed ++ [input: :sc16], sc16)
    assert raw.ts == hd(outs).ts
  end

  @tag :dvbt
  test "pipeline (two processes) gives the stream's TS, whatever the piece sizes", %{
    impaired: x
  } do
    fixed = [front_end: :fixed, fixed_opts: [fallback: false]]
    expected = stream(x, [2112 * 8], fixed)

    for sizes <- [[2112 * 8], [1, 7777, 300_001, 45_000, 2111]], credits <- [1, 3] do
      {out, pipe, summary} = pipeline(x, sizes, fixed ++ [credits: credits])
      assert out.ts == expected.ts
      assert summary.sync == expected.sync
      assert {summary.rs_corrected, summary.rs_failed} == {expected.rs_corrected, 0}
      # the pieces' indices are chunk starts, in order, and latency is measured
      # (the tail from the flush has none)
      indices = out.pieces |> Enum.map(& &1.index) |> Enum.reject(&is_nil/1)
      assert indices == Enum.sort(indices)
      assert Enum.all?(indices, &(rem(&1 - expected.sync.start, 2112 * 8) == 0))
      assert Enum.all?(out.pieces, &(&1.latency_us >= 0))
      # the credit bound holds (and 1 chunk in flight at least)
      stats = ExSdr.DVBT.Receiver.Pipeline.stats(pipe)
      assert %{dropped: 0, pending: 0} = stats
      assert stats.max_in_flight in 1..credits
    end

    # also with the float front end, sc16 input and the batch receiver
    batch = Receiver.receive_sync(x, @opts)
    {out, _, _} = pipeline(x, [7777, 123_457], front_end: :float)
    assert binary_part(out.ts, 0, byte_size(batch.ts)) == batch.ts
    sc16 = ExSdr.Comm.OFDM.FixedFrontEnd.to_sc16(x, 0.47)
    {raw, _, _} = pipeline(x, [7777, 123_457], fixed ++ [input: :sc16], sc16)
    assert raw.ts == expected.ts
  end

  @tag :dvbt
  test "pipeline with overflow: :drop discards and counts pushes the front cannot take", %{
    impaired: x
  } do
    alias ExSdr.DVBT.Receiver.Pipeline
    {:ok, pipe} = Pipeline.start_link(@opts ++ [overflow: :drop, max_pending: 1])
    results = for piece <- pieces(x, [div(Nx.size(x), 40)]), do: Pipeline.push(pipe, piece)
    errors = Enum.count(results, &(&1 == {:error, :overflow}))
    assert errors > 0 and errors < length(results)
    assert Pipeline.stats(pipe).dropped == errors
    assert {:ok, _} = Pipeline.flush(pipe)
  end

  # pushes x through a Pipeline in pieces, collects its output
  defp pipeline(x, sizes, extra, pushed \\ nil) do
    alias ExSdr.DVBT.Receiver.Pipeline
    pieces = pieces(pushed || x, sizes, if(pushed, do: 2, else: 1))
    {:ok, pipe} = Pipeline.start_link(@opts ++ extra)
    for piece <- pieces, do: :ok = Pipeline.push(pipe, piece)
    {:ok, summary} = Pipeline.flush(pipe)
    out = collect(pipe.ref, [])
    ts = out |> Enum.map(& &1.bytes) |> IO.iodata_to_binary()
    {%{ts: ts, pieces: out}, pipe, summary}
  end

  defp collect(ref, acc) do
    receive do
      {:ex_sdr_ts, ^ref, piece} -> collect(ref, [piece | acc])
    after
      0 -> Enum.reverse(acc)
    end
  end

  # x cut into pieces of the given sizes (cycled); sc16 input is a flat s16
  # tensor with two values per sample
  defp pieces(x, sizes, per \\ 1) do
    total = div(Nx.size(x), per)

    {pieces, _} =
      sizes
      |> Stream.cycle()
      |> Enum.reduce_while({[], 0}, fn n, {acc, pos} ->
        if pos >= total,
          do: {:halt, {acc, pos}},
          else: {:cont, {[Nx.slice(x, [pos * per], [per * min(n, total - pos)]) | acc], pos + n}}
      end)

    Enum.reverse(pieces)
  end

  # pushes x in pieces of the given sizes (cycled), then flushes
  defp stream(x, sizes, extra, pushed \\ nil) do
    pieces = pieces(pushed || x, sizes, if(pushed, do: 2, else: 1))

    st = Receiver.Stream.new(@opts ++ extra)

    {out, st} =
      pieces
      |> Enum.reduce({[], st}, fn piece, {out, st} ->
        {ts, st} = Receiver.Stream.push(st, piece)
        {[out, ts], st}
      end)

    {tail, st} = Receiver.Stream.flush(st)

    %{
      ts: IO.iodata_to_binary([out, tail]),
      sync: st.sync,
      rs_corrected: st.rs_corrected,
      rs_failed: st.rs_failed
    }
  end

  @tag :dvbt_live
  test "gr-dtv decodes the identical impaired signal to the same source TS", %{
    ts: ts,
    impaired: x
  } do
    dir = Path.join(System.tmp_dir!(), "exhdl_dvbt_live_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    File.write!(Path.join(dir, "samples.c64"), x |> Nx.as_type(:c64) |> Nx.to_binary())
    gr_dir = Path.expand("gnuradio", __DIR__)

    {log, status} =
      System.cmd(
        "podman",
        [
          "run",
          "--rm",
          "-v",
          "#{dir}:/w:Z",
          "-v",
          "#{gr_dir}:/g:ro,Z",
          "exhdl-gnuradio-dtv",
          "python3",
          "/g/dvbt_rx.py",
          "/w/samples.c64",
          "/w/gr.ts"
        ],
        stderr_to_stdout: true
      )

    assert status == 0, log
    gr = File.read!(Path.join(dir, "gr.ts"))
    File.rm_rf!(dir)

    ours = Receiver.receive_sync(x, @opts)
    gr_start = assert_source(ts, gr)
    our_start = assert_source(ts, ours.ts)
    # ExSdr starts at the first complete superframe; gr-dtv locks during it
    assert our_start <= gr_start
  end
end
