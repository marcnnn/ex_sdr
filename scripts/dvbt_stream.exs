# Streaming DVB-T receiver: processing time per front end chunk against the
# chunk's duration, for several chunk sizes (2k, QPSK 1/2, 9.14 Msamples/s).
#   MIX_ENV=test mix run scripts/dvbt_stream.exs [chunk_symbols ...]
# Signal as in scripts/dvbt_realtime.exs. Acquisition runs once beforehand;
# each push then carries exactly one chunk. The host is shared: the tails
# (p99, max) are noisy.
alias ExSdr.DVBT.{Channel, Receiver, Sync, TestSignal, Transmitter}

opts = [constellation: :qpsk, code_rate: {1, 2}, guard_interval: {1, 32}, cell_id: 0]

x =
  TestSignal.ts_packets(1008)
  |> Transmitter.transmit(opts)
  |> Channel.delay(3333, 3)
  |> Channel.frequency_offset(1.37)
  |> Channel.awgn(8.0, 42)

{:ok, s} = Sync.acquire(x)
x = Nx.slice(x, [s.start], [Nx.size(x) - s.start])
rate = 9_142_857.0

sizes =
  case System.argv() do
    [] -> [4, 8, 16, 32, 68]
    args -> Enum.map(args, &String.to_integer/1)
  end

run = fn chunk ->
  samples = chunk * 2112
  st = Receiver.Stream.new(Keyword.merge(opts, freq_offset: s.freq_offset, chunk_symbols: chunk))
  pieces = for p <- 0..(div(Nx.size(x), samples) - 1), do: Nx.slice(x, [p * samples], [samples])

  {times, {bytes, _}} =
    Enum.map_reduce(pieces, {0, st}, fn piece, {bytes, st} ->
      {us, {ts, st}} = :timer.tc(fn -> Receiver.Stream.push(st, piece) end)
      {us, {bytes + byte_size(ts), st}}
    end)

  {Enum.sort(times), bytes}
end

IO.puts("chunk     duration   median    p99      max      real time   TS bytes")

for chunk <- sizes do
  # first run compiles the front end for this chunk size
  run.(chunk)
  {times, bytes} = run.(chunk)
  duration = chunk * 2112 / rate * 1.0e6
  pct = fn p -> Enum.at(times, min(length(times) - 1, round(p * length(times)))) end
  ms = fn us -> :erlang.float_to_binary(us / 1000, decimals: 2) |> String.pad_leading(6) end

  IO.puts(
    "#{String.pad_leading(Integer.to_string(chunk), 3)} sym  #{ms.(duration)} ms  #{ms.(pct.(0.5))}  #{ms.(pct.(0.99))}  #{ms.(List.last(times))}   #{:erlang.float_to_binary(duration / pct.(0.5), decimals: 2)}x      #{bytes}"
  )
end
