# Streaming DVB-T receiver (2k, QPSK 1/2, 9.14 Msamples/s), for several
# chunk sizes and both front ends (fixed-point on sc16, float on c64):
#   * Stream (one process): setup (`new/1`, cold = first of its kind in the
#     VM, warm = front end from the cache), the first push, and the
#     steady state (median, p99, max) of the processing time per chunk
#   * Pipeline (front and back stage in two processes, `:block` policy): wall
#     time for the whole signal against its duration, and the latency from
#     the arrival of a push to the TS piece it completes
#   MIX_ENV=test mix run scripts/dvbt_stream.exs [chunk_symbols ...]
# Signal as in scripts/dvbt_realtime.exs. Each push carries exactly one
# chunk. The host is shared: the tails (p99, max) are noisy. Scheduler
# busy-wait flags reduced the p99 in profiling; compare
#   ELIXIR_ERL_OPTIONS="+sbwt very_long +sbwtdcpu very_long +sbwtdio very_long +swt very_low" \\
#     MIX_ENV=test mix run scripts/dvbt_stream.exs
# with the plain run (the flags in effect are printed first).
alias ExSdr.DVBT.{Channel, Receiver, Sync, TestSignal, Transmitter}
alias ExSdr.DVBT.Receiver.Pipeline

opts = [constellation: :qpsk, code_rate: {1, 2}, guard_interval: {1, 32}, cell_id: 0]

IO.puts("ELIXIR_ERL_OPTIONS=#{System.get_env("ELIXIR_ERL_OPTIONS")}")

IO.puts(
  "schedulers=#{:erlang.system_info(:schedulers_online)} emulator args: " <>
    inspect(Keyword.drop(:init.get_arguments(), [:root, :bindir, :progname, :home, :pa, :pz]))
)

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

# the input of each front end, converted once outside the timing
sc16 = ExSdr.Comm.OFDM.FixedFrontEnd.to_sc16(x, 0.47)

stream_opts = fn chunk, fe ->
  Keyword.merge(opts,
    freq_offset: s.freq_offset,
    chunk_symbols: chunk,
    front_end: fe,
    input: if(fe == :fixed, do: :sc16, else: :complex)
  )
end

# the signal in pieces of exactly one chunk
pieces = fn chunk, fe ->
  samples = chunk * 2112
  {src, per} = if fe == :fixed, do: {sc16, 2}, else: {x, 1}

  for p <- 0..(div(Nx.size(x), samples) - 1),
      do: Nx.slice(src, [p * samples * per], [samples * per])
end

ms = fn us -> :erlang.float_to_binary(us / 1000, decimals: 2) |> String.pad_leading(7) end
pct = fn sorted, p -> Enum.at(sorted, min(length(sorted) - 1, round(p * length(sorted)))) end

stats = fn sorted ->
  "#{ms.(pct.(sorted, 0.5))} #{ms.(pct.(sorted, 0.99))} #{ms.(List.last(sorted))}"
end

run_stream = fn chunk, fe ->
  {setup, st} = :timer.tc(fn -> Receiver.Stream.new(stream_opts.(chunk, fe)) end)

  {times, {bytes, _}} =
    Enum.map_reduce(pieces.(chunk, fe), {0, st}, fn piece, {bytes, st} ->
      {us, {ts, st}} = :timer.tc(fn -> Receiver.Stream.push(st, piece) end)
      {us, {bytes + byte_size(ts), st}}
    end)

  [first | steady] = times
  {setup, first, Enum.sort(steady), bytes}
end

IO.puts("\nStream, one process (ms per chunk push)")

IO.puts(
  "front end  chunk   duration  setup cold  warm   first push     median     p99     max   real time  TS bytes"
)

for chunk <- sizes, fe <- [:fixed, :float] do
  # cold: the first Stream of its kind in this VM compiles the front end
  {cold, _, _, _} = run_stream.(chunk, fe)
  {warm, first, steady, bytes} = run_stream.(chunk, fe)
  duration = chunk * 2112 / rate * 1.0e6

  IO.puts(
    "#{String.pad_trailing(to_string(fe), 9)}  #{String.pad_leading(Integer.to_string(chunk), 3)} sym #{ms.(duration)}  #{ms.(cold)} #{ms.(warm)} #{ms.(first)}  #{stats.(steady)}  #{:erlang.float_to_binary(duration / pct.(steady, 0.5), decimals: 2)}x   #{bytes}"
  )
end

defmodule Collect do
  def run(ref, acc \\ []) do
    receive do
      {:ex_sdr_ts, ^ref, piece} -> run(ref, [piece | acc])
    after
      0 -> Enum.reverse(acc)
    end
  end
end

run_pipeline = fn chunk, fe ->
  {setup, {:ok, pipe}} =
    :timer.tc(fn -> Pipeline.start_link(stream_opts.(chunk, fe)) end)

  ps = pieces.(chunk, fe)
  t0 = System.monotonic_time(:microsecond)
  for piece <- ps, do: :ok = Pipeline.push(pipe, piece)
  {:ok, _} = Pipeline.flush(pipe)
  wall = System.monotonic_time(:microsecond) - t0
  out = Collect.run(pipe.ref)
  # the first piece carries the start-up
  lat = out |> Enum.drop(1) |> Enum.map(& &1.latency_us) |> Enum.sort()
  bytes = out |> Enum.map(&byte_size(&1.bytes)) |> Enum.sum()
  {setup, wall, lat, bytes, Pipeline.stats(pipe).max_in_flight}
end

IO.puts("\nPipeline, two processes, credits 2 (setup = start_link; latency = push to TS piece)")

IO.puts(
  "front end  chunk   duration  setup  wall/signal  real time   latency median     p99     max   in flight  TS bytes"
)

for chunk <- sizes, fe <- [:fixed, :float] do
  # one pipeline first for the cold front end, then two measured, best kept
  run_pipeline.(chunk, fe)

  {setup, wall, lat, bytes, in_flight} =
    [run_pipeline.(chunk, fe), run_pipeline.(chunk, fe)] |> Enum.min_by(&elem(&1, 1))

  signal = div(Nx.size(x), 2112) * 2112 / rate * 1.0e6

  IO.puts(
    "#{String.pad_trailing(to_string(fe), 9)}  #{String.pad_leading(Integer.to_string(chunk), 3)} sym #{ms.(chunk * 2112 / rate * 1.0e6)}  #{ms.(setup)}  #{ms.(wall)}/#{ms.(signal)}  #{:erlang.float_to_binary(signal / wall, decimals: 2)}x      #{stats.(lat)}   #{in_flight}  #{bytes}"
  )
end
