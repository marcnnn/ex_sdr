defmodule ExSdr.DVBT.Receiver.Pipeline do
  @moduledoc """
  `ExSdr.DVBT.Receiver.Stream` as two processes, like GNU Radio's thread
  per block, with bounded queues so latency cannot grow without limit:

      caller --push--> front --chunks--> back --TS--> consumer
                       (buffering, acquisition,       (Viterbi, deinterleaver,
                        front end)                     Reed-Solomon, descrambler)

  The stages are `Stream.push_front/2` and `Stream.push_back/2`, so the
  output is byte for byte that of `Stream`.

  Bounds:

    * Between the stages at most `:credits` chunks (default 2) are in
      flight. The front takes a credit per chunk and the back returns it
      when it has decoded the chunk and sent its output; with none left
      the front waits, and so does everything behind it.
    * Between caller and front, `:overflow` is the policy of `push/2`:
      `:block` (default) returns when the front has taken the samples and
      handed all their chunks to the back, so the caller is slowed to the
      pipeline's speed (use it for files and tests). `:drop` never waits:
      when `:max_pending` (default 1) pushes are already waiting for the
      front, the samples are discarded, `{:error, :overflow}` returned and
      counted in `stats/1`. For a radio, whose source cannot be slowed,
      that keeps the latency bounded. The stream then has a gap that the
      stages do not repair (no resynchronisation): the output is
      corrupt until the end, so a drop means "too slow, restart".

  Output: the consumer (`:consumer`, default the process that started the
  pipeline) gets `{:ex_sdr_ts, pipeline.ref, piece}` for each non-empty
  piece, `piece` a map:

    * `:bytes`: transport stream bytes
    * `:index`: absolute index of the first sample of the chunk whose
      decoding completed the piece. The bytes themselves come from earlier
      samples (traceback window, deinterleaver and Reed-Solomon delay,
      about 2400 bytes, plus up to a chunk of buffering), so this is an
      upper bound for a timestamp. TODO: the exact index of the first
      symbol for PTP timestamps, by carrying indices through the Viterbi
      and interleaver queues. `nil` for the piece from `flush/1`.
    * `:latency_us`: time from the arrival of the push that completed the
      chunk until the piece left the back stage

  `flush/1` ends the stream and returns `{:ok, summary}` after the last
  piece was sent (so the consumer, if it is the caller, has all pieces in
  its mailbox). Both processes then exit. They are linked to the caller.
  """
  alias ExSdr.DVBT.Receiver.Stream

  defstruct [:ref, :front, :back, :counters, :overflow, :max_pending, :per]

  # counters
  @pending 1
  @dropped 2
  @dropped_samples 3
  @in_flight_max 4

  @doc """
  Starts the pipeline. Options as for `Stream.new/1`, plus `:credits`,
  `:overflow`, `:max_pending` and `:consumer` (see the moduledoc).
  """
  def start_link(opts) do
    credits = Keyword.get(opts, :credits, 2)
    overflow = Keyword.get(opts, :overflow, :block)
    unless credits >= 1 and overflow in [:block, :drop], do: raise(ArgumentError, "bad options")

    consumer = Keyword.get(opts, :consumer, self())
    ref = make_ref()
    counters = :counters.new(4, [:atomics])
    # built here, so the front end is compiled (or comes from the cache)
    # before the processes start; the processes get it without compiled
    # front ends, which must not be copied (see `Stream.prepare_front/1`)
    st = %{Stream.new(opts) | front_ends: %{}}
    starter = self()

    back =
      spawn_stage(fn ->
        receive do
          {:front, front} -> back_loop(st, consumer, ref, front)
        end
      end)

    front =
      spawn_stage(fn ->
        st = Stream.prepare_front(st)
        send(starter, {ref, :ready})
        front_loop(st, back, credits, credits, counters)
      end)

    send(back, {:front, front})

    receive do
      {^ref, :ready} -> :ok
    end

    {:ok,
     %__MODULE__{
       ref: ref,
       front: front,
       back: back,
       counters: counters,
       overflow: overflow,
       max_pending: Keyword.get(opts, :max_pending, 1),
       per: if(Keyword.get(opts, :input) == :sc16, do: 2, else: 1)
     }}
  end

  # a long-lived stage: a big initial heap and rare full sweeps keep the
  # collector out of the steady state; large messages stay off the heap
  defp spawn_stage(fun) do
    :erlang.spawn_opt(fun, [
      :link,
      min_heap_size: 100_000,
      fullsweep_after: 20,
      message_queue_data: :off_heap
    ])
  end

  @doc """
  Feeds samples as `Stream.push/2`. Returns `:ok`, or with `overflow: :drop`
  `{:error, :overflow}` when the samples were discarded.
  """
  def push(%__MODULE__{overflow: :drop} = p, samples) do
    if :counters.get(p.counters, @pending) >= p.max_pending do
      :counters.add(p.counters, @dropped, 1)
      :counters.add(p.counters, @dropped_samples, samples_in(samples, p.per))
      {:error, :overflow}
    else
      :counters.add(p.counters, @pending, 1)
      send(p.front, {:push, nil, samples, now()})
      :ok
    end
  end

  def push(%__MODULE__{} = p, samples) do
    mon = Process.monitor(p.front)
    :counters.add(p.counters, @pending, 1)
    send(p.front, {:push, {self(), mon}, samples, now()})
    await(mon, :ok)
  end

  @doc "Ends the stream, see the moduledoc. Returns `{:ok, summary}`."
  def flush(%__MODULE__{} = p) do
    mon = Process.monitor(p.back)
    send(p.front, {:flush, {self(), mon}})
    await(mon, nil)
  end

  @doc """
  Counters: `:dropped` pushes, `:dropped_samples`, `:pending` pushes not yet
  taken by the front, `:max_in_flight` (highest number of chunks between the
  stages so far).
  """
  def stats(%__MODULE__{counters: c}) do
    %{
      dropped: :counters.get(c, @dropped),
      dropped_samples: :counters.get(c, @dropped_samples),
      pending: :counters.get(c, @pending),
      max_in_flight: :counters.get(c, @in_flight_max)
    }
  end

  defp await(mon, _) do
    receive do
      {^mon, reply} ->
        Process.demonitor(mon, [:flush])
        reply

      {:DOWN, ^mon, _, _, reason} ->
        exit({:pipeline_down, reason})
    end
  end

  # -- stages -------------------------------------------------------------------------

  defp front_loop(st, back, credits, max, counters) do
    receive do
      {:push, from, samples, t_in} ->
        st = Stream.ingest(st, samples)
        {st, credits} = dispatch(st, back, credits, max, counters, t_in)
        :counters.sub(counters, @pending, 1)
        if from, do: reply(from, :ok)
        front_loop(st, back, credits, max, counters)

      {:flush, from} ->
        {chunks, st} = Stream.flush_front(st)

        Enum.reduce(chunks, credits, fn chunk, credits ->
          send_chunk(chunk, back, credits, max, counters, now())
        end)

        send(back, {:flush, from, st.sync})
    end
  end

  # sends every whole chunk in the buffer, one front end run at a time
  defp dispatch(st, back, credits, max, counters, t_in) do
    case Stream.next_chunk(st) do
      {:ok, chunk, st} ->
        credits = send_chunk(chunk, back, credits, max, counters, t_in)
        dispatch(st, back, credits, max, counters, t_in)

      {:none, st} ->
        {st, credits}
    end
  end

  defp send_chunk(chunk, back, credits, max, counters, t_in) do
    credits = take_credit(credits) - 1
    send(back, {:chunk, chunk, t_in})
    in_flight = max - credits

    if in_flight > :counters.get(counters, @in_flight_max),
      do: :counters.put(counters, @in_flight_max, in_flight)

    credits
  end

  # a credit per chunk; waits for the back when none is left
  defp take_credit(credits) do
    receive do
      :credit -> take_credit(credits + 1)
    after
      if(credits > 0, do: 0, else: :infinity) -> credits
    end
  end

  defp reply({pid, mon}, reply), do: send(pid, {mon, reply})

  defp back_loop(st, consumer, ref, front) do
    receive do
      {:chunk, {index, bits}, t_in} ->
        {ts, st} = Stream.push_back(st, bits)
        emit(consumer, ref, ts, index, t_in)
        send(front, :credit)
        back_loop(st, consumer, ref, front)

      {:flush, from, sync} ->
        {tail, st} = if sync, do: Stream.flush_back(st), else: {<<>>, st}
        emit(consumer, ref, tail, nil, now())
        reply(from, {:ok, %{sync: sync, rs_corrected: st.rs_corrected, rs_failed: st.rs_failed}})
    end
  end

  defp emit(_, _, <<>>, _, _), do: :ok

  defp emit(consumer, ref, ts, index, t_in),
    do: send(consumer, {:ex_sdr_ts, ref, %{bytes: ts, index: index, latency_us: now() - t_in}})

  defp now, do: System.monotonic_time(:microsecond)

  # samples in a pushed tensor (`per` values each) or sc16 binary
  defp samples_in(x, _) when is_binary(x), do: div(byte_size(x), 4)
  defp samples_in(x, per), do: div(Nx.size(x), per)
end
