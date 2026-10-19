defmodule ExSdr.FEC.Conv do
  @moduledoc """
  GNU Radio's `fec.cc_encoder` and `fec.cc_decoder` (rate 1/n convolutional
  codes, four frame modes) as `ExSdr.FEC.Generic` objects.

  Conventions of GNU Radio, kept exactly:

    * `polys` are decimal masks over the K-bit register, bit 0 is the newest
      input; negative inverts the output. K = 7, `[79, 109]` is the classic
      NASA / CCSDS code (`0o171`, `0o133` reversed); `[109, 79]` swaps the outputs.
    * `start_state` is the register content before the first bit, its bit
      K - 2 is the oldest input.
    * output per input bit: the `rate` code bits in polynomial order.

  Modes:

    * `:streaming` - frames follow each other, the register is carried
      (encoder state), the decoder needs `rate * (K - 1)` items of look-ahead
      (history) and chains its start state from the previous frame.
    * `:terminated` - every frame starts in `start_state` and is followed by
      K - 1 tail bits that drive the register back to it; the code frame has
      `rate * (F + K - 1)` bits, padded with zero bits to a whole number of
      bytes if `padded: true` (GNU Radio leaves them unwritten).
    * `:truncated` - every frame starts in `start_state`, no tail.
    * `:tailbiting` - the register starts with the last K - 1 bits of the
      frame, so it ends where it began (`start_state` is ignored); the
      decoder wraps the frame around 6 * (K - 1) steps (needs `F >= K - 1`).

  The decoder is a soft Viterbi decoder in GNU Radio's 8 bit format
  (`ExSdr.FEC.ConvViterbi`): input one `u8` per code bit, 0 sure 0, 255 sure
  1, 128 no information (punctured). GNU Radio supports only K = 7, rate 2;
  here any K and rate work (the integer metrics are unchanged); results for
  K = 7, rate 2 equal GNU Radio's. The decoder state is the chained start state
  (`-1` before the first frame of a stream: all states equally likely).

  Differences from GNU Radio: the frame modes are the same; the padding bits
  of a padded terminated frame are zeros; the 8 bit conversion from floats is
  `ExSdr.FEC.Generic.to_u8/3`.
  """
  @behaviour ExSdr.FEC.Generic

  import Bitwise

  alias ExSdr.FEC.ConvViterbi, as: V
  alias ExSdr.FEC.Generic

  @modes [:streaming, :terminated, :truncated, :tailbiting]

  @doc """
  An encoder object. Options: `:frame_size`, `:k`, `:polys` (`:rate`
  defaults to their number), `:start_state` (0), `:mode` (`:streaming`),
  `:padded` (false).
  """
  def encoder(opts), do: make(:encoder, opts)

  @doc """
  A decoder object. Options as `encoder/1` plus `:end_state` (-1: the best
  final state; only used by `:terminated`).
  """
  def decoder(opts), do: make(:decoder, opts)

  defp make(kind, opts) do
    k = Keyword.fetch!(opts, :k)
    polys = Keyword.fetch!(opts, :polys)
    rate = Keyword.get(opts, :rate, length(polys))
    frame_size = Keyword.fetch!(opts, :frame_size)
    mode = Keyword.get(opts, :mode, :streaming)
    start_state = Keyword.get(opts, :start_state, 0)
    padded = Keyword.get(opts, :padded, false)

    cond do
      rate != length(polys) ->
        raise ArgumentError, "number of polynomials must be the same as the value of rate"

      rate < 2 ->
        raise ArgumentError, "rate must be >= 2"

      k not in 2..31 ->
        raise ArgumentError, "constraint length K must be in the range 2..31"

      start_state >= 1 <<< (k - 1) or start_state < 0 ->
        raise ArgumentError, "start state must be in 0..2^(K-1)-1"

      frame_size < 1 ->
        raise ArgumentError, "frame_size must be > 0"

      mode not in @modes ->
        raise ArgumentError, "mode must be one of #{inspect(@modes)}"

      Enum.any?(polys, &(abs(&1) >= 1 <<< k or &1 == 0)) ->
        raise ArgumentError, "polynomials must be nonzero and fit the K-bit register"

      true ->
        :ok
    end

    padding =
      if padded and mode == :terminated,
        do: div(rate * (k - 1) + 7, 8) * 8 - rate * (k - 1),
        else: 0

    conf = %{
      k: k,
      rate: rate,
      polys: polys,
      mode: mode,
      start_state: start_state,
      end_state: Keyword.get(opts, :end_state, -1),
      padded: padded,
      padding: padding
    }

    name = if kind == :encoder, do: "cc_encoder", else: "cc_decoder"
    Generic.new(__MODULE__, name, kind, conf, frame_size)
  end

  # -- sizes -------------------------------------------------------------------------------

  @impl true
  def input_size(%{kind: :encoder, frame_size: f}), do: f

  def input_size(%{kind: :decoder, frame_size: f, conf: %{mode: :terminated} = c}),
    do: c.rate * (f + c.k - 1) + c.padding

  def input_size(%{kind: :decoder, frame_size: f, conf: c}), do: c.rate * f

  @impl true
  def output_size(%{kind: :decoder, frame_size: f}), do: f

  def output_size(%{kind: :encoder, frame_size: f, conf: %{mode: :terminated} = c}),
    do: c.rate * (f + c.k - 1) + c.padding

  def output_size(%{kind: :encoder, frame_size: f, conf: c}), do: c.rate * f

  @impl true
  def rate(%{kind: :encoder, conf: c}), do: c.rate * 1.0
  def rate(%{kind: :decoder, conf: c}), do: 1.0 / c.rate

  @impl true
  def history(%{kind: :decoder, conf: %{mode: :streaming} = c}), do: c.rate * (c.k - 1)
  def history(_), do: 0

  @impl true
  def shift(%{kind: :decoder}), do: 128.0
  def shift(_), do: 0.0

  @impl true
  def input_item(_), do: :u8

  @impl true
  def input_conversion(%{kind: :decoder}), do: "uchar"
  def input_conversion(_), do: "none"

  @impl true
  def output_conversion(_), do: "none"

  @impl true
  def hardware(%{kind: :encoder}), do: :feed_forward
  def hardware(%{kind: :decoder}), do: :memory

  @impl true
  def latency(%{kind: :decoder, frame_size: f, conf: %{mode: :streaming}}), do: f
  def latency(_), do: 0

  # GNU Radio's decoder removes the padding bits from the frame size here
  @impl true
  def set_frame_size(%{kind: :decoder, conf: %{mode: :terminated} = c} = g, n) do
    {ok, g} = Generic.default_set_frame_size(g, n)
    {ok, %{g | frame_size: g.frame_size - c.padding * c.rate}}
  end

  def set_frame_size(g, n), do: Generic.default_set_frame_size(g, n)

  @impl true
  def init(%{kind: :encoder, conf: c}), do: Nx.tensor(c.start_state, type: :u32)
  def init(%{kind: :decoder, conf: c}), do: Nx.tensor(decoder_start(c), type: :s32)

  defp decoder_start(%{mode: :streaming}), do: -1
  defp decoder_start(%{start_state: s}), do: s

  # -- encoder --------------------------------------------------------------------------------

  @impl true
  def work(%{kind: :encoder} = g, input, state) do
    %{k: k, rate: rate, polys: polys, mode: mode} = g.conf
    f = g.frame_size
    st = Nx.to_number(state)

    bits =
      input |> Nx.as_type(:u8) |> Nx.bitwise_and(1) |> Nx.slice_along_axis(0, f)

    prefix =
      case mode do
        :tailbiting -> Nx.slice_along_axis(bits, f - (k - 1), k - 1)
        _ -> Nx.tensor(state_bits(st, k), type: :u8)
      end

    body = Nx.concatenate([prefix, bits])

    {seq, steps} =
      if mode == :terminated,
        do: {Nx.concatenate([body, Nx.tensor(state_bits(st, k), type: :u8)]), f + k - 1},
        else: {body, f}

    # register at step t: seq[t .. t + K - 1], bit m is seq[t + K - 1 - m]
    cols =
      for p <- polys do
        taps = for m <- 0..(k - 1), (abs(p) >>> m &&& 1) == 1, do: m

        taps
        |> Enum.map(&Nx.slice_along_axis(seq, k - 1 - &1, steps))
        |> Enum.reduce(&Nx.add/2)
        |> Nx.bitwise_and(1)
        |> then(&if(p < 0, do: Nx.bitwise_xor(&1, 1), else: &1))
      end

    out = cols |> Nx.stack(axis: 1) |> Nx.reshape({steps * rate})

    out =
      if g.conf.padding > 0,
        do: Nx.concatenate([out, Nx.broadcast(Nx.tensor(0, type: :u8), {g.conf.padding})]),
        else: out

    next =
      case mode do
        :streaming -> last_state(body, k)
        :tailbiting -> last_state(bits, k)
        _ -> st
      end

    {out, Nx.tensor(next, type: :u32)}
  end

  # -- decoder --------------------------------------------------------------------------------

  def work(%{kind: :decoder} = g, input, state) do
    %{k: k, rate: rate, polys: polys, mode: mode} = g.conf
    f = g.frame_size
    k1 = k - 1
    t = V.tables(k, polys)
    st = Nx.to_number(state)

    veclen =
      case mode do
        :tailbiting -> f + 6 * k1
        :truncated -> f
        _ -> f + k1
      end

    all = Nx.to_flat_list(input)

    syms =
      case mode do
        :tailbiting -> all |> Enum.take(rate * f) |> wrap(rate * veclen)
        _ -> Enum.take(all, rate * veclen)
      end

    init =
      case {mode, st} do
        {:tailbiting, _} -> V.init_metrics(t, :unbiased, 31)
        {:streaming, -1} -> V.init_metrics(t, :unbiased, 31)
        {:streaming, s} -> V.init_metrics(t, {:state, s}, 63)
        _ -> V.init_metrics(t, {:state, g.conf.start_state}, 63)
      end

    {decs, metrics} = V.run(t, syms, init)
    best = V.best_state(metrics)

    {bits, chain} =
      case mode do
        :tailbiting ->
          {bits, _} = V.traceback(t, decs, best, veclen - 1, veclen - f)
          # step t gives input t - K1 (modulo the frame)
          ordered =
            bits
            |> Enum.with_index(veclen - f)
            |> Enum.map(fn {b, step} -> {Integer.mod(step - k1, f), b} end)
            |> Enum.sort()
            |> Enum.map(&elem(&1, 1))

          {ordered, st}

        :truncated ->
          {head, _} = V.traceback(t, decs, best, f - 1, k1)
          tail = for i <- (k1 - 1)..0//-1, do: best >>> i &&& 1
          {head ++ tail, st}

        _ ->
          end_state =
            if mode == :terminated and g.conf.end_state != -1,
              do: g.conf.end_state &&& t.states - 1,
              else: best

          {tail_bits, chain} = V.traceback(t, decs, end_state, veclen - 1, f)
          {head, _} = V.traceback(t, decs, chain, f - 1, k1)
          {head ++ tail_bits, if(mode == :streaming, do: chain, else: st)}
      end

    {Nx.tensor(bits, type: :u8), Nx.tensor(chain, type: :s32)}
  end

  defp state_bits(st, k), do: for(i <- 0..(k - 2), do: st >>> (k - 2 - i) &&& 1)

  defp last_state(seq, k) do
    n = Nx.axis_size(seq, 0)

    seq
    |> Nx.slice_along_axis(n - (k - 1), k - 1)
    |> Nx.to_flat_list()
    |> Enum.reduce(0, fn b, acc -> acc <<< 1 ||| b end)
  end

  # the first `n` items of the frame repeated as often as needed
  defp wrap(syms, n) do
    len = length(syms)
    tuple = List.to_tuple(syms)
    for i <- 0..(n - 1), do: elem(tuple, rem(i, len))
  end
end
