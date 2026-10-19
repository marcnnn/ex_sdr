defmodule ExSdr.FEC.LDPC.Kernels do
  @moduledoc """
  Tensor kernels of the LDPC blocks (`:frame` functions
  `fun(items, state, opts) -> {outputs, state}`), batched over frames.

  Inside, a batch is laid out with the frame index last (`{n, frames}`
  for bits and posteriors, `{m, dc, frames}` for edge messages), so every
  graph access is an `Nx.take` along axis 0 with constant index tables
  from `ExSdr.FEC.LDPC.Graph`. Only operations `NxFixed.WidthCheck`
  understands are used (takes, constant slices, elementwise ops, sums),
  so integer instances can be width-checked and become hardware.

  Iterative decoders run `max_iter` iterations. With `loop: :unrolled`
  (the default, a pure dataflow graph for the width check and hardware)
  every iteration is computed; with `loop: :while` (used by
  `ExSdr.FEC.LDPC.run/3`) the loop stops when every frame is done. Both
  give the same result: a frame whose hard decisions satisfy all parity
  checks is frozen (as GNU Radio stops decoding it).

  LLR convention at the interface: positive means 1 (as GNU Radio's
  decoders and `ExSdr.Digital`'s soft decoder). Internally the sign is
  flipped (positive means 0, the usual LLR), and a bit is 1 if its
  posterior is negative.
  """
  import Nx.Defn
  import Bitwise

  alias NxFixed.Fixed

  # -- encoders ---------------------------------------------------------------------

  @doc """
  Encoder: information bits (`u8`, `k` per frame) to codewords (`n` per
  frame), `c = G' s` over GF(2) with the code's dense generator
  (`opts[:generator]`, `n` rows of `k` 0/1 values).
  """
  def encode(bits, state, opts) do
    n = opts[:n]
    k = opts[:k]
    frames = div(Nx.size(bits), k)
    g = Nx.tensor(opts[:generator], type: :s32, backend: Nx.BinaryBackend)
    s = bits |> Nx.reshape({frames, k}) |> Nx.transpose() |> Nx.as_type(:s32)

    c =
      g
      |> Nx.dot(s)
      |> Nx.bitwise_and(1)
      |> Nx.as_type(:u8)
      |> Nx.transpose()
      |> Nx.reshape({frames * n})

    {c, state}
  end

  # -- bit flipping -------------------------------------------------------------------

  @doc """
  gr-fec's bit flip decoder: hard decisions (`llr > 0` is 1); while some
  parity check fails (at most `max_iter` times), flip every bit that is
  in the largest number of failed checks.
  """
  def bit_flip(llr, state, opts) do
    {ch, frames} = channel_bits(llr, opts[:n])
    x = ch
    s0 = %{x: x, active: any_failed(x, opts)}

    final = loop(s0, opts, &bit_flip_iteration(&1, &2, opts), &tables(&1, :bit_flip))
    {output(final.x, frames, opts), state}
  end

  @doc "One bit-flip iteration on `%{x: u8[n, frames], active: u8[frames]}`."
  def bit_flip_iteration(%{x: x, active: active}, t, _opts) do
    syn = syndrome(x, t)
    syn_ext = Nx.concatenate([Nx.as_type(syn, :s32), zeros_like_row(syn, :s32)])
    counts = syn_ext |> Nx.take(t.var_checks) |> Nx.sum(axes: [1])
    max = tree_max(counts)
    flip = Nx.logical_and(Nx.equal(counts, Nx.new_axis(max, 0)), Nx.new_axis(active, 0))
    x = Nx.bitwise_xor(x, Nx.as_type(flip, :u8))
    %{x: x, active: Nx.logical_and(active, any_failed(x, t)) |> Nx.as_type(:u8)}
  end

  # -- soft decoders ------------------------------------------------------------------

  @doc """
  Soft decoder on LLRs (`f32`, or `NxFixed.Fixed` for integer instances).

  Options: `:check` (`:bp` for the tanh rule, `:min_sum`), `:schedule`
  (`:flooding`, `:layered`), `:scale` and `:offset` (min-sum magnitude
  correction), `:float_type` (`:f32` or `:f64`), fixed point: `:msg_max`,
  `:state_max`, `:scale16` (scale in 1/16), `:offset_q` (offset in LLR
  LSBs); plus the graph tables, `:max_iter`, `:loop` and `:output`.
  """
  def soft(llr, state, opts) do
    {ch, frames} = channel(llr, opts)
    x = Nx.less(ch, zero(ch)) |> Nx.as_type(:u8)
    t0 = tables(opts, :soft)
    active = any_failed(x, t0)

    s0 =
      case opts[:schedule] do
        :flooding ->
          %{ch: ch, l: ch, r: zero_messages(ch, opts[:m], opts), x: x, active: active}

        :layered ->
          r =
            opts[:layers]
            |> Enum.map(&zero_messages(ch, length(&1.rows), opts))
            |> List.to_tuple()

          %{l: ch, r: r, x: x, active: active}
      end

    iteration =
      case opts[:schedule] do
        :flooding -> &flooding_iteration(&1, &2, opts)
        :layered -> &layered_iteration(&1, &2, opts)
      end

    final = loop(s0, opts, iteration, &tables(&1, :soft))
    {output(final.x, frames, opts), state}
  end

  @doc "One flooding iteration: all checks from the previous messages, then all variables."
  def flooding_iteration(s, t, opts) do
    q = s.l |> extend() |> Nx.take(t.row_vars) |> sub(s.r, opts) |> state_sat(opts)
    r = check_update(q, t.pad, opts)
    l = posterior(s.ch, r, t, opts)
    x = Nx.less(l, zero(l)) |> Nx.as_type(:u8)
    freeze(s, %{s | l: l, r: r, x: x}, t)
  end

  @doc "One layered iteration: the layers in order, each updating the posteriors it touches."
  def layered_iteration(s, t, opts) do
    {l, rs} =
      t.layers
      |> Enum.with_index()
      |> Enum.reduce({s.l, []}, fn {layer, i}, {l, rs} ->
        r_old = elem(s.r, i)
        q = l |> extend() |> Nx.take(layer.row_vars) |> sub(r_old, opts) |> state_sat(opts)
        r = check_update(q, layer.pad, opts)
        lq = q |> add(r, opts) |> state_sat(opts)
        {rows, dc, frames} = Nx.shape(lq)
        upd = lq |> Nx.reshape({rows * dc, frames}) |> Nx.take(layer.pos)
        l = sel(Nx.new_axis(layer.in_layer, 1), upd, l)
        {l, [r | rs]}
      end)

    r = rs |> Enum.reverse() |> List.to_tuple()
    x = Nx.less(l, zero(l)) |> Nx.as_type(:u8)
    freeze(s, %{s | l: l, r: r, x: x}, t)
  end

  # New values for active frames only; then a frame is active while its
  # decisions fail some check.
  defp freeze(old, new, t) do
    a = old.active

    merged =
      Map.new(new, fn
        {:active, v} ->
          {:active, v}

        {key, v} ->
          {key, select_frames(a, v, Map.fetch!(old, key))}
      end)

    %{merged | active: Nx.logical_and(a, any_failed(merged.x, t)) |> Nx.as_type(:u8)}
  end

  defp select_frames(a, new, old) when is_tuple(new) do
    new
    |> Tuple.to_list()
    |> Enum.zip(Tuple.to_list(old))
    |> Enum.map(fn {n, o} -> select_frames(a, n, o) end)
    |> List.to_tuple()
  end

  defp select_frames(a, new, old) do
    pred = Nx.reshape(a, List.to_tuple(List.duplicate(1, Nx.rank(new) - 1) ++ [Nx.size(a)]))
    sel(pred, new, old)
  end

  # l = ch + sum of the variable's messages, added in column order.
  defp posterior(ch, r, t, opts) do
    {m, dc, frames} = Nx.shape(r)
    flat = Nx.concatenate([Nx.reshape(r, {m * dc, frames}), zeros_like_row(r, Nx.type(r))])
    g = Nx.take(flat, t.var_edges)
    dv = Nx.axis_size(g, 1)

    Enum.reduce(0..(dv - 1), ch, fn j, acc ->
      add(acc, g |> Nx.slice_along_axis(j, 1, axis: 1) |> Nx.squeeze(axes: [1]), opts)
    end)
    |> state_sat(opts)
  end

  # Check node update on q[rows, dc, frames]; padding entries (pad = 1)
  # behave as certain zeros (sign +, infinite magnitude).
  defp check_update(q, pad, opts) do
    case opts[:check] do
      :min_sum -> min_sum(q, pad, opts)
      :bp -> tanh_rule(q, pad, opts)
    end
  end

  defp min_sum(q, pad, opts) do
    dc = Nx.axis_size(q, 1)
    fixed? = opts[:fixed]

    big =
      if fixed?,
        do: Nx.tensor(opts[:msg_max], type: :s32),
        else: Nx.tensor(1.0e30, type: Nx.type(q))

    pad3 = Nx.new_axis(pad, 2)
    q = sel(pad3, big, q)

    cols =
      for j <- 0..(dc - 1), do: q |> Nx.slice_along_axis(j, 1, axis: 1) |> Nx.squeeze(axes: [1])

    neg = Enum.map(cols, &(Nx.less(&1, zero(&1)) |> Nx.as_type(:u8)))

    mag =
      Enum.map(cols, fn c ->
        a = Nx.abs(c)
        if fixed?, do: Nx.min(a, Nx.tensor(opts[:msg_max], type: :s32)), else: a
      end)

    [m0 | rest] = mag

    {m1, m2} =
      Enum.reduce(rest, {m0, big_like(m0, big)}, fn a, {m1, m2} ->
        {Nx.min(m1, a), Nx.min(m2, Nx.max(m1, a))}
      end)

    parity = Enum.reduce(neg, &Nx.bitwise_xor/2)

    out =
      Enum.zip_with(mag, neg, fn a, ng ->
        ext = Nx.select(Nx.equal(a, m1), m2, m1) |> correct(opts)
        Nx.select(Nx.bitwise_xor(parity, ng), Nx.negate(ext), ext)
      end)

    r = Nx.stack(out, axis: 1)
    r = sel(pad3, zero(r), r)
    if fixed?, do: Nx.as_type(r, opts[:msg_type]), else: r
  end

  defp big_like(t, big), do: Nx.broadcast(big, Nx.shape(t))

  # Normalised and offset min-sum.
  defp correct(ext, opts) do
    ext =
      case {opts[:fixed], opts[:scale]} do
        {_, nil} -> ext
        {true, _} -> ext |> Nx.multiply(opts[:scale16]) |> NxFixed.Primitives.shift_round(4)
        {_, s} -> Nx.multiply(ext, s)
      end

    case {opts[:fixed], opts[:offset]} do
      {_, nil} -> ext
      {true, _} -> Nx.max(Nx.subtract(ext, opts[:offset_q]), 0)
      {_, o} -> Nx.max(Nx.subtract(ext, o), zero(ext))
    end
  end

  # gr-fec's awgn_bp: R = 2 atanh(prod over the others of tanh(clip(Q/2, 18))),
  # the product in row order starting from 1.0.
  defp tanh_rule(q, pad, _opts) do
    dc = Nx.axis_size(q, 1)
    one = Nx.tensor(1.0, type: Nx.type(q))

    th =
      q
      |> Nx.divide(2.0)
      |> Nx.clip(-18.0, 18.0)
      |> Nx.tanh()
      |> then(&sel(Nx.new_axis(pad, 2), one, &1))

    cols =
      for j <- 0..(dc - 1), do: th |> Nx.slice_along_axis(j, 1, axis: 1) |> Nx.squeeze(axes: [1])

    out =
      for j <- 0..(dc - 1) do
        prod =
          cols
          |> Enum.with_index()
          |> Enum.reject(fn {_, i} -> i == j end)
          |> Enum.reduce(Nx.broadcast(one, Nx.shape(hd(cols))), fn {c, _}, acc ->
            Nx.multiply(acc, c)
          end)

        prod |> Nx.atanh() |> Nx.multiply(2.0)
      end

    r = Nx.stack(out, axis: 1)
    sel(Nx.new_axis(pad, 2), zero(r), r)
  end

  # -- shared pieces ------------------------------------------------------------------

  # Fixed point: exact s32 arithmetic, saturated to the state range.
  defp sub(a, b, opts) do
    if opts[:fixed],
      do: Nx.subtract(Nx.as_type(a, :s32), Nx.as_type(b, :s32)),
      else: Nx.subtract(a, b)
  end

  defp add(a, b, opts) do
    if opts[:fixed],
      do: Nx.add(Nx.as_type(a, :s32), Nx.as_type(b, :s32)),
      else: Nx.add(a, b)
  end

  defp state_sat(x, opts) do
    if opts[:fixed] do
      max = opts[:state_max]
      x |> Nx.as_type(:s32) |> Nx.clip(-max, max) |> Nx.as_type(opts[:state_type])
    else
      x
    end
  end

  defp zero(t), do: Nx.tensor(0, type: Nx.type(t))

  # Nx.select with the predicate broadcast to the values' shape.
  defp sel(pred, a, b) do
    shape = if Nx.rank(b) >= Nx.rank(a), do: Nx.shape(b), else: Nx.shape(a)
    Nx.select(Nx.broadcast(pred, shape), a, b)
  end

  defp zero_messages(ch, rows, opts) do
    frames = Nx.axis_size(ch, 1)
    type = if opts[:fixed], do: opts[:msg_type], else: Nx.type(ch)
    Nx.broadcast(Nx.tensor(0, type: type), {rows, opts[:dc], frames})
  end

  defp zeros_like_row(t, type) do
    frames = t |> Nx.shape() |> Tuple.to_list() |> List.last()
    Nx.broadcast(Nx.tensor(0, type: type), {1, frames})
  end

  # Append the dummy variable (index n) with a value that padding never uses.
  defp extend(l), do: Nx.concatenate([l, zeros_like_row(l, Nx.type(l))])

  # syndrome[m, frames] (u8) of hard decisions x[n, frames]
  defp syndrome(x, t) do
    [Nx.as_type(x, :s32), zeros_like_row(x, :s32)]
    |> Nx.concatenate()
    |> Nx.take(t.row_vars)
    |> Nx.sum(axes: [1])
    |> Nx.bitwise_and(1)
    |> Nx.as_type(:u8)
  end

  defp any_failed(x, %{row_vars: _} = t) do
    x |> syndrome(t) |> Nx.as_type(:s32) |> Nx.sum(axes: [0]) |> Nx.greater(0) |> Nx.as_type(:u8)
  end

  defp any_failed(x, opts) when is_list(opts), do: any_failed(x, tables(opts, :bit_flip))

  # Maximum over axis 0 by halving (n padded with zeros to a power of two).
  defp tree_max(x) do
    n = Nx.axis_size(x, 0)
    size = 1 <<< ceil_log2(n)

    x =
      if size > n do
        frames = Nx.axis_size(x, 1)
        Nx.concatenate([x, Nx.broadcast(Nx.tensor(0, type: Nx.type(x)), {size - n, frames})])
      else
        x
      end

    reduce_halves(x, size)
  end

  defp reduce_halves(x, 1), do: Nx.squeeze(x, axes: [0])

  defp reduce_halves(x, size) do
    h = div(size, 2)
    x = Nx.max(Nx.slice_along_axis(x, 0, h, axis: 0), Nx.slice_along_axis(x, h, h, axis: 0))
    reduce_halves(x, h)
  end

  defp ceil_log2(1), do: 0
  defp ceil_log2(n), do: 1 + ceil_log2(div(n + 1, 2))

  # Input LLRs {frames * n} -> internal channel values {n, frames}.
  defp channel(%Fixed{t: t}, opts) do
    n = opts[:n]
    frames = div(Nx.size(t), n)
    max = opts[:msg_max]

    ch =
      t
      |> Nx.as_type(:s32)
      |> Nx.negate()
      |> Nx.clip(-max, max)
      |> Nx.as_type(opts[:state_type])
      |> Nx.reshape({frames, n})
      |> Nx.transpose()

    {ch, frames}
  end

  defp channel(llr, opts) do
    n = opts[:n]
    frames = div(Nx.size(llr), n)

    ch =
      llr
      |> Nx.as_type(opts[:float_type])
      |> Nx.negate()
      |> Nx.reshape({frames, n})
      |> Nx.transpose()

    {ch, frames}
  end

  # Hard decisions of the input (llr > 0 is 1), {n, frames}.
  defp channel_bits(%Fixed{t: t}, n), do: channel_bits(t, n)

  defp channel_bits(llr, n) do
    frames = div(Nx.size(llr), n)

    x =
      llr |> Nx.greater(zero(llr)) |> Nx.as_type(:u8) |> Nx.reshape({frames, n}) |> Nx.transpose()

    {x, frames}
  end

  defp output(x, frames, opts) do
    case opts[:output] do
      :codeword ->
        x |> Nx.transpose() |> Nx.reshape({frames * opts[:n]})

      _ ->
        info = Nx.tensor(opts[:info], type: :s32, backend: Nx.BinaryBackend)
        x |> Nx.take(info) |> Nx.transpose() |> Nx.reshape({frames * length(opts[:info])})
    end
  end

  # The graph tables as constant tensors.
  defp tables(opts, kind) do
    c = &Nx.tensor(&1, type: :s32, backend: Nx.BinaryBackend)
    n = opts[:n]
    t = %{row_vars: c.(opts[:row_vars])}

    case kind do
      :bit_flip ->
        Map.put(t, :var_checks, c.(opts[:var_checks]))

      :soft ->
        pad = fn rv ->
          Nx.tensor(for(r <- rv, do: for(v <- r, do: if(v == n, do: 1, else: 0))),
            type: :u8,
            backend: Nx.BinaryBackend
          )
        end

        t
        |> Map.put(:var_edges, c.(opts[:var_edges]))
        |> Map.put(:pad, pad.(opts[:row_vars]))
        |> Map.put(
          :layers,
          for l <- opts[:layers] || [] do
            %{
              row_vars: c.(l.row_vars),
              pad: pad.(l.row_vars),
              pos: c.(l.pos),
              in_layer: Nx.tensor(l.in_layer, type: :u8, backend: Nx.BinaryBackend)
            }
          end
        )
    end
  end

  # -- iteration drivers ----------------------------------------------------------------

  defp loop(s0, opts, iteration, make_tables) do
    case opts[:loop] || :unrolled do
      :unrolled ->
        t = make_tables.(opts)
        Enum.reduce(1..opts[:max_iter]//1, s0, fn _, s -> iteration.(s, t) end)

      :while ->
        while_loop(s0,
          step: fn s -> iteration.(s, make_tables.(opts)) end,
          max_iter: opts[:max_iter]
        )
    end
  end

  defn while_loop(s0, opts \\ []) do
    step = opts[:step]
    max_iter = opts[:max_iter]

    {s, _} =
      while {s = s0, i = 0}, i < max_iter and Nx.sum(s.active) > 0 do
        {step.(s), i + 1}
      end

    s
  end
end
