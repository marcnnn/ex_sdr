defmodule ExSdr.FEC.PolarDecoder do
  @moduledoc """
  gr-fec's polar decoders as `ExSdr.FEC.Generic` objects: `polar_decoder_sc`
  (successive cancellation), `polar_decoder_sc_list` (SC list) and
  `polar_decoder_sc_systematic`, for the codes of `ExSdr.FEC.Polar`.

  Input: `block_size` `f32` values, positive for bit 1 (GNU Radio multiplies
  them by `ln(0.1 / 0.9) = -2.19722458` to get LLRs, so the decoders are
  tuned for +-1 style inputs; only the scale of the list decoder's path
  metrics depends on it, not the decisions of SC). Output: `num_info_bits`
  bits.

  The algorithm is GNU Radio's: the codeword of `ExSdr.FEC.Polar.transform/1`
  has two halves, the codewords of `a_j = u_2j xor u_2j+1` and `b_j = u_2j+1`;
  the decoder is a tree of nodes that run in lockstep over their pairs: bit
  `2j` has the LLR `minsum(La_j, Lb_j)`, bit `2j + 1` the LLR
  `Lb_j + (1 - 2 u_2j) La_j`, and the decisions `a_j = u_2j xor u_2j+1`, `b_j =
  u_2j+1` are fed back to the children. Every operation is rounded to `f32`
  like the C code, so the decisions are the same as GNU Radio's.

  `list_mode: :gnuradio` (default) reproduces the path bookkeeping of GNU Radio's
  `scl_list` exactly, including a flaw of it: when a surviving path has to take
  over a buffer (`steal_vector_ownership`) it takes the first discarded buffer,
  which can still be the source of a later survivor, so that survivor then
  continues with another path's history. This costs a little error rate and
  shows in about one frame in a few hundred at the noise where list decoding
  matters. `list_mode: :exact` is the textbook algorithm (the best
  `max_list_size` of all extensions survive).

  The list decoder keeps the `max_list_size` paths with the smallest metric
  (a path pays `|llr|` when its bit disagrees with the sign of the LLR). When
  metrics tie exactly, GNU Radio's order depends on `std::sort`, here the sort
  is stable (lower path first); with real valued input this never matters.
  The paths are persistent trees, so splitting a path costs nothing.
  """
  @behaviour ExSdr.FEC.Generic

  import Bitwise

  alias ExSdr.FEC.Polar

  @llr_factor -2.19722458

  @doc """
  Decoder objects: `kind` is `:sc`, `:sc_list` (option `:max_list_size`) or
  `:sc_systematic`. Options as `ExSdr.FEC.Polar.encoder/1`.
  """
  def decoder(kind, opts) do
    k = Keyword.fetch!(opts, :num_info_bits)
    systematic = kind == :sc_systematic
    values = if systematic, do: [], else: Keyword.get(opts, :frozen_bit_values, [])

    g =
      Polar.make(:decoder, "polar_decoder_#{kind}", Keyword.put(opts, :frozen_bit_values, values))

    conf =
      Map.merge(g.conf, %{
        variant: kind,
        list: Keyword.get(opts, :max_list_size, 1),
        list_mode: Keyword.get(opts, :list_mode, :gnuradio)
      })

    if kind == :sc_list and conf.list < 1, do: raise(ArgumentError, "max_list_size must be >= 1")

    if kind == :sc_list and conf.list_mode == :gnuradio and Polar.log2(conf.list) == nil,
      do: raise(ArgumentError, "list_mode :gnuradio needs a power of two list size")

    %{g | impl: __MODULE__, conf: conf, frame_size: k, max_frame_size: k, kind: :decoder}
  end

  @impl true
  def input_size(%{conf: %{n: n}}), do: n
  @impl true
  def output_size(%{conf: %{k: k}}), do: k
  @impl true
  def rate(%{conf: %{k: k, n: n}}), do: k / n
  @impl true
  def input_item(_), do: :f32
  @impl true
  def hardware(_), do: :transform
  @impl true
  def set_frame_size(g, _), do: {false, g}

  @impl true
  def work(%{conf: c}, input, state) do
    llrs =
      input
      |> Nx.as_type(:f32)
      |> Nx.slice_along_axis(0, c.n)
      |> Nx.to_flat_list()
      |> Enum.map(&mul32(&1, @llr_factor))

    u =
      case c.variant do
        :sc_list when c.list_mode == :exact -> list_decode(c, llrs)
        :sc_list -> list_decode_gr(c, llrs)
        _ -> sc_decode(c, llrs)
      end

    out =
      case c.variant do
        :sc_systematic ->
          frame =
            u |> Nx.tensor(type: :u8) |> Polar.transform() |> Nx.to_flat_list() |> List.to_tuple()

          for pos <- c.info_reversed, do: elem(frame, pos)

        _ ->
          ut = List.to_tuple(u)
          for pos <- c.info, do: elem(ut, pos)
      end

    {Nx.tensor(out, type: :u8), state}
  end

  # -- f32 arithmetic ---------------------------------------------------------------------------

  defp r32(x) do
    <<v::float-32>> = <<x::float-32>>
    v
  end

  defp mul32(a, b), do: r32(a * b)

  # copysignf(1.0, x)
  defp sign(x) do
    <<s::1, _::31>> = <<x::float-32>>
    if s == 1, do: -1.0, else: 1.0
  end

  defp minsum(la, lb), do: sign(la) * sign(lb) * min(abs(la), abs(lb))

  defp g(lb, la, 0), do: r32(lb + la)
  defp g(lb, la, 1), do: r32(lb - la)

  # -- the lockstep tree ----------------------------------------------------------------------------

  defp tree([l]), do: {:leaf, l}

  defp tree(llrs) do
    {a, b} = Enum.split(llrs, div(length(llrs), 2))
    {:node, 0, tree(a), tree(b), nil, nil, nil}
  end

  # LLR of the node's next bit; returns {llr, node}
  defp next({:leaf, l} = node), do: {l, node}

  defp next({:node, j, a, b, _la, _lb, p}) when (j &&& 1) == 0 do
    {la, a} = next(a)
    {lb, b} = next(b)
    {minsum(la, lb), {:node, j, a, b, la, lb, p}}
  end

  defp next({:node, j, _a, _b, la, lb, p} = node) when (j &&& 1) == 1,
    do: {g(lb, la, p), node}

  # the decided value of the node's bit
  defp feed({:leaf, _} = node, _bit), do: node

  defp feed({:node, j, a, b, la, lb, _p}, bit) when (j &&& 1) == 0,
    do: {:node, j + 1, a, b, la, lb, bit}

  defp feed({:node, j, a, b, la, lb, p}, bit) do
    {:node, j + 1, feed(a, bxor(p, bit)), feed(b, bit), la, lb, p}
  end

  defp decide(llr), do: if(llr < 0.0, do: 1, else: 0)

  # -- SC ----------------------------------------------------------------------------------------------

  defp sc_decode(c, llrs) do
    frozen = frozen_map(c)

    {u, _} =
      Enum.reduce(0..(c.n - 1), {[], tree(llrs)}, fn i, {u, node} ->
        {llr, node} = next(node)

        bit =
          case frozen do
            %{^i => v} -> v
            _ -> decide(llr)
          end

        {[bit | u], feed(node, bit)}
      end)

    Enum.reverse(u)
  end

  defp frozen_map(c), do: c.frozen |> Enum.zip(c.values) |> Map.new()

  # -- SC list -------------------------------------------------------------------------------------------

  defp list_decode(c, llrs) do
    frozen = frozen_map(c)
    size = c.list
    paths = [{0.0, [], tree(llrs)}]

    paths =
      Enum.reduce(0..(c.n - 1), paths, fn i, paths ->
        # LLR of bit i on every path
        with_llr =
          Enum.map(paths, fn {pm, u, node} ->
            {llr, node} = next(node)
            {pm, u, node, llr}
          end)

        case frozen do
          %{^i => v} ->
            Enum.map(with_llr, fn {pm, u, node, llr} ->
              {penalty(pm, llr, v), [v | u], feed(node, v)}
            end)

          _ ->
            zeros =
              Enum.map(with_llr, fn {pm, u, node, llr} ->
                {penalty(pm, llr, 0), [0 | u], feed(node, 0)}
              end)

            ones =
              Enum.map(with_llr, fn {pm, u, node, llr} ->
                {penalty(pm, llr, 1), [1 | u], feed(node, 1)}
              end)

            both = zeros ++ ones

            if length(both) <= size,
              do: both,
              else: both |> Enum.sort_by(&elem(&1, 0)) |> Enum.take(size)
        end
      end)

    {_, u, _} = Enum.min_by(paths, &elem(&1, 0))
    Enum.reverse(u)
  end

  defp penalty(pm, llr, 0) when llr > 0.0, do: pm
  defp penalty(pm, llr, 1) when llr < 0.0, do: pm
  defp penalty(pm, llr, _), do: r32(pm + abs(llr))

  # -- SC list with GNU Radio's buffer handling --------------------------------------------------

  # Paths are objects (metric, owns, buffer id), buffers are shared by id like
  # the raw pointers of scl_list.cc. A buffer holds the decided bits `u`
  # (a map position -> bit; a decision is written before it is used, and may
  # be overwritten), the node tree after computing the LLR of the last
  # position and that LLR. The decision of the previous position is fed to
  # the tree when the next LLR is computed, so that a rewritten bit counts.
  defp list_decode_gr(c, llrs) do
    frozen = frozen_map(c)
    size = c.list
    fresh = %{u: %{}, tree: tree(llrs), llr: 0.0}

    bufs = Map.new(0..(size - 1), &{&1, fresh})

    paths =
      for i <- 0..(2 * size - 1) do
        if i < size,
          do: %{pm: 0.0, owns: true, buf: i},
          else: %{pm: 0.0, owns: false, buf: nil}
      end

    {paths, bufs, _active} =
      Enum.reduce(0..(c.n - 1), {paths, bufs, 1}, fn i, {paths, bufs, active} ->
        bufs =
          paths
          |> Enum.take(active)
          |> Enum.reduce(bufs, fn p, bufs -> Map.update!(bufs, p.buf, &calc(&1, i)) end)

        case frozen do
          %{^i => v} ->
            {paths, bufs} =
              paths
              |> Enum.with_index()
              |> Enum.reduce({paths, bufs}, fn
                {p, idx}, {paths, bufs} when idx < active ->
                  b = bufs[p.buf]
                  p = %{p | pm: penalty(p.pm, b.llr, v)}
                  {List.replace_at(paths, idx, p), Map.put(bufs, p.buf, put_u(b, i, v))}

                _, acc ->
                  acc
              end)

            {paths, bufs, active}

          _ when active < size ->
            # duplicate every active path: the copy takes bit 1
            {paths, bufs} =
              Enum.reduce(0..(active - 1), {paths, bufs}, fn j, {paths, bufs} ->
                src = Enum.at(paths, j)
                dst = Enum.at(paths, j + active)
                sb = bufs[src.buf]
                bufs = Map.put(bufs, dst.buf, sb)
                dst = %{dst | pm: penalty(src.pm, sb.llr, 1)}
                src = %{src | pm: penalty(src.pm, sb.llr, 0)}

                bufs =
                  bufs
                  |> Map.update!(src.buf, &put_u(&1, i, 0))
                  |> Map.update!(dst.buf, &put_u(&1, i, 1))

                paths = paths |> List.replace_at(j, src) |> List.replace_at(j + active, dst)
                {paths, bufs}
              end)

            {paths, bufs, 2 * active}

          _ ->
            {paths, bufs} = branch_and_prune(paths, bufs, size, i)
            {paths, bufs, active}
        end
      end)

    best = paths |> Enum.take(size) |> Enum.min_by(& &1.pm)
    u = bufs[best.buf].u
    for i <- 0..(c.n - 1), do: Map.fetch!(u, i)
  end

  defp calc(b, i) do
    tree = if i > 0, do: feed(b.tree, Map.fetch!(b.u, i - 1)), else: b.tree
    {llr, tree} = next(tree)
    %{b | tree: tree, llr: llr}
  end

  defp put_u(b, i, bit), do: %{b | u: Map.put(b.u, i, bit)}

  defp branch_and_prune(paths, bufs, size, i) do
    {first, second} = Enum.split(paths, size)

    # every path and its branch copy (bit 1) share a buffer
    pairs =
      Enum.zip(first, second)
      |> Enum.map(fn {orig, target} ->
        llr = bufs[orig.buf].llr

        {%{orig | pm: penalty(orig.pm, llr, 0)},
         %{target | pm: penalty(orig.pm, llr, 1), buf: orig.buf}}
      end)

    all = Enum.map(pairs, &elem(&1, 0)) ++ Enum.map(pairs, &elem(&1, 1))
    sorted = Enum.sort_by(all, & &1.pm)

    Enum.reduce(0..(size - 1), {sorted, bufs}, fn idx, {paths, bufs} ->
      p = Enum.at(paths, idx)

      if p.owns do
        {paths, Map.update!(bufs, p.buf, &put_u(&1, i, 0))}
      else
        tpos = size + Enum.find_index(Enum.drop(paths, size), & &1.owns)
        q = Enum.at(paths, tpos)
        # q's buffer receives a copy of p's (the one it shared with its twin)
        bufs = Map.put(bufs, q.buf, bufs[p.buf])
        p = %{p | buf: q.buf, owns: true}
        q = %{q | owns: false}
        bufs = Map.update!(bufs, p.buf, &put_u(&1, i, 1))
        {paths |> List.replace_at(idx, p) |> List.replace_at(tpos, q), bufs}
      end
    end)
  end
end
