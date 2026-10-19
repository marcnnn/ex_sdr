defmodule ExSdr.Math.Vector do
  @moduledoc """
  Rate-changing kernels of the math family.

  Frame kernels (`:frame` blocks, `fun(input, state, opts)`) take whole
  groups of `rate` input items with the item index first, so their input
  length is a multiple of the input rate: `integrate`, `repeat`, `max`,
  `min`. Host kernels (`:host`, `fun.(input, state)`) take any number of
  items and carry the position in the decimation cycle as state, so their
  output does not depend on how the input is split: `keep_one_in_n`,
  `keep_m_in_n`.
  """
  import Nx.Defn

  alias NxFixed.Fixed
  alias NxFixed.Num

  @doc """
  Sum of each group of `opts[:decim]` items (GNU Radio `integrate_*` with
  dump 1). Fixed point sums in 64-bit containers and saturates once at the
  output, so the format needs headroom for `decim` times the input range.
  """
  defn integrate(input, state, opts \\ []) do
    opts = keyword!(opts, [:format, :decim])
    {sum_groups(input, opts[:decim], opts[:format]), state}
  end

  @doc "Each item repeated `opts[:interp]` times."
  defn repeat(input, state, opts \\ []) do
    opts = keyword!(opts, [:format, :interp])
    {repeat_items(input, opts[:interp]), state}
  end

  @doc "Maximum of each group of `opts[:vlen]` items (real)."
  defn max_ff(input, state, opts \\ []) do
    opts = keyword!(opts, [:format, :vlen])
    {reduce_groups(input, opts[:vlen], :max), state}
  end

  @doc "Minimum of each group of `opts[:vlen]` items (real)."
  defn min_ff(input, state, opts \\ []) do
    opts = keyword!(opts, [:format, :vlen])
    {reduce_groups(input, opts[:vlen], :min), state}
  end

  # -- host kernels: keep one item of every n ----------------------------------------

  @doc """
  Keeps the item at position `n - 1` of every group of `n` (GNU Radio
  `keep_one_in_n`). The state is the position in the cycle, `0 <= state < n`.
  """
  def keep_one_in_n(n) do
    fn input, pos ->
      keep_positions(input, pos, n, fn p -> p == n - 1 end)
    end
  end

  @doc """
  Keeps the first `m` items of every group of `n` (GNU Radio `keep_m_in_n`
  with offset 0). The state is the position in the cycle.
  """
  def keep_m_in_n(m, n) do
    fn input, pos ->
      keep_positions(input, pos, n, fn p -> p < m end)
    end
  end

  # -- helpers -------------------------------------------------------------------------

  deftransformp sum_groups({re, im}, n, format),
    do: {sum_groups(re, n, format), sum_groups(im, n, format)}

  deftransformp sum_groups(%Fixed{t: t, frac: frac}, n, %Num.Format{kind: :fixed} = format) do
    summed = t |> Nx.as_type(:s64) |> Nx.reshape({:auto, n}) |> Nx.sum(axes: [1])
    %Fixed{t: summed, frac: frac} |> Num.quantize(format)
  end

  deftransformp sum_groups(x, n, %Num.Format{kind: :float}),
    do: x |> Nx.reshape({:auto, n}) |> Nx.sum(axes: [1])

  deftransformp repeat_items({re, im}, k), do: {repeat_items(re, k), repeat_items(im, k)}

  deftransformp repeat_items(%Fixed{t: t, frac: frac}, k),
    do: %Fixed{t: repeat_items(t, k), frac: frac}

  deftransformp repeat_items(x, k) do
    items = elem(Nx.shape(x), 0)

    x
    |> Nx.reshape({items, 1})
    |> Nx.broadcast({items, k})
    |> Nx.reshape({:auto})
  end

  deftransformp reduce_groups(%Fixed{t: t, frac: frac}, v, op),
    do: %Fixed{t: reduce_groups(t, v, op), frac: frac}

  deftransformp reduce_groups(x, v, :max),
    do: x |> Nx.reshape({:auto, v}) |> Nx.reduce_max(axes: [1])

  deftransformp reduce_groups(x, v, :min),
    do: x |> Nx.reshape({:auto, v}) |> Nx.reduce_min(axes: [1])

  # Keeps the positions `pos + i` (i = 0..length-1) whose position in the
  # cycle of `n` satisfies `keep?`. Returns the kept items and the next state.
  defp keep_positions(input, pos, n, keep?) do
    len = input_length(input)
    kept = for i <- 0..(len - 1)//1, keep?.(rem(pos + i, n)), do: i
    next = rem(pos + len, n)

    if kept == [] do
      raise ArgumentError,
            "no item of this chunk is kept (chunks of at least n items always keep one): " <>
              "Nx cannot hold an empty tensor"
    end

    {take_items(input, kept), next}
  end

  defp input_length({re, _im}), do: input_length(re)
  defp input_length(%Fixed{t: t}), do: input_length(t)
  defp input_length(t), do: elem(Nx.shape(t), 0)

  defp take_items({re, im}, idx), do: {take_items(re, idx), take_items(im, idx)}
  defp take_items(%Fixed{t: t, frac: frac}, idx), do: %Fixed{t: take_items(t, idx), frac: frac}

  defp take_items(t, idx) do
    t
    |> Nx.take(Nx.tensor(idx, type: :s64))
  end
end
