defmodule ExSdr.Sources.Vector do
  @moduledoc """
  `vector_source`: plays a stored vector, repeating or then silent.

  The vector is a constant parameter (a block RAM in hardware), the state
  the read position. Without `:repeat` the position stops at the end and
  the output is zero from then on (GNU Radio ends the stream instead; a
  stream kernel cannot, so callers take `length(vector)` items).
  """
  import Nx.Defn
  import NxFixed.Num

  alias NxFixed.Fixed
  alias NxFixed.Num.Format

  @doc """
  Stream kernel `step(_x, position, table, opts)`; `table` is a real vector
  in the format (a `{re, im}` pair of vectors for complex). Options:
  `:format`, `:repeat` (true).
  """
  defn vector_source(_x, position, table, opts \\ []) do
    opts = keyword!(opts, [:format, repeat: true])
    read(position, table, opts[:repeat])
  end

  deftransform read(position, {re, im}, repeat) do
    {y_re, next} = read(position, re, repeat)
    {y_im, _} = read(position, im, repeat)
    {{y_re, y_im}, next}
  end

  deftransform read(position, table, repeat) do
    len = table |> raw() |> Nx.axis_size(0)
    padded = pad(table)
    pos = Nx.as_type(position, :s64)
    y = take(padded, Nx.min(pos, len))

    next =
      if repeat,
        do: Nx.remainder(Nx.add(pos, 1), len),
        else: Nx.min(Nx.add(pos, 1), len)

    {y, Nx.as_type(next, :u32)}
  end

  defp raw(%Fixed{t: t}), do: t
  defp raw(t), do: t

  # One zero entry after the data, read once a non repeating vector ends.
  defp pad(%Fixed{t: t, frac: frac}), do: %Fixed{t: pad_tensor(t), frac: frac}
  defp pad(t), do: pad_tensor(t)

  defp pad_tensor(t), do: Nx.concatenate([t, Nx.broadcast(Nx.tensor(0, type: Nx.type(t)), {1})])

  defp take(%Fixed{t: t, frac: frac}, i), do: %Fixed{t: Nx.take(t, i), frac: frac}
  defp take(t, i), do: Nx.take(t, i)

  @doc "The table parameter for `values` (numbers, or `{re, im}` tuples) in `format`."
  def table(values, %Format{} = format) do
    if match?([{_, _} | _], values) do
      {re, im} = Enum.unzip(values)
      {table(re, format), table(im, format)}
    else
      case format do
        %Format{kind: :float} ->
          Nx.tensor(values, type: :f32, backend: Nx.BinaryBackend)

        %Format{kind: :fixed} = f ->
          ints = quantize_constants(values, f)
          type = if f.bits <= 16, do: :s16, else: :s32
          %Fixed{t: Nx.tensor(ints, type: type, backend: Nx.BinaryBackend), frac: f.frac}
      end
    end
  end
end
