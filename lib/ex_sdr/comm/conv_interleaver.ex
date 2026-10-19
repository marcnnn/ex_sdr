defmodule ExSdr.Comm.ConvInterleaver do
  @moduledoc """
  Convolutional (Forney) symbol interleaver with `branches` I and unit
  delay `depth` M: symbols cycle through the branches, branch j delays by
  j * M symbols when interleaving and by (I - 1 - j) * M when
  deinterleaving, so the pair delays every symbol by I * (I - 1) * M.
  The delay lines start filled with zeros. Byte symbols (binaries).
  """
  defstruct [:branches, :depth]

  def new(opts),
    do: %__MODULE__{
      branches: Keyword.fetch!(opts, :branches),
      depth: Keyword.fetch!(opts, :depth)
    }

  @doc "Delay of interleaver + deinterleaver in symbols."
  def delay(%__MODULE__{branches: i, depth: m}), do: i * (i - 1) * m

  @doc "Option `:native` as in `ExSdr.Comm.Scrambler.apply/3`."
  def interleave(c, bytes, opts \\ []), do: run(c, bytes, false, opts)
  def deinterleave(c, bytes, opts \\ []), do: run(c, bytes, true, opts)

  @doc """
  Streaming state: `{history, position}`. Every output byte is the input
  byte I * d earlier (d: delay of its branch), so the state is the last
  I * (I - 1) * M input bytes (zeros at the start, like the delay lines)
  and the number of bytes so far. `stream/4` gives the same output for any
  split of the input as one call of `interleave/3` or `deinterleave/3`.
  """
  def stream_state(%__MODULE__{} = c), do: {:binary.copy(<<0>>, delay(c)), 0}

  @doc "`{out, state}`; `direction` is `:interleave` or `:deinterleave`. Option `:native`."
  def stream(%__MODULE__{branches: i, depth: m} = c, direction, bytes, {history, pos}, opts \\ []) do
    inverse? = direction == :deinterleave

    if Keyword.get_lazy(opts, :native, &ExSdr.SIMD.available?/0) do
      {out, history} = ExSdr.NIF.conv_interleave_stream(history, bytes, i, m, inverse?, pos)
      {out, {history, pos + byte_size(bytes)}}
    else
      hist = delay(c)
      ext = history <> bytes

      out =
        for k <- 0..(byte_size(bytes) - 1)//1, into: <<>> do
          j = rem(pos + k, i)
          d = if inverse?, do: (i - 1 - j) * m, else: j * m
          <<:binary.at(ext, hist + k - i * d)>>
        end

      {out, {binary_part(ext, byte_size(bytes), hist), pos + byte_size(bytes)}}
    end
  end

  defp run(%__MODULE__{branches: i, depth: m}, bytes, inverse?, opts) do
    if Keyword.get_lazy(opts, :native, &ExSdr.SIMD.available?/0) do
      ExSdr.NIF.conv_interleave(bytes, i, m, inverse?)
    else
      depth = fn j -> if inverse?, do: (i - 1 - j) * m, else: j * m end

      fifos =
        for j <- 0..(i - 1), into: %{}, do: {j, :queue.from_list(List.duplicate(0, depth.(j)))}

      {out, _} =
        bytes
        |> :binary.bin_to_list()
        |> Enum.with_index()
        |> Enum.map_reduce(fifos, fn {b, n}, fifos ->
          j = rem(n, i)
          {{:value, o}, q} = :queue.out(:queue.in(b, fifos[j]))
          {o, Map.put(fifos, j, q)}
        end)

      :binary.list_to_bin(out)
    end
  end
end
