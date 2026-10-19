defmodule ExSdr.Timing.TestKit do
  @moduledoc false
  # Shared by test/timing/*_test.exs: builds timing blocks from case files,
  # runs them over GNU Radio's input and compares.
  alias ExSdr.Test.GnuRadioFixture, as: F
  alias NxFixed.{Block, Num}

  def family, do: F.family("timing")

  @doc "The block of a (shared runner) case in `format`."
  def block(case, format, extra \\ []) do
    %{"block" => %{"name" => name, "opts" => opts}, "params" => params} = case
    opts = Enum.map(opts, fn {k, v} -> {String.to_atom(k), Map.get(params, v, v)} end)
    ExSdr.Timing.block(String.to_atom(name), opts ++ [format: format] ++ extra)
  end

  @doc "Input items (Q15 integers, `{re, im}` pairs for complex) in `format`."
  def items(input, format) do
    case input do
      [{_, _} | _] ->
        {re, im} = Enum.unzip(input)
        {samples(re, format), samples(im, format)}

      _ ->
        samples(input, format)
    end
  end

  defp samples(ints, format), do: Num.from_int(Nx.tensor(ints, type: :s16), 15, format)

  @doc "Runs `input` (Q15 integers) from `state`; returns `{floats, state}`."
  def run(block, input, state \\ :init) do
    {out, state} = Block.run(block, items(input, block.opts[:format]), state)
    {floats(out), state}
  end

  @doc "Block output as Elixir floats (`{re, im}` for complex); `[]` for none."
  def floats(nil), do: []

  def floats({re, im}) do
    Enum.zip(Num.to_float(re) |> List.wrap(), Num.to_float(im) |> List.wrap())
  end

  def floats(x), do: Num.to_float(x) |> List.wrap()

  @doc "Componentwise absolute differences of two equally long lists."
  def errors(a, b) do
    Enum.zip_with(a, b, fn
      {ar, ai}, {br, bi} -> max(abs(ar - br), abs(ai - bi))
      x, y -> abs(x - y)
    end)
  end

  @doc "Common prefix of two output lists."
  def prefix(a, b) do
    n = min(length(a), length(b))
    {Enum.take(a, n), Enum.take(b, n)}
  end

  @doc "Format of a case's variant."
  def format(%{"bits" => bits, "frac" => frac} = f),
    do: Num.q(bits, frac, rounding: String.to_atom(f["rounding"] || "nearest"))

  @doc "Splits a list at the given cut points."
  def chunks(list, cuts) do
    {parts, rest} =
      Enum.reduce(cuts, {[], list}, fn n, {acc, rest} ->
        {a, b} = Enum.split(rest, n)
        {[a | acc], b}
      end)

    Enum.reverse([rest | parts]) |> Enum.reject(&(&1 == []))
  end

  @doc "Runs `input` in pieces, carrying the state; outputs concatenated."
  def run_chunked(block, input, cuts) do
    {outs, _} =
      Enum.map_reduce(chunks(input, cuts), block.init, fn piece, state ->
        run(block, piece, state)
      end)

    Enum.concat(outs)
  end
end
