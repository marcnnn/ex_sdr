# Shared by the math tests: runs a case's chain of ExSdr.Math blocks over the
# case input in a format, returns plain lists, and compares them.
defmodule ExSdr.MathTest.Case do
  @moduledoc false
  alias ExSdr.Test.GnuRadioFixture, as: F
  alias NxFixed.Num
  alias NxFixed.Fixed

  @cases_dir Path.expand("../dsp_cases/math", __DIR__)

  def paths, do: Path.wildcard(Path.join(@cases_dir, "*.json")) |> Enum.sort()

  @doc "`{case, data}` with GNU Radio data (see ExSdr.Test.GnuRadioFixture.load/3)."
  def load(path) do
    case = F.read_case(path)
    {case, F.load(case, path, Path.expand("../fixtures/gnuradio/math", __DIR__))}
  end

  def format("float"), do: Num.float()
  def format(map) when is_map(map), do: F.format(map)

  @doc "Our blocks of a case, in order, configured for `format`."
  def blocks(case, format) do
    specs = List.wrap(case["block"])
    params = case["params"]

    for spec <- specs do
      opts =
        for {k, v} <- Map.get(spec, "opts", %{}), into: [] do
          {String.to_atom(k), resolve(v, params)}
        end

      ExSdr.Math.block(String.to_atom(spec["name"]), [format: format] ++ opts)
    end
  end

  defp resolve("msb", _params), do: :msb
  defp resolve("lsb", _params), do: :lsb
  defp resolve(name, params) when is_binary(name), do: Map.fetch!(params, name)
  defp resolve(v, _params), do: v

  @doc """
  Input items of the case in `format` (as containers). Cases with two inputs
  have items `[a, b]`; the container is the pair of the two inputs' containers.
  """
  def container(items, case, format) do
    types = for spec <- case["inputs"] || [case["input"]], do: spec["type"]

    case types do
      [type] ->
        items_container(items, type, format)

      types ->
        for {type, i} <- Enum.with_index(types) do
          items_container(Enum.map(items, &Enum.at(&1, i)), type, format)
        end
        |> List.to_tuple()
    end
  end

  defp items_container(items, "bytes", _format), do: Nx.tensor(items, type: :u8)

  defp items_container(items, "complex", format) do
    {re, im} = Enum.unzip(items)
    {samples(re, format), samples(im, format)}
  end

  defp items_container(items, "real", format), do: samples(items, format)

  defp samples(ints, format), do: Num.from_int(Nx.tensor(ints, type: :s16), 15, format)

  @doc "Runs the chain from `states` (default: initial); returns `{outputs, states}`."
  def run(blocks, input, states \\ nil) do
    states = states || Enum.map(blocks, & &1.init)

    {out, next} =
      Enum.zip(blocks, states)
      |> Enum.reduce({input, []}, fn {b, st}, {x, acc} ->
        {y, st2} = NxFixed.Block.run(b, x, st)
        {y, [st2 | acc]}
      end)

    {out, Enum.reverse(next)}
  end

  @doc "Plain values of an output container: a list of floats, or of `{re, im}` for complex."
  def to_list({re, im}), do: Enum.zip(to_list(re), to_list(im))
  def to_list(%Fixed{} = f), do: Num.to_float(f) |> List.wrap()
  def to_list(b) when is_binary(b), do: b |> :binary.bin_to_list() |> Enum.map(&(&1 * 1.0))
  def to_list(t), do: t |> Nx.to_flat_list() |> Enum.map(&(&1 * 1.0))

  @doc "Reference values (signed bytes read as signed, complex as pairs)."
  def reference(case, data) do
    ref = data.reference

    if case["output"] && case["output"]["signed"] do
      Enum.map(ref, &if(&1 >= 128, do: &1 - 256, else: &1))
    else
      ref
    end
  end

  @doc "Largest absolute error and its index (complex: magnitude of the difference)."
  def worst(reference, signal) do
    reference
    |> Enum.zip(signal)
    |> Enum.with_index()
    |> Enum.map(fn {{r, s}, i} -> {dist(r, s), i} end)
    |> Enum.max()
  end

  defp dist({a, b}, {c, d}), do: :math.sqrt((a - c) * (a - c) + (b - d) * (b - d))
  defp dist(a, b), do: abs(a - b) * 1.0
end
