defmodule ExSdr.Carrier.TestHelper do
  @moduledoc false
  # Runs the carrier cases (test/dsp_cases/carrier/*.json, GNU Radio output in
  # test/fixtures/gnuradio/carrier/) through ExSdr.Carrier blocks.

  alias ExSdr.Digital.Constellation
  alias ExSdr.Test.GnuRadioFixture, as: F
  alias NxFixed.Num

  @rad_per_word :math.pi() / 2_147_483_648

  def root, do: Path.join(F.fixtures_dir(), "carrier")

  def case_paths,
    do: Path.wildcard(Path.join([F.cases_dir(), "carrier", "*.json"])) |> Enum.sort()

  @doc "`[{path, case}]` of the cases with `\"compare\": kind` (for generating tests at compile time)."
  def cases_of(kind) do
    for path <- case_paths(),
        case = path |> File.read!() |> JSON.decode!(),
        (case["compare"] || "ports") == kind,
        do: {path, case}
  end

  @doc "`[{variant, format}]` of a case's fixed-point variants."
  def fixed_formats(case),
    do: for(v <- case["fixed_vs_float"] || [], do: {v, format(v["format"])})

  @doc "`{case, data}` with the GNU Radio data: `data.ports` (all outputs), `data.tags`."
  def load(path) do
    case = F.read_case(path)
    data = F.load(case, path, root())
    dir = Path.join(root(), case["name"])
    tags_path = Path.join(dir, "tags.json")
    tags = if File.exists?(tags_path), do: tags_path |> File.read!() |> JSON.decode!(), else: %{}
    {case, Map.merge(data, %{ports: data.references, tags: tags})}
  end

  defp output_specs(case) do
    case["outputs"] ||
      [%{"type" => get_in(case, ["output", "type"]) || case["input"]["type"]}]
  end

  def format(%{} = f), do: F.format(f)
  def format("float"), do: Num.float()

  # -- blocks -------------------------------------------------------------------------

  @doc "Block options of a case: strings are looked up in the case's params, constellations built."
  def block_opts(case, format) do
    opts =
      for {k, v} <- case["block"]["opts"] || %{} do
        {String.to_atom(k), value(k, v, case["params"])}
      end

    [format: format] ++ opts
  end

  defp value("outputs", names, _), do: Enum.map(names, &String.to_atom/1)
  defp value("constellation", spec, _), do: constellation(spec)
  defp value("algorithm", name, _), do: String.to_atom(name)

  defp value("type", name, _), do: String.to_atom(name)
  defp value("item", name, _), do: String.to_atom(name)

  # symbol lists (training sequence, sync word): the constellation points at the
  # indices of a params list
  defp value(_, %{"from" => key, "constellation" => spec}, params) do
    c = constellation(spec)
    for i <- Map.fetch!(params, key), do: Enum.at(c.points, i)
  end

  defp value("initial_taps", taps, _), do: Enum.map(taps, &List.to_tuple/1)
  defp value(_, "$" <> name, params), do: Map.fetch!(params, name)
  defp value(_, v, params) when is_binary(v), do: Map.get(params, v, v)
  defp value(_, v, _), do: v

  @doc "A constellation spec: a name, or `{name, scale}` as a map."
  def constellation(%{"name" => name, "scale" => k}) do
    c = apply(Constellation, String.to_atom(name), [])

    Constellation.new(for({a, b} <- c.points, do: {a * k, b * k}),
      normalization: :none,
      pre_diff_code: c.pre_diff_code,
      apply_pre_diff_code: c.apply_pre_diff_code,
      rotational_symmetry: c.rotational_symmetry
    )
  end

  def constellation(name) when is_binary(name), do: apply(Constellation, String.to_atom(name), [])

  def block(case, format),
    do: ExSdr.Carrier.block(String.to_atom(case["block"]["name"]), block_opts(case, format))

  # -- input and output conversion ----------------------------------------------------------

  def input(case, data, format) do
    case case["input"]["type"] do
      "complex" ->
        {re, im} = Enum.unzip(data.input)
        x = {samples(re, format), samples(im, format)}

        case case["input"]["tags"] do
          nil -> x
          tags -> {x, flags(length(re), Enum.map(tags, & &1["offset"]))}
        end

      "real" ->
        samples(data.input, format)

      "bytes" ->
        Nx.tensor(data.input, type: :u8)
    end
  end

  # a u8 flag stream, 1 at the tagged offsets
  defp flags(n, offsets) do
    set = MapSet.new(offsets)
    Nx.tensor(for(i <- 0..(n - 1), do: if(i in set, do: 1, else: 0)), type: :u8)
  end

  defp samples(ints, format), do: Num.from_int(Nx.tensor(ints, type: :s16), 15, format)

  @doc "Runs a case's block in `format`; returns the output ports as lists (words as radians)."
  def run(case, data, format, extra_input \\ nil) do
    b = block(case, format)
    input = extra_input || input(case, data, format)
    {out, state} = ExSdr.Carrier.run(b, input)
    {ports(out, case), state}
  end

  @doc "Splits a block output into the ports of the case's \"outputs\" list as Elixir lists."
  def ports(out, case) do
    specs = output_specs(case)
    parts = flatten_ports(out)

    for {{spec, part}, _} <- Enum.zip(Enum.zip(specs, parts), specs) do
      port(part, spec)
    end
  end

  defp flatten_ports({main, extras}) when is_tuple(extras), do: [main | Tuple.to_list(extras)]
  defp flatten_ports(out), do: [out]

  defp port(part, spec) do
    if spec["word"], do: words(part), else: items(part)
  end

  def items({re, im}), do: Enum.zip(items(re), items(im))
  def items(%NxFixed.Fixed{t: t} = x), do: Num.to_float(%{x | t: Nx.reshape(t, {:auto})})
  def items(%Nx.Tensor{} = t), do: t |> Nx.reshape({:auto}) |> Nx.to_flat_list()

  def words(t),
    do: t |> Nx.reshape({:auto}) |> Nx.to_flat_list() |> Enum.map(&(&1 * @rad_per_word))

  # -- comparison ------------------------------------------------------------------------------

  @doc "Largest distance between two port lists, from `skip` on; `circular` compares angles mod 2 pi."
  def max_diff(a, b, skip \\ 0, circular \\ false) do
    a
    |> Enum.zip(b)
    |> Enum.drop(skip)
    |> Enum.map(fn
      {{x, y}, {u, v}} -> :math.sqrt((x - u) * (x - u) + (y - v) * (y - v))
      {x, u} when circular -> circ(x - u)
      {x, u} -> abs(x - u)
    end)
    |> Enum.max(fn -> 0.0 end)
  end

  defp circ(d) do
    two_pi = 2 * :math.pi()
    d = d - two_pi * Float.round(d / two_pi)
    abs(d)
  end

  def snr_db(reference, signal, skip \\ 0) do
    {p, e} =
      reference
      |> Enum.zip(signal)
      |> Enum.drop(skip)
      |> Enum.reduce({0.0, 0.0}, fn
        {{r1, r2}, {s1, s2}}, {p, e} ->
          {p + r1 * r1 + r2 * r2, e + (s1 - r1) * (s1 - r1) + (s2 - r2) * (s2 - r2)}

        {r, s}, {p, e} ->
          {p + r * r, e + (s - r) * (s - r)}
      end)

    10 * :math.log10(p / max(e, 1.0e-300))
  end

  # -- stream splitting ------------------------------------------------------------------------

  @doc """
  Runs the case's block over `data` in pieces cut at `cuts` (input item
  indices), carrying the state; the outputs of the pieces concatenated per port.
  """
  def run_split(case, data, format, cuts) do
    b = block(case, format)
    whole = input(case, data, format)
    n = length(data.input)
    bounds = Enum.zip([0 | cuts], cuts ++ [n])

    {pieces, _} =
      Enum.map_reduce(bounds, :init, fn {from, to}, state ->
        piece = slice_items(whole, from, to - from)
        {out, state} = ExSdr.Carrier.run(b, piece, state)
        {ports(out, case), state}
      end)

    pieces |> Enum.zip_with(&Enum.concat/1)
  end

  defp slice_items({{_, _} = x, flags}, from, len),
    do: {slice_items(x, from, len), slice_items(flags, from, len)}

  defp slice_items({a, b}, from, len), do: {slice_items(a, from, len), slice_items(b, from, len)}

  defp slice_items(%NxFixed.Fixed{t: t} = x, from, len),
    do: %{x | t: Nx.slice_along_axis(t, from, len)}

  defp slice_items(%Nx.Tensor{} = t, from, len), do: Nx.slice_along_axis(t, from, len)
end
