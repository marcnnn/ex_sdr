defmodule ExSdr.Digital.CaseHelper do
  @moduledoc false
  # Runs the digital family's JSON cases (test/dsp_cases/digital) through
  # ExSdr.Digital and compares with the GNU Radio data.
  alias ExSdr.Digital
  alias ExSdr.Digital.Constellation
  alias NxFixed.Num

  @cases_dir Path.expand("../../dsp_cases/digital", __DIR__)

  @doc "The case files (without GNU Radio data), for generating tests at compile time."
  def case_files,
    do:
      @cases_dir
      |> Path.join("*.json")
      |> Path.wildcard()
      |> Enum.sort()
      |> Enum.map(&JSON.decode!(File.read!(&1)))

  @doc "`{case, data}` with the GNU Radio data (cached)."
  def load(name) do
    key = {__MODULE__, :family}

    family =
      case :persistent_term.get(key, nil) do
        nil ->
          f = ExSdr.Test.GnuRadioFixture.family("digital")
          :persistent_term.put(key, f)
          f

        f ->
          f
      end

    Enum.find(family, fn {c, _} -> c["name"] == name end)
  end

  def format(nil), do: Num.float()
  def format("float"), do: Num.float()

  def format(%{"bits" => bits, "frac" => frac} = f),
    do: Num.q(bits, frac, rounding: String.to_atom(f["rounding"]), lut_bits: f["lut_bits"] || 10)

  def block(case, format) do
    %{"block" => %{"name" => name} = spec, "params" => params} = case

    opts =
      Enum.map(spec["opts"] || %{}, fn {k, v} -> {String.to_atom(k), value(k, v, params)} end)

    Digital.block(String.to_atom(name), [format: format] ++ opts)
  end

  defp value(_, "$" <> name, params), do: Map.fetch!(params, name)

  defp value(_, %{"constellation" => name}, _),
    do: apply(Constellation, String.to_atom(name), []).points

  defp value("constellation", name, _), do: String.to_atom(name)
  defp value(_, v, _), do: v

  # GNU Radio input (ints: bytes, or Q15) as block input
  def input(case, data, format) do
    case case["input"]["type"] do
      "bytes" ->
        Nx.tensor(data.input, type: :u8)

      "real" ->
        samples(data.input, format)

      "complex" ->
        {re, im} = Enum.unzip(data.input)
        {samples(re, format), samples(im, format)}
    end
  end

  defp samples(ints, format), do: Num.from_int(Nx.tensor(ints, type: :s16), 15, format)

  def output_type(case), do: get_in(case, ["output", "type"]) || case["input"]["type"]

  # block output as a flat list of ints, floats or {re, im}
  def items(out, "bytes"), do: Nx.to_flat_list(out)

  def items({re, im}, _), do: Enum.zip(items(re, nil), items(im, nil))
  def items(%NxFixed.Fixed{t: t} = x, _), do: Num.to_float(%{x | t: Nx.reshape(t, {:auto})})
  def items(%Nx.Tensor{} = t, _), do: t |> Nx.reshape({:auto}) |> Num.to_float()

  def max_abs(a, b) do
    Enum.zip_with(a, b, fn
      {x, y}, {u, v} -> :math.sqrt((x - u) * (x - u) + (y - v) * (y - v))
      x, y -> abs(x - y)
    end)
    |> Enum.max(fn -> 0.0 end)
  end

  def mismatches(a, b), do: Enum.zip(a, b) |> Enum.count(fn {x, y} -> x != y end)
end
