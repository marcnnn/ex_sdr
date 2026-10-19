defmodule ExSdr.Filters.TestKit do
  @moduledoc false
  # Shared by test/filters/*_test.exs and test/filters/gen_cases.exs (run it
  # with MIX_ENV=test): builds a filters block from a case file, runs it over
  # the GNU Radio input and aligns results with the reference.
  alias ExSdr.Test.GnuRadioFixture, as: F
  alias NxFixed.{Block, Num}

  def family, do: F.family("filters")

  @doc "The block of a case in `format`."
  def block(case, format) do
    %{"block" => %{"name" => name, "opts" => opts}, "params" => params} = case

    opts =
      Enum.map(opts, fn {k, v} -> {String.to_atom(k), value(k, v, params)} end) ++
        [format: format]

    ExSdr.Filters.block(String.to_atom(name), opts)
  end

  defp value(k, name, params) when is_binary(name) do
    case Map.fetch!(params, name) do
      list when is_list(list) and k in ["taps"] ->
        Enum.map(list, fn
          [r, i] -> {r, i}
          x -> x
        end)

      v ->
        v
    end
  end

  defp value(_k, v, _params), do: v

  @doc "Input items (Q15 integers from the fixture) in `format`."
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

  @doc "Runs the whole input in one call; floats, `{re, im}` for complex."
  def run(block, input), do: run(block, input, block.init) |> elem(0)

  def run(block, input, state) do
    items = items(input, block.opts[:format])
    {out, state} = Block.run(block, items, state)
    {floats(out), state}
  end

  def floats(out) do
    case Num.to_float(out) do
      {re, im} -> Enum.zip(re, im)
      values -> values
    end
  end

  @doc "Reference and signal cut to the common length after dropping `shift` items of the signal."
  def align(reference, signal, shift) do
    signal = Enum.drop(signal, shift)
    n = min(length(reference), length(signal))
    {Enum.take(reference, n), Enum.take(signal, n)}
  end

  @doc "SNRs of the float instance against GNU Radio and of each fixed variant against the float instance."
  def measure(case, data) do
    float = run(block(case, Num.float()), data.input)
    {ref, got} = align(data.reference, float, case["reference_shift"] || 0)

    fixed =
      Map.new(case["fixed_vs_float"], fn v ->
        out = run(block(case, F.format(v["format"])), data.input)
        {v["variant"], F.snr_db(float, out)}
      end)

    %{float_vs_gnuradio: F.snr_db(ref, got), fixed: fixed}
  end

  @doc """
  The width check of one step (stream) or of a frame of a few items,
  `NxFixed.analyze/3`: `{:ok, graph}` or `{:error, error}`.
  """
  def width_check(%Block{} = b) do
    n =
      case b.kind do
        :stream -> nil
        :frame -> elem(b.rate, 0) * 4
      end

    template = fn t -> Nx.to_template(Nx.tensor(t)) end
    f = b.opts[:format]
    {in_item, _} = b.item

    item = fn ->
      case n do
        nil -> NxFixed.Num.zeros({}, f)
        n -> NxFixed.Num.zeros({n}, f)
      end
    end

    x = if in_item == :complex, do: {item.(), item.()}, else: item.()

    templates =
      Enum.map([x, b.init | b.params], &Nx.Defn.Composite.traverse(&1, fn t -> template.(t) end))

    NxFixed.analyze(b.fun, templates ++ [b.opts])
  end
end
