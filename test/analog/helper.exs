defmodule ExSdr.Analog.TestHelper do
  @moduledoc false
  # Runs the analog cases (test/dsp_cases/analog/*.json, GNU Radio output in
  # test/fixtures/gnuradio/analog/) through ExSdr.Analog blocks.

  alias ExSdr.Test.GnuRadioFixture, as: F
  alias NxFixed.Num

  def cases, do: F.family("analog")

  def format(%{} = f), do: F.format(f)
  def format("float"), do: Num.float()

  @doc "Block opts of a case: numbers as given, strings looked up in the case's params."
  def block_opts(case, format) do
    opts =
      for {k, v} <- case["block"]["opts"] || %{} do
        {String.to_atom(k), if(is_binary(v), do: Map.fetch!(case["params"], v), else: v)}
      end

    [format: format] ++ opts
  end

  # agc3_cc starts from reference / mean |x| over its first block; the case
  # names that as "gain": "from_input".
  def resolve_input_gain(case, input) do
    opts = case["block"]["opts"] || %{}

    if opts["gain"] == "from_input" and input do
      p = case["params"]
      d = p["iir_update_decim"]
      samples = for {re, im} <- input, do: {re / 32768, im / 32768}

      put_in(
        case,
        ["params", "init_gain"],
        ExSdr.Analog.agc3_initial_gain(samples, p["reference"], d)
      )
      |> put_in(["block", "opts", "gain"], "init_gain")
    else
      case
    end
  end

  def block(case, format),
    do: ExSdr.Analog.block(String.to_atom(case["block"]["name"]), block_opts(case, format))

  @doc """
  The steps of a case: its `"pipeline"` (blocks and `{"fir": params key,
  "decim": n}` stages, which stand in for the filters family), or its single block.
  """
  def pipeline(case), do: case["pipeline"] || [%{"block" => case["block"]}]

  defp step_block(case, %{"block" => b}, format),
    do: ExSdr.Analog.block(String.to_atom(b["name"]), block_opts(%{case | "block" => b}, format))

  defp step_block(case, %{"fir" => key}, format) do
    %NxFixed.Block{
      name: :fir_real,
      kind: :stream,
      fun: &ExSdr.Analog.TestFir.step/3,
      init: Num.zeros({length(case["params"][key]) - 1}, format),
      opts: [format: format, taps: case["params"][key]],
      item: {:real, :real},
      hardware: :feed_forward
    }
  end

  @doc "Runs the whole pipeline; `{outputs container, [state per step]}`."
  def run_container(case, input, format, states \\ nil, raw_input \\ nil) do
    case = resolve_input_gain(case, raw_input)
    steps = pipeline(case)
    blocks = Enum.map(steps, &step_block(case, &1, format))
    states = states || Enum.map(blocks, & &1.init)

    {out, new_states} =
      steps
      |> Enum.zip(blocks)
      |> Enum.zip(states)
      |> Enum.reduce({input, []}, fn {{step, b}, state}, {x, acc} ->
        {y, st} = NxFixed.Block.run(b, x, state)
        {decimate(y, step["decim"] || 1), acc ++ [st]}
      end)

    {out, new_states}
  end

  defp decimate(y, 1), do: y

  defp decimate(%NxFixed.Fixed{t: t} = y, d), do: %{y | t: decimate(t, d)}

  # every d-th item starting with the first (n a multiple of d)
  defp decimate(t, d) do
    n = Nx.axis_size(t, 0)
    Nx.slice_along_axis(t, 0, n - rem(n, d), strides: d, axis: 0)
  end

  @doc "Q15 integer input of a case as the block's input container in `format`."
  def to_input(input, format) when is_list(input) do
    case input do
      [{_, _} | _] ->
        {re, im} = Enum.unzip(input)
        {samples(re, format), samples(im, format)}

      _ ->
        samples(input, format)
    end
  end

  defp samples(ints, format), do: Num.from_int(Nx.tensor(ints, type: :s16), 15, format)

  @doc "Outputs as Elixir floats, `{re, im}` pairs for complex."
  def to_floats(outputs) do
    case Num.to_float(outputs) do
      {re, im} -> Enum.zip(re, im)
      values -> values
    end
  end

  @doc "Runs the case over `input`; `{floats, final states}`."
  def run(case, input, format) do
    {out, states} = run_container(case, to_input(input, format), format, nil, input)
    {to_floats(out), states}
  end

  @doc "Runs in pieces (sizes in `cuts`), carrying the state; the concatenated floats."
  def run_split(case, input, format, cuts) do
    # the start gain of agc3_cc is computed from the first block of the stream
    case = resolve_input_gain(case, input)

    {pieces, _} =
      Enum.map_reduce(chunks(input, cuts), nil, fn chunk, states ->
        {out, states} = run_container(case, to_input(chunk, format), format, states)
        {to_floats(out), states}
      end)

    Enum.concat(pieces)
  end

  defp chunks(input, cuts) do
    {pieces, rest} =
      Enum.map_reduce(cuts, input, fn n, rest -> {Enum.take(rest, n), Enum.drop(rest, n)} end)

    pieces ++ [rest]
  end

  @doc "Compares from sample `skip` on: `{max error in LSB of the float scale, SNR dB}`."
  def compare(reference, signal, skip \\ 0) do
    r = Enum.drop(reference, skip)
    s = Enum.drop(signal, skip)
    {worst, _} = F.worst_sample(r, s)
    {worst, F.snr_db(r, s)}
  end
end

defmodule ExSdr.Analog.TestFir do
  @moduledoc false
  # A real FIR with constant taps on real samples, standing in for the
  # filters family where a hier block of gr-analog contains one.
  alias NxFixed.Num

  def step(x, line, opts) do
    format = opts[:format]
    {acc, next} = Num.fir(x, line, opts[:taps], format)
    {Num.quantize(acc, format), next}
  end
end
