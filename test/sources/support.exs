defmodule ExSdr.SourcesCase do
  @moduledoc false
  # Helpers for the sources family: cases in test/dsp_cases/sources, GNU Radio
  # output in test/fixtures/gnuradio/sources (made by
  # scripts/gnuradio_golden.sh).

  alias ExSdr.Test.GnuRadioFixture, as: F
  alias NxFixed.Block
  alias NxFixed.Num

  @dir Path.expand("../dsp_cases/sources", __DIR__)
  @fixtures Path.expand("../fixtures/gnuradio/sources", __DIR__)

  defp all_cases do
    for path <- @dir |> Path.join("*.json") |> Path.wildcard() |> Enum.sort() do
      {path, F.read_case(path)}
    end
  end

  @doc "Cases with GNU Radio sample output: `[{case, data}]`."
  def cases do
    for {path, case} <- all_cases(), !case["stats_only"] do
      {case, F.load(case, path, @fixtures)}
    end
  end

  @doc """
  Cases of random blocks, for which only the statistics of GNU Radio's output are
  kept (`stats.json`, made by test/gnuradio/golden.py): `[{case, stats}]`.
  """
  def stats_cases do
    for {path, case} <- all_cases(), case["stats_only"] do
      dir = Path.join(@fixtures, case["name"])
      manifest = dir |> Path.join("manifest.json") |> File.read!() |> JSON.decode!()
      sha = :crypto.hash(:sha256, File.read!(path)) |> Base.encode16(case: :lower)
      if manifest["case_sha256"] != sha, do: raise("stale statistics for #{case["name"]}")
      {case, dir |> Path.join("stats.json") |> File.read!() |> JSON.decode!()}
    end
  end

  def stats_case_named(name),
    do: Enum.find(stats_cases(), fn {c, _} -> c["name"] == name end) || raise("no case #{name}")

  def case_named(name),
    do: Enum.find(cases(), fn {c, _} -> c["name"] == name end) || raise("no case #{name}")

  @doc "A stats-only case run in `format` (no GNU Radio samples needed)."
  def run_stats_case(case, format) do
    input =
      if case["input"]["type"] == "complex" do
        # noise free tones, quantised to Q15 like golden.py does
        for k <- 0..(case["samples"] - 1) do
          {re, im} =
            Enum.reduce(case["input"]["tones"], {0.0, 0.0}, fn t, {a, b} ->
              angle = 2 * :math.pi() * t["freq"] * k + (t["phase"] || 0.0)
              {a + t["amp"] * :math.cos(angle), b + t["amp"] * :math.sin(angle)}
            end)

          {q15(re), q15(im)}
        end
      end

    run(case, %{input: input}, format)
  end

  defp q15(x), do: x |> Kernel.*(32768) |> round() |> max(-32768) |> min(32767)

  @atom_values ~w(waveform type)

  @doc "The block of a case's `block` section in `format` (a `NxFixed.Num` format or a case format map)."
  def build(case, format) do
    %{"name" => name, "opts" => opts} = case["block"]

    opts =
      for {k, v} <- opts do
        {String.to_atom(k), if(k in @atom_values, do: String.to_atom(v), else: tuples(v))}
      end

    ExSdr.Sources.block(String.to_atom(name), [format: format] ++ opts)
  end

  # JSON has no tuples: lists of [re, im] pairs are complex values
  defp tuples([[_, _] | _] = pairs), do: Enum.map(pairs, fn [a, b] -> {a, b} end)
  defp tuples(v), do: v

  @doc "Runs the case's block in `format`; floats (1.0 = full scale), `{re, im}` for complex."
  def run(case, data, format) do
    block = build(case, format)
    to_floats(run_block(block, case, data, format))
  end

  def run_block(block, case, data, format) do
    n = case["samples"]

    if case["input"]["type"] == "none" do
      {out, _state} = ExSdr.Sources.generate(block, n)
      out
    else
      {out, _} = Block.run(block, input(case, data, format), block.init, allow_float: true)
      out
    end
  end

  def input(case, data, format) do
    if case["input"]["type"] == "complex" do
      {re, im} = Enum.unzip(data.input)
      {samples(re, format), samples(im, format)}
    else
      samples(data.input, format)
    end
  end

  def samples(ints, format), do: Num.from_int(Nx.tensor(ints, type: :s16), 15, format)

  def to_floats(nil), do: []

  def to_floats(out) do
    case Num.to_float(out) do
      {re, im} -> Enum.zip(re, im)
      values -> values
    end
  end

  def format(%{"bits" => bits, "frac" => frac} = f),
    do: Num.q(bits, frac, rounding: String.to_atom(f["rounding"]), lut_bits: f["lut_bits"] || 10)

  def format("float"), do: Num.float()

  @doc "Worst error in Q15 LSB and SNR in dB over the common prefix of two streams."
  def compare(reference, signal) do
    n = min(length(reference), length(signal))
    r = Enum.take(reference, n)
    s = Enum.take(signal, n)
    {worst, i} = F.worst_sample(r, s)
    %{worst_lsb: worst, index: i, snr: F.snr_db(r, s), n: n}
  end

  # -- statistics -------------------------------------------------------------------------------

  def mean(xs), do: Enum.sum(xs) / length(xs)

  def variance(xs) do
    m = mean(xs)
    Enum.sum(Enum.map(xs, &((&1 - m) * (&1 - m)))) / length(xs)
  end

  def kurtosis(xs) do
    m = mean(xs)
    v = variance(xs)
    Enum.sum(Enum.map(xs, &:math.pow(&1 - m, 4))) / length(xs) / (v * v)
  end

  @doc "Kolmogorov-Smirnov distance of the samples to the CDF `cdf`."
  def ks_distance(xs, cdf) do
    n = length(xs)

    xs
    |> Enum.sort()
    |> Enum.with_index()
    |> Enum.reduce(0.0, fn {x, i}, d ->
      c = cdf.(x)
      d |> max(abs((i + 1) / n - c)) |> max(abs(c - i / n))
    end)
  end

  def normal_cdf(x), do: 0.5 * :math.erfc(-x / :math.sqrt(2))

  def laplace_cdf(x) when x < 0, do: 0.5 * :math.exp(x)
  def laplace_cdf(x), do: 1 - 0.5 * :math.exp(-x)

  @doc """
  Spectral flatness (geometric over arithmetic mean) of the averaged
  periodogram of `xs` (reals) in segments of `seg` samples with a Hann
  window; 1.0 is white. Averaging `k` segments biases it to about
  `1 - 1/(2k)` for white noise.
  """
  def spectral_flatness(xs, seg \\ 256) do
    k = div(length(xs), seg)
    t = Nx.tensor(Enum.take(xs, k * seg), type: :f64, backend: Nx.BinaryBackend)
    segs = Nx.reshape(t, {k, seg})
    n = Nx.iota({seg}, type: :f64)
    win = Nx.subtract(0.5, Nx.multiply(0.5, Nx.cos(Nx.multiply(n, 2 * :math.pi() / seg))))

    power =
      segs
      |> Nx.multiply(win)
      |> Nx.fft()
      |> Nx.abs()
      |> Nx.pow(2)
      |> Nx.mean(axes: [0])
      |> Nx.slice([1], [div(seg, 2) - 1])

    Nx.to_number(Nx.exp(Nx.mean(Nx.log(power)))) / Nx.to_number(Nx.mean(power))
  end
end
