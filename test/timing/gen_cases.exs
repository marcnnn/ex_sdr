# Writes the resampler cases of the timing family to test/dsp_cases/timing/
# (run by the shared runner test/gnuradio/golden.py). From the repository root:
#
#     mix run test/timing/gen_cases.exs
#     scripts/gnuradio_golden.sh test/fixtures/gnuradio/timing test/dsp_cases/timing
#
# The bounds are derived in the "rationale" texts; measured values are in
# docs/blocks/timing.md. The synchronizer cases (symbol timing recovery)
# are in test/dsp_cases/timing/sync/ and run by test/gnuradio/timing_golden.py.
alias ExSdr.Filters.Firdes

defmodule Gen do
  def write(case) do
    path = Path.join("test/dsp_cases/timing", case["name"] <> ".json")
    File.write!(path, JSON.encode!(case) |> pretty())
    IO.puts(path)
  end

  # JSON.encode! has no pretty printer
  defp pretty(json) do
    tmp = Path.join(System.tmp_dir!(), "timing_case_#{System.unique_integer([:positive])}.json")
    File.write!(tmp, json)
    {out, 0} = System.cmd("python3", ["-m", "json.tool", "--indent", "2", tmp])
    File.rm!(tmp)
    out
  end

  def input(type, tones, noise) do
    %{
      "type" => type,
      "tones" => for({f, a, p} <- tones, do: %{"freq" => f, "amp" => a, "phase" => p}),
      "noise_rms" => noise
    }
  end

  def variants do
    [
      %{
        "variant" => "q15_nearest",
        "format" => %{"bits" => 16, "frac" => 15, "rounding" => "nearest"}
      },
      %{
        "variant" => "q17_nearest",
        "format" => %{"bits" => 18, "frac" => 17, "rounding" => "nearest"}
      }
    ]
  end
end

complex_in =
  Gen.input("complex", [{0.013, 0.35, 0.0}, {-0.071, 0.3, 1.0}, {0.12, 0.1, 2.0}], 0.002)

real_in = Gen.input("real", [{0.011, 0.4, 0.0}, {0.083, 0.3, 1.0}], 0.002)

mmse_float = %{
  "max_error" => 2.0e-6,
  "rationale" =>
    "Positions are computed in double exactly like GNU Radio (same additions and floor), so both pick the same window and table row; the outputs differ only by f32 rounding of the 8-tap sum (GNU Radio uses a SIMD dot product): per component below 8 * 2^-24 * peak * sum|taps| (sum|taps| <= 1.06, peak 0.76) = 4e-7 for each order, the bound allows 2e-6."
}

mmse_fixed = %{
  "max_error_lsb" => 3.5,
  "rationale" =>
    "The ratio is an f32 value (exact on the 2^-32 position grid), so positions and table rows equal the float instance. Taps rounded to the format: sum of |tap rounding error| <= 8 * 0.5 LSB acting on a peak of 0.76 FS, plus 0.5 LSB output rounding: <= 3.5 LSB per component."
}

for {name, block, ratio, phase, input} <- [
      {"mmse_resampler_cc_0p75", "mmse_resampler_cc", 0.75, 0.3, complex_in},
      {"mmse_resampler_cc_1p375", "mmse_resampler_cc", 1.375, 0.0, complex_in},
      {"mmse_resampler_ff_2p5", "mmse_resampler_ff", 2.5, 0.6, real_in},
      {"mmse_resampler_ff_0p3", "mmse_resampler_ff", 0.3, 0.0, real_in}
    ] do
  Gen.write(%{
    "name" => name,
    "description" => "#{block} at resampling ratio #{ratio}, initial mu #{phase}",
    "samples" => 2048,
    "seed" => 7070,
    "input" => input,
    "output" => %{"type" => input["type"]},
    "params" => %{"ratio" => ratio, "phase" => phase},
    "block" => %{
      "name" => block,
      "opts" => %{"resamp_ratio" => "ratio", "phase_shift" => "phase"}
    },
    "gnuradio" => [%{"block" => "filter.#{block}", "args" => ["phase", "ratio"]}],
    "float_vs_gnuradio" => mmse_float,
    "fixed_vs_float" => Enum.map(Gen.variants(), &Map.merge(&1, mmse_fixed))
  })
end

arb_float = %{
  "max_error" => 5.0e-6,
  "rationale" =>
    "Filter arm j and the f32 accumulator acc evolve exactly as in GNU Radio (same f32 additions, floor and fmod), so the same arms are used; the outputs o0 + o1 * acc differ only by f32 rounding of the two arm sums (SIMD order in GNU Radio) and the final multiply-add: below 4 * ntaps * 2^-24 * peak * sum|arm| (sum|arm| <= 1.3) = 1e-6 per component, the bound allows 5e-6."
}

arb_fixed = %{
  "max_error_lsb" => 6.0,
  "rationale" =>
    "Fixed point: acc is a 32 bit fraction, exact, while GNU Radio and the float instance round acc + flt_rate to f32; with flt_rate a multiple of 2^-8 (the cases) both agree. Error per component: tap rounding (ntaps taps of o0 and of o1, 0.5 LSB each, on a peak of 0.8 FS: <= ntaps * 0.8 LSB for o0 + o1 * acc), o1 rounded to the format (0.5 LSB) and acc to 18 bits, and 0.5 LSB output rounding; bound 6 LSB for 4-6 taps per arm."
}

for {name, block, rate, n, input} <- [
      {"pfb_arb_resampler_ccf_1p5059", "pfb_arb_resampler_ccf", 32 / 21.25, 32, complex_in},
      {"pfb_arb_resampler_ccf_0p8", "pfb_arb_resampler_ccf", 0.8, 32, complex_in},
      {"pfb_arb_resampler_fff_2p56", "pfb_arb_resampler_fff", 2.56, 32, real_in}
    ] do
  # prototype at n times the input rate; pass band below 0.4 of the input
  # (or output) sample rate
  cutoff = 0.4 * min(1.0, rate)
  taps = Firdes.low_pass(n * 1.0, n * 1.0, cutoff, 0.2 * min(1.0, rate), :blackman)

  Gen.write(%{
    "name" => name,
    "description" =>
      "#{block} at rate #{Float.round(rate, 5)} with #{n} arms (#{length(taps)} taps)",
    "samples" => 2048,
    "seed" => 7171,
    "input" => input,
    "output" => %{"type" => input["type"]},
    "params" => %{"rate" => rate, "taps" => taps, "nfilts" => n},
    "block" => %{
      "name" => block,
      "opts" => %{"rate" => "rate", "taps" => "taps", "filter_size" => "nfilts"}
    },
    "gnuradio" => [%{"block" => "filter.#{block}", "args" => ["rate", "taps", "nfilts"]}],
    "float_vs_gnuradio" => arb_float,
    "fixed_vs_float" => Enum.map(Gen.variants(), &Map.merge(&1, arb_fixed))
  })
end
