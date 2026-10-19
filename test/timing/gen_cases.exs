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
    File.write!(path, pretty(JSON.encode!(case)))
    IO.puts(path)
  end

  # JSON.encode! has no pretty printer
  def pretty(json) do
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

# -- synchronizer cases (test/gnuradio/timing_golden.py) ------------------------------------

defmodule SyncGen do
  def write(case) do
    File.mkdir_p!("test/dsp_cases/timing/sync")
    path = Path.join("test/dsp_cases/timing/sync", case["name"] <> ".json")
    File.write!(path, Gen.pretty(JSON.encode!(case)))
    IO.puts(path)
  end

  def py(true), do: "True"
  def py(false), do: "False"
  def py(x) when is_float(x), do: :erlang.float_to_binary(x, [:short])
  def py(x), do: to_string(x)

  @ted_py %{
    "mueller_muller" => "TED_MUELLER_AND_MULLER",
    "modified_mueller_muller" => "TED_MOD_MUELLER_AND_MULLER",
    "zero_crossing" => "TED_ZERO_CROSSING",
    "gardner" => "TED_GARDNER",
    "early_late" => "TED_EARLY_LATE",
    "dandrea_mengali_gen_msk" => "TED_DANDREA_AND_MENGALI_GEN_MSK",
    "mengali_dandrea_gmsk" => "TED_MENGALI_AND_DANDREA_GMSK",
    "signal_times_slope_ml" => "TED_SIGNAL_TIMES_SLOPE_ML",
    "signum_times_slope_ml" => "TED_SIGNUM_TIMES_SLOPE_ML"
  }

  @ir_py %{"mmse" => "IR_MMSE_8TAP", "pfb_no_mf" => "IR_PFB_NO_MF", "pfb_mf" => "IR_PFB_MF"}

  def symbol_sync(name, kind, signal, o) do
    o =
      Map.merge(
        %{
          "loop_bw" => 0.045,
          "damping_factor" => 1.0,
          "ted_gain" => 1.0,
          "max_deviation" => 1.5,
          "osps" => 1,
          "constellation" => "bpsk",
          "resampling_type" => "mmse",
          "n_filters" => 128
        },
        o
      )

    taps = if o["resampling_type"] == "pfb_mf", do: o["taps_spec"], else: nil

    gr =
      "digital.symbol_sync_#{kind}(digital.#{@ted_py[o["ted_type"]]}, #{py(o["sps"])}, " <>
        "#{py(o["loop_bw"])}, #{py(o["damping_factor"])}, #{py(o["ted_gain"])}, " <>
        "#{py(o["max_deviation"])}, #{o["osps"]}, digital.constellation_#{o["constellation"]}().base(), " <>
        "digital.#{@ir_py[o["resampling_type"]]}, #{o["n_filters"]}, taps)"

    %{
      "name" => name,
      "signal" => signal,
      "taps" => taps,
      "block" => %{"name" => "symbol_sync_#{kind}", "opts" => Map.delete(o, "taps_spec")},
      "gnuradio" => gr
    }
  end
end

rrc32 = %{
  "kind" => "rrc",
  "gain" => 32.0,
  "fs" => 128.0,
  "symbol_rate" => 1.0,
  "alpha" => 0.35,
  "ntaps" => 32 * 4 * 11
}

sync_cases =
  [
    SyncGen.symbol_sync("symbol_sync_cc_gardner", "cc", "qpsk_rc_sps4", %{
      "ted_type" => "gardner",
      "sps" => 4.0
    }),
    SyncGen.symbol_sync("symbol_sync_cc_mueller_muller", "cc", "qpsk_rc_sps4", %{
      "ted_type" => "mueller_muller",
      "sps" => 4.0,
      "constellation" => "qpsk"
    }),
    SyncGen.symbol_sync("symbol_sync_cc_mod_mueller_muller", "cc", "qpsk_rc_sps4", %{
      "ted_type" => "modified_mueller_muller",
      "sps" => 4.0,
      "constellation" => "qpsk"
    }),
    SyncGen.symbol_sync("symbol_sync_cc_zero_crossing", "cc", "qpsk_rc_sps4", %{
      "ted_type" => "zero_crossing",
      "sps" => 4.0,
      "constellation" => "qpsk"
    }),
    SyncGen.symbol_sync("symbol_sync_cc_early_late", "cc", "qpsk_rc_sps4", %{
      "ted_type" => "early_late",
      "sps" => 4.0
    }),
    SyncGen.symbol_sync("symbol_sync_cc_signal_times_slope", "cc", "qpsk_rc_sps4", %{
      "ted_type" => "signal_times_slope_ml",
      "sps" => 4.0
    }),
    SyncGen.symbol_sync("symbol_sync_cc_signum_times_slope", "cc", "qpsk_rc_sps4", %{
      "ted_type" => "signum_times_slope_ml",
      "sps" => 4.0
    }),
    SyncGen.symbol_sync("symbol_sync_cc_gmsk", "cc", "msk_sps4", %{
      "ted_type" => "mengali_dandrea_gmsk",
      "sps" => 4.0
    }),
    SyncGen.symbol_sync("symbol_sync_cc_gen_msk", "cc", "msk_sps4", %{
      "ted_type" => "dandrea_mengali_gen_msk",
      "sps" => 4.0
    }),
    SyncGen.symbol_sync("symbol_sync_cc_gardner_osps2", "cc", "qpsk_rc_sps4", %{
      "ted_type" => "gardner",
      "sps" => 4.0,
      "osps" => 2
    }),
    SyncGen.symbol_sync("symbol_sync_cc_gardner_sps3p3", "cc", "qpsk_rc_sps3p3", %{
      "ted_type" => "gardner",
      "sps" => 3.3
    }),
    SyncGen.symbol_sync("symbol_sync_cc_gardner_pfb_no_mf", "cc", "qpsk_rc_sps4", %{
      "ted_type" => "gardner",
      "sps" => 4.0,
      "resampling_type" => "pfb_no_mf",
      "n_filters" => 32
    }),
    SyncGen.symbol_sync("symbol_sync_cc_sts_pfb_mf", "cc", "qpsk_rrc_sps4", %{
      "ted_type" => "signal_times_slope_ml",
      "sps" => 4.0,
      "resampling_type" => "pfb_mf",
      "n_filters" => 32,
      "taps_spec" => rrc32
    }),
    SyncGen.symbol_sync("symbol_sync_ff_gardner", "ff", "bpsk_rc_sps4_real", %{
      "ted_type" => "gardner",
      "sps" => 4.0
    }),
    SyncGen.symbol_sync("symbol_sync_ff_mueller_muller", "ff", "bpsk_rc_sps4_real", %{
      "ted_type" => "mueller_muller",
      "sps" => 4.0
    }),
    SyncGen.symbol_sync("symbol_sync_ff_mod_mueller_muller", "ff", "bpsk_rc_sps4_real", %{
      "ted_type" => "modified_mueller_muller",
      "sps" => 4.0
    }),
    SyncGen.symbol_sync("symbol_sync_ff_signum_times_slope", "ff", "bpsk_rc_sps4_real", %{
      "ted_type" => "signum_times_slope_ml",
      "sps" => 4.0
    }),
    SyncGen.symbol_sync("symbol_sync_ff_gardner_pfb_mf", "ff", "bpsk_rrc_sps4_real", %{
      "ted_type" => "gardner",
      "sps" => 4.0,
      "resampling_type" => "pfb_mf",
      "n_filters" => 32,
      "taps_spec" => rrc32
    }),
    %{
      "name" => "clock_recovery_mm_cc",
      "signal" => "qpsk_rc_sps4",
      "block" => %{
        "name" => "clock_recovery_mm_cc",
        "opts" => %{
          "omega" => 4.0,
          "gain_omega" => 0.0076562,
          "mu" => 0.5,
          "gain_mu" => 0.175,
          "omega_relative_limit" => 0.005
        }
      },
      "gnuradio" => "digital.clock_recovery_mm_cc(4.0, 0.0076562, 0.5, 0.175, 0.005)"
    },
    %{
      "name" => "clock_recovery_mm_ff",
      "signal" => "bpsk_rc_sps4_real",
      "block" => %{
        "name" => "clock_recovery_mm_ff",
        "opts" => %{
          "omega" => 4.0,
          "gain_omega" => 0.0076562,
          "mu" => 0.5,
          "gain_mu" => 0.175,
          "omega_relative_limit" => 0.005
        }
      },
      "gnuradio" => "digital.clock_recovery_mm_ff(4.0, 0.0076562, 0.5, 0.175, 0.005)"
    },
    %{
      "name" => "pfb_clock_sync_ccf",
      "signal" => "qpsk_rrc_sps4",
      "taps" => rrc32,
      "block" => %{
        "name" => "pfb_clock_sync_ccf",
        "opts" => %{
          "sps" => 4.0,
          "loop_bw" => 0.0628,
          "filter_size" => 32,
          "init_phase" => 16.0,
          "max_rate_deviation" => 1.5,
          "osps" => 1
        }
      },
      "gnuradio" => "digital.pfb_clock_sync_ccf(4.0, 0.0628, taps, 32, 16, 1.5, 1)"
    },
    %{
      "name" => "pfb_clock_sync_ccf_osps2",
      "signal" => "qpsk_rrc_sps4",
      "taps" => rrc32,
      "block" => %{
        "name" => "pfb_clock_sync_ccf",
        "opts" => %{
          "sps" => 4.0,
          "loop_bw" => 0.0628,
          "filter_size" => 32,
          "init_phase" => 16.0,
          "max_rate_deviation" => 1.5,
          "osps" => 2
        }
      },
      "gnuradio" => "digital.pfb_clock_sync_ccf(4.0, 0.0628, taps, 32, 16, 1.5, 2)"
    },
    %{
      "name" => "pfb_clock_sync_fff",
      "signal" => "bpsk_rrc_sps4_real",
      "taps" => rrc32,
      "block" => %{
        "name" => "pfb_clock_sync_fff",
        "opts" => %{
          "sps" => 4.0,
          "loop_bw" => 0.0628,
          "filter_size" => 32,
          "init_phase" => 16.0,
          "max_rate_deviation" => 1.5,
          "osps" => 1
        }
      },
      "gnuradio" => "digital.pfb_clock_sync_fff(4.0, 0.0628, taps, 32, 16, 1.5, 1)"
    },
    %{
      "name" => "msk_timing_recovery_cc",
      "signal" => "msk_sps4",
      "block" => %{
        "name" => "msk_timing_recovery_cc",
        "opts" => %{"sps" => 4.0, "gain" => 0.05, "limit" => 0.1, "osps" => 1}
      },
      "gnuradio" => "digital.msk_timing_recovery_cc(4.0, 0.05, 0.1, 1)"
    },
    %{
      "name" => "msk_timing_recovery_cc_osps2",
      "signal" => "msk_sps4",
      "block" => %{
        "name" => "msk_timing_recovery_cc",
        "opts" => %{"sps" => 4.0, "gain" => 0.05, "limit" => 0.1, "osps" => 2}
      },
      "gnuradio" => "digital.msk_timing_recovery_cc(4.0, 0.05, 0.1, 2)"
    }
  ]

coarse? = fn c ->
  o = c["block"]["opts"]

  String.starts_with?(c["name"], "pfb_clock_sync") or
    o["resampling_type"] in ["pfb_no_mf", "pfb_mf"]
end

# Q17 as well for a few cases (each fixed-point run compiles its own kernel)
q17 =
  ~w(symbol_sync_cc_gardner symbol_sync_ff_gardner_pfb_mf clock_recovery_mm_cc pfb_clock_sync_ccf
         msk_timing_recovery_cc)

for c <- sync_cases do
  {max_gr, max_fx, rms_fx, arms} =
    if coarse?.(c), do: {0.015, 0.015, 2.5e-3, 32}, else: {0.004, 0.005, 1.5e-3, 128}

  float = %{
    "lock_after" => 200,
    "median_error" => 1.0e-6,
    "max_error" => max_gr,
    "max_fraction_over_1e-4" => 0.15,
    "mer_db_tolerance" => 0.1,
    "rationale" =>
      "The loop and position arithmetic is GNU Radio's, step by step in f32 (f64 where GNU Radio uses double), " <>
        "so both walk the same positions; the interpolator sums differ by f32 rounding order (GNU Radio uses SIMD " <>
        "dot products), about 1e-7: median error <= 1e-6. When a fractional delay lies that close to the boundary " <>
        "between two of the #{arms} table rows, the two pick neighbouring rows for that sample: an error up to " <>
        "max|slope| / #{arms} (slope <= 0.9 per sample here, #{Float.round(0.9 / arms, 4)}), and a loop state difference of " <>
        "the same order that decays with the loop (the bound allows #{max_gr}). Such events are rare (fraction of " <>
        "outputs off by more than 1e-4 <= 0.15, measured 0-0.10). After lock (#{200} outputs) the modulation error " <>
        "ratio (MER, over the common outputs) of both is equal within 0.1 dB."
  }

  fixed =
    for {v, bits, frac} <- [{"q15_nearest", 16, 15}, {"q17_nearest", 18, 17}],
        v == "q15_nearest" or c["name"] in q17 do
      %{
        "variant" => v,
        "format" => %{"bits" => bits, "frac" => frac, "rounding" => "nearest"},
        "max_error" => max_fx,
        "rms_error" => rms_fx,
        "mer_db_tolerance" => 0.2,
        "rationale" =>
          "Fixed point keeps positions and periods as exact 32 bit fractions instead of f32, rounds samples and " <>
            "taps to the format and the loop gains to 22 bits, so its loop follows a slightly different trajectory. " <>
            "After lock (200 outputs) the timing estimates differ by a fraction of a table step: output differences " <>
            "stay within the interpolation over one or two of the #{arms} rows (max <= #{max_fx}, rms <= #{rms_fx}; " <>
            "measured max 0.0023-0.01, rms 1e-4-1.5e-3), and the MER is equal within 0.2 dB (measured <= 0.06 dB). " <>
            "The output count is the same."
      }
    end

  c
  |> Map.put("float_vs_gnuradio", float)
  |> Map.put("fixed_vs_float", fixed)
  |> Map.reject(fn {_, v} -> is_nil(v) end)
  |> SyncGen.write()
end
