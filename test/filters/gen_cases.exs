# Writes the GNU Radio comparison cases of the filters family to
# test/dsp_cases/filters/. Run from the repository root:
#
#     mix run test/filters/gen_cases.exs             # cases with analytic error bounds
#     scripts/gnuradio_golden.sh test/fixtures/gnuradio/filters test/dsp_cases/filters
#     mix run test/filters/gen_cases.exs measure     # adds SNR minima from the measured runs
#     scripts/gnuradio_golden.sh test/fixtures/gnuradio/filters test/dsp_cases/filters
#
# The error bounds (max_error_lsb) are derived here from the tap rounding
# errors and the input peak; the text of each "rationale" says how. The
# SNR minima are measured minus 1 dB (the second step), as for
# test/dsp_cases/fir_ccf_31tap.json.
Code.require_file("test/filters/test_kit.exs")
alias ExSdr.Filters.Firdes

defmodule Gen do
  import Bitwise
  @lsb 32768

  def input(type, tones, noise, extra \\ %{}) do
    Map.merge(
      %{
        "type" => type,
        "tones" => for({f, a, p} <- tones, do: %{"freq" => f, "amp" => a, "phase" => p}),
        "noise_rms" => noise
      },
      extra
    )
  end

  # Peak per component: tones add up, noise is below 5 sigma in 4000 samples.
  def peak(input) do
    tones = input["tones"] |> Enum.map(& &1["amp"]) |> Enum.sum()
    steps = (input["steps"] || []) |> Enum.map(&abs(&1["amp"])) |> Enum.sum()
    tones + steps + 5 * input["noise_rms"]
  end

  def l1_rounding_error(values, tf) do
    values
    |> Enum.map(fn v -> abs(round(v * (1 <<< tf)) / (1 <<< tf) - v) end)
    |> Enum.sum()
  end

  def parts(taps) do
    {Enum.map(taps, fn
       {r, _} -> r
       r -> r
     end),
     Enum.map(taps, fn
       {_, i} -> i
       _ -> 0.0
     end)}
  end

  def l1(values), do: values |> Enum.map(&abs/1) |> Enum.sum()

  def fmt(x), do: :erlang.float_to_binary(x * 1.0, [:short])

  @doc "Per component error of the fixed FIR taps (value units, per unit input peak)."
  def tap_error({in_type, taps}, tf) do
    {tr, ti} = parts(taps)
    complex_taps = Enum.any?(taps, &is_tuple/1)
    er = l1_rounding_error(tr, tf)
    ei = l1_rounding_error(ti, tf)

    cond do
      complex_taps and in_type == :complex ->
        {er + ei, "re + im tap errors #{fmt(er)} + #{fmt(ei)}"}

      complex_taps ->
        {max(er, ei), "tap errors #{fmt(er)} (re), #{fmt(ei)} (im)"}

      true ->
        {er, "sum of |tap rounding error| #{fmt(er)}"}
    end
  end

  def fixed_variants do
    [
      {"q15_nearest", 16, 15, "nearest", 0.5},
      {"q15_truncate", 16, 15, "truncate", 1.0},
      {"q17_nearest", 18, 17, "nearest", 0.5}
    ]
  end

  def variant_json({name, bits, frac, rounding, _}, bound, why) do
    %{
      "variant" => name,
      "format" => %{"bits" => bits, "frac" => frac, "rounding" => rounding, "lut_bits" => 10},
      "snr_db_min" => 0.0,
      "max_error_lsb" => bound,
      "rationale" => why
    }
  end

  # FIR style bounds. `extra` is added per component in Q15 LSB (e.g. NCO).
  def fir_case(opts) do
    %{peak: a, taps: taps, in_type: in_type, out_complex: out_complex} = opts
    tfd = Map.get(opts, :tap_frac_extra, 0)
    sqrt = if out_complex, do: :math.sqrt(2), else: 1.0

    variants =
      for {_, _, frac, _, r} = v <- fixed_variants() do
        tf = frac + tfd
        {err, desc} = tap_error({in_type, taps}, tf)
        comp = @lsb * (r / (1 <<< frac) + a * err) + 0.02

        why =
          "Taps rounded to Q#{tf} (#{desc}) acting on an input peak of #{fmt(a)} FS per component, " <>
            "plus #{r} LSB of output rounding (Q#{frac}) and 0.02 Q15 LSB for the f32 reference itself; per component <= #{fmt(comp)} Q15 LSB" <>
            if(out_complex, do: ", x sqrt(2) for the complex magnitude", else: "") <>
            ". SNR minimum is the measured value minus 1 dB."

        variant_json(v, Float.round(comp * sqrt + 1.0e-9, 3), why)
      end

    {tr, ti} = parts(taps)
    n = length(tr)
    sum = l1(tr) + l1(ti)
    u = :math.pow(2, -24)
    bound = Float.round(@lsb * 2 * n * u * a * sum * sqrt, 4)

    {bound,
     "f32 rounding in an #{n}-tap sum: each of the n additions errs by at most 2^-24 of the partial sum (<= peak * sum|taps| = #{fmt(a * sum)}); " <>
       "twice that covers both evaluation orders (GNU Radio uses SIMD kernels): #{fmt(bound)} Q15 LSB. SNR minimum is the measured value minus 1 dB.",
     variants}
  end

  def case_map(
        name,
        description,
        samples,
        seed,
        input,
        params,
        block,
        gnuradio,
        bound_info,
        extra \\ %{}
      ) do
    {bound, why, variants} = bound_info

    Map.merge(
      %{
        "name" => name,
        "description" => description,
        "samples" => samples,
        "seed" => seed,
        "input" => input,
        "params" => params,
        "block" => block,
        "gnuradio" => gnuradio,
        "float_vs_gnuradio" => %{
          "max_error_lsb" => bound,
          "snr_db_min" => 0.0,
          "rationale" => why
        },
        "fixed_vs_float" => variants
      },
      extra
    )
  end

  def tuples(taps),
    do:
      Enum.map(taps, fn
        {r, i} -> [r, i]
        r -> r
      end)

  # Butterworth low-pass, 2nd order (bilinear transform): new-style fb taps [1, a1, a2].
  def butter2(fc) do
    w = :math.tan(:math.pi() * fc)
    k1 = :math.sqrt(2) * w
    k2 = w * w
    a0 = 1 + k1 + k2
    {[k2 / a0, 2 * k2 / a0, k2 / a0], [1.0, 2 * (k2 - 1) / a0, (1 - k1 + k2) / a0]}
  end

  # Impulse response of the IIR in the "added feedback" form, n samples.
  def impulse(ff, fb, n) do
    Enum.reduce(0..(n - 1), [], fn k, ys ->
      xs = for {c, j} <- Enum.with_index(ff), k - j == 0, do: c
      fbs = for {c, j} <- Enum.with_index(fb), Enum.at(ys, j) != nil, do: c * Enum.at(ys, j)
      [Enum.sum(xs) + Enum.sum(fbs) | ys]
    end)
    |> Enum.reverse()
  end

  # Bound for an IIR block in fixed point: tap rounding changes the response
  # (||h_q - h||_1 * peak) and every step adds a rounding error that the
  # feedback shapes by g = 1 / A_q(z) (||g||_1 * r).
  def iir_case(a, ff, fb, name_suffix_complex) do
    n = 4000
    h = impulse(ff, fb, n)
    u = :math.pow(2, -24)

    variants =
      for {_, _, frac, _, r} = v <- fixed_variants() do
        q = fn c -> round(c * (1 <<< frac)) / (1 <<< frac) end
        qff = Enum.map(ff, q)
        qfb = Enum.map(fb, q)
        hq = impulse(qff, qfb, n)
        g = impulse([1.0], qfb, n)
        dh = Enum.zip(h, hq) |> Enum.map(fn {x, y} -> abs(x - y) end) |> Enum.sum()
        gain = l1(g)
        comp = @lsb * (a * dh + r / (1 <<< frac) * gain) + 0.02
        sqrt = if name_suffix_complex, do: :math.sqrt(2), else: 1.0

        why =
          "Coefficients rounded to Q#{frac} change the impulse response by ||dh||_1 = #{fmt(dh)} (times the input peak #{fmt(a)}), " <>
            "and each output rounding (#{r} LSB of Q#{frac}, plus 0.02 Q15 LSB for the f32 reference) is shaped by the feedback 1/A(z), ||g||_1 = #{fmt(gain)}: " <>
            "<= #{fmt(comp)} Q15 LSB per component. SNR minimum is the measured value minus 1 dB."

        variant_json(v, Float.round(comp * sqrt + 1.0e-9, 3), why)
      end

    gh = l1(impulse([1.0], fb, n))
    ymax = a * l1(h)
    bound = Float.round(@lsb * u * 4 * (l1(ff) * a + l1(fb) * ymax) * gh, 4)

    {bound,
     "GNU Radio keeps the state in double, this is f32: each step rounds the sum (terms of total size #{fmt(l1(ff) * a + l1(fb) * ymax)}) by 4 * 2^-24 at most, " <>
       "amplified by ||1/A(z)||_1 = #{fmt(gh)}: #{fmt(bound)} Q15 LSB. SNR minimum is the measured value minus 1 dB.",
     variants}
  end
end

complex_in = Gen.input("complex", [{0.03, 0.4, 0.0}, {0.3, 0.3, 1.0}], 0.001)
real_in = Gen.input("real", [{0.03, 0.4, 0.0}, {0.3, 0.3, 1.0}], 0.001)

dc_in =
  Gen.input("real", [{0.02, 0.3, 0.0}], 0.001, %{"steps" => [%{"at" => 1024, "amp" => 0.25}]})

dc_in_c =
  Gen.input("complex", [{0.02, 0.3, 0.0}], 0.001, %{"steps" => [%{"at" => 1024, "amp" => 0.25}]})

a_in = Gen.peak(complex_in)

cases = []

# FIR family ---------------------------------------------------------------------------
t = Firdes.low_pass(1.0, 1.0, 0.1, 0.08, :hamming)

cases =
  cases ++
    [
      Gen.case_map(
        "fir_fff_hamming",
        "low-pass FIR on real samples vs fir_filter_fff",
        4096,
        2001,
        real_in,
        %{"taps" => t, "decim" => 1},
        %{"name" => "fir_filter_fff", "opts" => %{"taps" => "taps", "decimation" => "decim"}},
        [%{"block" => "filter.fir_filter_fff", "args" => ["decim", "taps"]}],
        Gen.fir_case(%{peak: a_in, taps: t, in_type: :real, out_complex: false})
      )
    ]

t = Firdes.low_pass(1.0, 1.0, 0.08, 0.04, :blackman_harris)

cases =
  cases ++
    [
      Gen.case_map(
        "fir_ccf_decim4",
        "decimating low-pass (105 taps, Blackman-Harris) on complex samples vs fir_filter_ccf, decimation 4",
        4096,
        2002,
        complex_in,
        %{"taps" => t, "decim" => 4},
        %{"name" => "fir_filter_ccf", "opts" => %{"taps" => "taps", "decimation" => "decim"}},
        [%{"block" => "filter.fir_filter_ccf", "args" => ["decim", "taps"]}],
        Gen.fir_case(%{peak: a_in, taps: t, in_type: :complex, out_complex: true})
      )
    ]

t = Firdes.complex_band_pass(1.0, 1.0, 0.05, 0.35, 0.06, :hamming)

cases =
  cases ++
    [
      Gen.case_map(
        "fir_ccc_band_decim2",
        "complex band-pass taps on complex samples vs fir_filter_ccc, decimation 2",
        4096,
        2003,
        complex_in,
        %{"taps" => Gen.tuples(t), "decim" => 2},
        %{"name" => "fir_filter_ccc", "opts" => %{"taps" => "taps", "decimation" => "decim"}},
        [
          %{
            "block" => "filter.fir_filter_ccc",
            "args" => ["decim", "[a + 1j*b for a, b in taps]"]
          }
        ],
        Gen.fir_case(%{peak: a_in, taps: t, in_type: :complex, out_complex: true})
      )
    ]

t = Firdes.complex_band_pass(1.0, 1.0, -0.2, 0.1, 0.05, :hamming)

cases =
  cases ++
    [
      Gen.case_map(
        "fir_fcc_band",
        "complex taps on real samples vs fir_filter_fcc",
        4096,
        2004,
        real_in,
        %{"taps" => Gen.tuples(t), "decim" => 1},
        %{"name" => "fir_filter_fcc", "opts" => %{"taps" => "taps", "decimation" => "decim"}},
        [
          %{
            "block" => "filter.fir_filter_fcc",
            "args" => ["decim", "[a + 1j*b for a, b in taps]"]
          }
        ],
        Gen.fir_case(%{peak: a_in, taps: t, in_type: :real, out_complex: true}),
        %{"output" => %{"type" => "complex"}}
      )
    ]

t = Firdes.low_pass(1.0, 1.0, 0.2, 0.1, :hamming)

cases =
  cases ++
    [
      Gen.case_map(
        "fft_filter_ccf_decim2",
        "fft_filter_ccf (decimation 2) vs the direct FIR",
        4096,
        2005,
        complex_in,
        %{"taps" => t, "decim" => 2},
        %{"name" => "fft_filter_ccf", "opts" => %{"taps" => "taps", "decimation" => "decim"}},
        [%{"block" => "filter.fft_filter_ccf", "args" => ["decim", "taps", 1]}],
        Gen.fir_case(%{peak: a_in, taps: t, in_type: :complex, out_complex: true})
      )
    ]

# interpolation, resampling ------------------------------------------------------------
t = Firdes.low_pass(3.0, 3.0, 0.45, 0.25, :hamming)

cases =
  cases ++
    [
      Gen.case_map(
        "interp_fir_fff_x3",
        "interpolation by 3 with a low-pass on real samples vs interp_fir_filter_fff",
        4096,
        2006,
        real_in,
        %{"taps" => t, "interp" => 3},
        %{
          "name" => "interp_fir_filter_fff",
          "opts" => %{"taps" => "taps", "interpolation" => "interp"}
        },
        [%{"block" => "filter.interp_fir_filter_fff", "args" => ["interp", "taps"]}],
        Gen.fir_case(%{peak: a_in, taps: t, in_type: :real, out_complex: false})
      )
    ]

t = Firdes.root_raised_cosine(4.0, 4.0, 1.0, 0.35, 41)

cases =
  cases ++
    [
      Gen.case_map(
        "interp_fir_ccf_x4_rrc",
        "interpolation by 4 with a root raised cosine on complex samples vs interp_fir_filter_ccf",
        4096,
        2007,
        complex_in,
        %{"taps" => t, "interp" => 4},
        %{
          "name" => "interp_fir_filter_ccf",
          "opts" => %{"taps" => "taps", "interpolation" => "interp"}
        },
        [%{"block" => "filter.interp_fir_filter_ccf", "args" => ["interp", "taps"]}],
        Gen.fir_case(%{peak: a_in, taps: t, in_type: :complex, out_complex: true})
      )
    ]

resample = fn name, desc, seed, kind, in_spec, i, d, taps, cplx ->
  k = max(div(length(taps) + i - 1, i), 2)
  shift = (k - 1) * i
  letters = if cplx, do: "ccf", else: "fff"
  a = Gen.peak(in_spec)

  Gen.case_map(
    name,
    desc,
    4092,
    seed,
    in_spec,
    %{"taps" => taps, "interp" => i, "decim" => d, "phase" => rem(shift, d)},
    %{
      "name" => "rational_resampler_#{letters}",
      "opts" => %{
        "taps" => "taps",
        "interpolation" => "interp",
        "decimation" => "decim",
        "phase" => "phase"
      }
    },
    [%{"block" => "filter.rational_resampler_#{letters}", "args" => ["interp", "decim", "taps"]}],
    Gen.fir_case(%{peak: a, taps: taps, in_type: kind, out_complex: cplx}),
    %{
      "reference_shift" => div(shift, d),
      "reference_shift_note" =>
        "GNU Radio's first output is the full-rate filter output at position (K-1)*I = #{shift} (K = taps per branch); ours counts from 0 with :phase = #{rem(shift, d)}, so GNU Radio output m is our output m + #{div(shift, d)}."
    }
  )
end

cases =
  cases ++
    [
      resample.(
        "rational_resampler_ccf_3_2",
        "rational resampler 3/2 on complex samples vs rational_resampler_ccf",
        2008,
        :complex,
        complex_in,
        3,
        2,
        Firdes.low_pass(3.0, 3.0, 0.45, 0.2, :hamming),
        true
      ),
      resample.(
        "rational_resampler_fff_2_3",
        "rational resampler 2/3 on real samples vs rational_resampler_fff",
        2009,
        :real,
        real_in,
        2,
        3,
        Firdes.low_pass(2.0, 2.0, 0.3, 0.15, :hamming),
        false
      ),
      resample.(
        "rational_resampler_ccf_5_3",
        "rational resampler 5/3 on complex samples vs rational_resampler_ccf",
        2010,
        :complex,
        complex_in,
        5,
        3,
        Firdes.low_pass(5.0, 5.0, 0.5, 0.3, :hamming),
        true
      )
    ]

# frequency translating FIR -----------------------------------------------------------------
xlating = fn name, desc, seed, kind, decim, fc, t ->
  ctaps = ExSdr.Filters.xlating_taps(t, fc, 1.0)
  in_spec = if kind == :complex, do: complex_in, else: real_in
  a = Gen.peak(in_spec)
  {bound, why, variants} = Gen.fir_case(%{peak: a, taps: ctaps, in_type: kind, out_complex: true})
  n_out = div(4096, decim)
  # The NCO: GNU Radio rotates by a complex recurrence in f32 (<= 4 * 2^-24 rad per output).
  b_out = a * Enum.sum(Enum.map(t, &abs/1)) * :math.sqrt(2)
  drift = 32768 * b_out * n_out * 4 * :math.pow(2, -24)
  bound = Float.round(bound + drift, 3)

  why =
    why <>
      " Rotation: GNU Radio's rotator recurrence drifts at most #{n_out} * 4 * 2^-24 rad over the case, on an output of at most #{Gen.fmt(b_out)} FS."

  variants =
    for v <- variants do
      lut = 10
      phi = 2 * :math.pi() / :math.pow(2, lut)
      frac = v["format"]["frac"]
      phi = if v["format"]["rounding"] == "nearest", do: phi / 2, else: phi
      extra = 32768 * b_out * (phi + 1.5 / :math.pow(2, frac))

      v
      |> Map.put("max_error_lsb", Float.round(v["max_error_lsb"] + extra, 3))
      |> Map.put(
        "rationale",
        v["rationale"] <>
          " NCO: 10-bit table, phase error up to #{Gen.fmt(phi)} rad plus amplitude rounding, on an output of at most #{Gen.fmt(b_out)} FS: +#{Gen.fmt(extra)} Q15 LSB."
      )
    end

  letters = if kind == :complex, do: "ccf", else: "fcf"

  Gen.case_map(
    name,
    desc,
    4096,
    seed,
    in_spec,
    %{"taps" => t, "decim" => decim, "center_freq" => fc, "sampling_freq" => 1.0},
    %{
      "name" => "freq_xlating_fir_filter_#{letters}",
      "opts" => %{
        "taps" => "taps",
        "decimation" => "decim",
        "center_freq" => "center_freq",
        "sampling_freq" => "sampling_freq"
      }
    },
    [
      %{
        "block" => "filter.freq_xlating_fir_filter_#{letters}",
        "args" => ["decim", "taps", "center_freq", "sampling_freq"]
      }
    ],
    {bound, why, variants},
    if(kind == :real, do: %{"output" => %{"type" => "complex"}}, else: %{})
  )
end

cases =
  cases ++
    [
      xlating.(
        "freq_xlating_fir_ccf",
        "decimation 4, shift down by 0.28125 cycles/sample vs freq_xlating_fir_filter_ccf",
        2011,
        :complex,
        4,
        0.28125,
        Firdes.low_pass(1.0, 1.0, 0.04, 0.03, :hamming)
      ),
      xlating.(
        "freq_xlating_fir_fcf",
        "real input, decimation 2, shift up by 0.125 cycles/sample vs freq_xlating_fir_filter_fcf",
        2020,
        :real,
        2,
        -0.125,
        Firdes.low_pass(1.0, 1.0, 0.1, 0.05, :hamming)
      )
    ]

# Hilbert, DC blocker, single pole, IIR --------------------------------------------------------
h = Firdes.hilbert(65, :hamming)
hc = Enum.with_index(h) |> Enum.map(fn {x, j} -> {if(j == 32, do: 1.0, else: 0.0), x} end)

cases =
  cases ++
    [
      Gen.case_map(
        "hilbert_fc_65",
        "65-tap Hamming Hilbert transformer vs hilbert_fc",
        4096,
        2012,
        real_in,
        %{"ntaps" => 65},
        %{"name" => "hilbert_fc", "opts" => %{"ntaps" => "ntaps"}},
        [%{"block" => "filter.hilbert_fc", "args" => ["ntaps"]}],
        Gen.fir_case(%{peak: a_in, taps: hc, in_type: :real, out_complex: true}),
        %{"output" => %{"type" => "complex"}}
      )
    ]

dcb = fn name, desc, seed, cplx, d, long ->
  taps = ExSdr.Filters.dc_blocker_taps(d, long)
  inp = if cplx, do: dc_in_c, else: dc_in
  a = Gen.peak(inp)

  info =
    Gen.fir_case(%{
      peak: a,
      taps: taps,
      in_type: if(cplx, do: :complex, else: :real),
      out_complex: cplx,
      tap_frac_extra: 8
    })

  Gen.case_map(
    name,
    desc,
    4096,
    seed,
    inp,
    %{"length" => d, "long_form" => long},
    %{
      "name" => if(cplx, do: "dc_blocker_cc", else: "dc_blocker_ff"),
      "opts" => %{"length" => "length", "long_form" => "long_form"}
    },
    [
      %{
        "block" => if(cplx, do: "filter.dc_blocker_cc", else: "filter.dc_blocker_ff"),
        "args" => ["length", "long_form"]
      }
    ],
    info
  )
end

cases =
  cases ++
    [
      dcb.(
        "dc_blocker_cc_long_32",
        "DC blocker, long form, length 32, on complex samples with a step vs dc_blocker_cc",
        2013,
        true,
        32,
        true
      ),
      dcb.(
        "dc_blocker_cc_short_16",
        "DC blocker, short form, length 16, on complex samples with a step vs dc_blocker_cc",
        2014,
        true,
        16,
        false
      ),
      dcb.(
        "dc_blocker_ff_long_8",
        "DC blocker, long form, length 8, on real samples with a step vs dc_blocker_ff",
        2015,
        false,
        8,
        true
      )
    ]

single = fn name, desc, seed, cplx, alpha ->
  inp = if cplx, do: dc_in_c, else: dc_in
  a = Gen.peak(inp)
  letters = if cplx, do: "cc", else: "ff"
  # y = alpha x + (1 - alpha) y: ff [alpha], fb [1 - alpha]
  Gen.case_map(
    name,
    desc,
    4096,
    seed,
    inp,
    %{"alpha" => alpha},
    %{"name" => "single_pole_iir_filter_#{letters}", "opts" => %{"alpha" => "alpha"}},
    [%{"block" => "filter.single_pole_iir_filter_#{letters}", "args" => ["alpha"]}],
    Gen.iir_case(a, [alpha], [1.0 - alpha], cplx)
  )
end

cases =
  cases ++
    [
      single.(
        "single_pole_ff_a0p02",
        "single pole IIR alpha = 0.02 on real samples vs single_pole_iir_filter_ff",
        2016,
        false,
        0.02
      ),
      single.(
        "single_pole_cc_a0p2",
        "single pole IIR alpha = 0.2 on complex samples vs single_pole_iir_filter_cc",
        2017,
        true,
        0.2
      )
    ]

{ff, fb_new} = Gen.butter2(0.1)
[_ | fb_tail] = fb_new
fb_added = Enum.map(fb_tail, &(-&1))

iir = fn name, desc, seed, oldstyle ->
  fbtaps = if oldstyle, do: [1.0 | fb_added], else: fb_new

  Gen.case_map(
    name,
    desc,
    4096,
    seed,
    real_in,
    %{"fftaps" => ff, "fbtaps" => fbtaps, "oldstyle" => oldstyle},
    %{
      "name" => "iir_filter_ffd",
      "opts" => %{"fftaps" => "fftaps", "fbtaps" => "fbtaps", "oldstyle" => "oldstyle"}
    },
    [%{"block" => "filter.iir_filter_ffd", "args" => ["fftaps", "fbtaps", "oldstyle"]}],
    Gen.iir_case(a_in, ff, fb_added, false)
  )
end

cases =
  cases ++
    [
      iir.(
        "iir_ffd_butter2_new",
        "2nd order Butterworth low-pass (fc = 0.1), new style feedback taps, vs iir_filter_ffd",
        2018,
        false
      ),
      iir.(
        "iir_ffd_butter2_old",
        "2nd order Butterworth low-pass (fc = 0.1), old style feedback taps, vs iir_filter_ffd",
        2019,
        true
      )
    ]

# -- measure mode: SNR minima from the committed GNU Radio output --------------------------------------
dir = "test/dsp_cases/filters"

cases =
  if System.argv() == ["measure"] do
    for c <- cases do
      path = Path.join(dir, c["name"] <> ".json")
      stale = path |> File.read!() |> JSON.decode!()
      data = ExSdr.Test.GnuRadioFixture.load(stale, path, "test/fixtures/gnuradio/filters")
      m = ExSdr.Filters.TestKit.measure(c, data)
      snr = fn v -> Float.floor(v - 1.0, 1) end

      c
      |> put_in(["float_vs_gnuradio", "snr_db_min"], snr.(m.float_vs_gnuradio))
      |> update_in(["fixed_vs_float"], fn vs ->
        for v <- vs, do: Map.put(v, "snr_db_min", snr.(m.fixed[v["variant"]]))
      end)
    end
  else
    cases
  end

File.mkdir_p!(dir)

for c <- cases do
  tmp = Path.join(System.tmp_dir!(), c["name"] <> ".json")
  File.write!(tmp, JSON.encode!(c))
  {pretty, 0} = System.cmd("python3", ["-m", "json.tool", "--indent", "2", tmp])
  File.write!(Path.join(dir, c["name"] <> ".json"), pretty)
end

IO.puts("#{length(cases)} cases written to #{dir}")
