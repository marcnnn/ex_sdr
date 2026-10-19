# gr-analog

Modulation, demodulation, phase locked loops, gain control and squelch, as
`ExSdr.Analog.block(name, opts)` returning an `%NxFixed.Block{}`. Code in
`lib/ex_sdr/analog.ex` (blocks), `lib/ex_sdr/analog/kernels.ex` (stream
kernels) and `lib/ex_sdr/analog/math.ex` (angle words, CORDIC, square root,
constants). Cases in `test/dsp_cases/analog/`, GNU Radio output in
`test/fixtures/gnuradio/analog/`, tests in `test/analog/`.

## Coverage

| GNU Radio block | ExSdr | Status | Hardware |
|---|---|---|---|
| `frequency_modulator_fc` | `:frequency_modulator_fc` | done | feedback (phase accumulator) |
| `phase_modulator_fc` | `:phase_modulator_fc` | done | stateless |
| `quadrature_demod_cf` | `:quadrature_demod_cf` | done (CORDIC atan2 in fixed point) | feed forward |
| `complex_to_mag` + `add_const_ff(-1)` (front of `am_demod_cf`) | `:am_demod` | done | stateless |
| `am_demod_cf`, `demod_10k0a3e_cf` | `:am_demod` plus a decimating FIR | partial: the filter belongs to the filters family; tested as a composition with taps supplied by the test | |
| `fm_deemph` | `:fm_deemph` | done | feedback |
| `fm_preemph` | `:fm_preemph` | done | feedback |
| `agc_cc`, `agc_ff` | `:agc_cc`, `:agc_ff` | done | feedback |
| `agc2_cc`, `agc2_ff` | `:agc2_cc`, `:agc2_ff` | done | feedback |
| `agc3_cc` | `:agc3_cc` | done, start gain is an option (see below) | feedback |
| `feedforward_agc_cc` | `:feedforward_agc_cc` | done | feed forward (sliding maximum) |
| `simple_squelch_cc` | `:simple_squelch_cc` | done | feedback |
| `pwr_squelch_cc`, `pwr_squelch_ff` | `:pwr_squelch_cc`, `:pwr_squelch_ff` | partial: `ramp = 0`, `gate = false` only | feedback |
| `pll_carriertracking_cc` | `:pll_carriertracking_cc` | done | feedback |
| `pll_freqdet_cf` | `:pll_freqdet_cf` | done | feedback |
| `pll_refout_cc` | `:pll_refout_cc` | done | feedback |
| `rail_ff` | `:rail_ff` | done | stateless |
| `blocks.sample_and_hold_ff` | `:sample_and_hold_ff` | done, checked against hand computed values only (the GNU Radio harness has one source) | feedback (one register) |
| `nbfm_tx` | `:nbfm_tx` | partial: `audio_rate == quad_rate`; with the interpolating filter tested as a composition (taps supplied by the test) | feedback |
| `nbfm_rx` | `:nbfm_rx_core` plus a decimating FIR | partial, as `nbfm_tx`: demodulator and de-emphasis are the block, the audio filter is a composition | feedback |
| `wfm_rcv` | `:quadrature_demod_cf`, FIR, `:fm_deemph` | composition, tested; no fused block | |
| `wfm_tx`, `wfm_rcv_pll`, `wfm_rcv_fmdet`, `fm_demod_cf`, `demod_200kf3e_cf` | | pending the filters family (interpolating and decimating FIRs, optfir taps); the analog parts exist | |
| `standard_squelch` | | pending: needs a filter | |
| `cpfsk_bc` | | skipped: byte input, interpolating Gaussian-like pulse shaping; belongs with the modulation/filter work | |
| `ctcss_squelch_ff` | | skipped: needs a Goertzel tone detector | |
| `dpll_bb` | | skipped: bit level clock recovery, not in the priority list | |
| `sig_source_*`, `noise_source_*`, `fastnoise_source_*`, `random_uniform_source_*` | | skipped: sources without input; the NCO is `NxFixed.Num.oscillator/3` | |
| `probe_avg_mag_sqrd_*`, `squelch_base_*` | | skipped: probes and a base class | |

## Using a block

    fmt = NxFixed.Num.q(16, 15, rounding: :nearest, lut_bits: 12)
    b = ExSdr.Analog.block(:agc_cc, format: fmt, rate: 2.0e-3, reference: 0.3)
    {y, state} = NxFixed.Block.run(b, {i, q})          # i, q: Fixed s16 tensors
    {y2, state} = NxFixed.Block.run(b, {i2, q2}, state) # carry the state

`:format` is `NxFixed.Num.float()` (default) or `NxFixed.Num.q/3`. The kernels
are the same text in both instances. Constants (gains, loop coefficients,
thresholds) are options, so they are compile time constants, like FIR taps.

| Block | Options (defaults) |
|---|---|
| `frequency_modulator_fc`, `phase_modulator_fc` | `sensitivity` (radians per unit input) |
| `quadrature_demod_cf` | `gain` (1.0), `iterations` (16), `guard` (8) |
| `am_demod` | `offset` (1.0), `guard` (4) |
| `fm_deemph` | `fs`, `tau` (75e-6), `guard` (8) |
| `fm_preemph` | `fs`, `tau` (75e-6), `fh` (-1.0 means 0.925 fs / 2), `guard` (8) |
| `agc_cc`, `agc_ff` | `rate` (1e-4), `reference` (1.0), `gain` (1.0), `max_gain` (65536) |
| `agc2_cc`, `agc2_ff` | `attack_rate` (0.1), `decay_rate` (0.01), `reference`, `gain`, `max_gain` |
| `agc3_cc` | as `agc2`, plus `iir_update_decim` (1); `gain` is the start gain |
| `feedforward_agc_cc` | `n`, `reference` (1.0) |
| `simple_squelch_cc` | `threshold_db`, `alpha` (0.001) |
| `pwr_squelch_cc`, `pwr_squelch_ff` | `threshold_db`, `alpha` (0.0001) |
| `rail_ff` | `lo`, `hi` (representable in the format) |
| `sample_and_hold_ff` | none; the sample is `{data, control}` |
| `pll_*` | `loop_bw`, `max_freq` (pi), `min_freq` (-pi), `iterations`, `guard`; `pll_freqdet_cf` also `out_gain` (1.0) |
| `nbfm_tx` | `quad_rate`, `tau`, `max_dev` (5e3), `fh` (-1.0) |
| `nbfm_rx_core` | `quad_rate`, `tau`, `max_dev` |

Helpers: `ExSdr.Analog.deemph_taps/2` and `preemph_taps/3` (the taps
`fm_emph.py` computes, same formulas) and `agc3_initial_gain/3`.

## How the fixed-point instances work

* **Angle words.** Phases and frequencies are signed 32 bit words, a full turn
  being 2^32 (pi is 2^31), in both instances. A phase adds and wraps exactly,
  so GNU Radio's `phase_wrap` (floating point, to +-2 pi) and ours agree modulo
  one turn. Loop gains multiply the word by a constant with about 30
  significant bits (`Math.mul_real/2`).
* **Sine and cosine** (modulators, PLL): float is exact (f64 rounded to f32),
  fixed point is the table of `NxFixed.Num.oscillator/3`, addressed by the top
  bits of the word: `lut_bits` of the format. The phase error is at most
  pi / 2^lut_bits rad with rounding (twice that when truncating).
* **atan2** (demodulator, PLL phase detectors): float is `Nx.atan2`, fixed point
  is a CORDIC in vectoring mode (`Math.cordic_atan2/4`): reflection into the
  right half plane, then `iterations` steps of add, subtract, shift, compare
  and select with a constant angle table. The residual angle is below
  atan(2^-(iterations - 1)): 3.05e-5 rad for 16 iterations (about 2 LSB at Q15
  for a gain of 2), 1.9e-6 rad for 20. Vectors are shifted left by `guard`
  bits first, so a product of two small samples keeps its angle; a vector of
  only a few raw units (a product of two samples near 1 LSB) is still limited
  by its own integer coordinates, about `1 / (256 |v|)` rad. The zero vector gives 0 as in GNU Radio.
* **Magnitude** (AM detector, AGCs): exact integer square root (floor, digit by
  digit: compare, subtract, shift, select) with `guard` more fractional bits.
* **Constants** are `Fixed` values with their own binary point (about 24
  significant bits, `Math.wconst/2`), so a loop gain of 1e-4 or a demodulator
  gain of 7.96 is not forced into the sample format. Products are exact; the
  kernel rounds once where it stores a value.
* **Accumulators** (AGC gain, squelch power average, de-emphasis output) are
  wider than the samples: the gain has up to 28 fractional bits and room for
  `max_gain`, the power average has 8 more fractional bits than the squared
  samples, the IIR output state 8 more than the format. This avoids the dead
  band a slow loop would have at the sample resolution.
* **Width check.** Every fixed-point instance used in the tests passes
  `NxFixed`'s check. State values and inputs are narrowed with an explicit
  saturate (`bound/2` in the kernels) at the start of the step: by
  construction a no-op in hardware, but the check otherwise assumes the whole
  container type (a Q17 sample sits in an s32, a gain in an s64). Constants
  beyond 32 bits are explicit s64 tensors (the check types bare integer
  constants as s32).

## Accuracy

Every case file states its bound and reasoning (`"rationale"`). Summary, Q15
with a 12 bit table and Q17 with a 14 bit table (errors in Q15 LSB):

| Block | float vs GNU Radio | Q15 vs float | Q17 vs float | What limits it |
|---|---|---|---|---|
| frequency_modulator_fc, phase_modulator_fc | 1.4 LSB (fxpt table in GNU Radio), 0.002 LSB | 25 LSB, 67 dB | 6.4 LSB, 79 dB | sine table |
| quadrature_demod_cf | 0.07 LSB | 2.5 LSB, 78 dB | 2.1 LSB, 78 dB | CORDIC residual, 16 iterations |
| am_demod | 98 LSB (volk NEON `complex_to_mag` on aarch64) | 0.5 LSB, 95 dB | 0.1 LSB, 107 dB | rounding |
| fm_deemph, fm_preemph | 0.003, 0.013 LSB (GNU Radio uses double) | 0.5 LSB, 92 dB | 0.1 LSB, 104 dB | rounding |
| agc_cc/ff, agc2_cc/ff, agc3_cc | 0.0 (same recurrence) | 0.5-0.7 LSB, 87-90 dB | 0.1-0.2 LSB, 99-102 dB | rounding times gain |
| feedforward_agc_cc | 0.0 | 1.5 LSB, 87 dB | 0.4 LSB, 99 dB | rounding times gain |
| squelches, rail_ff | identical, 0.001 LSB | identical, 0.2 LSB | identical, 0.0 LSB | |
| pll_carriertracking_cc | 0.03 LSB | 17 LSB, 67 dB | 5 LSB, 79 dB | sine table |
| pll_freqdet_cf | 0.004 LSB | 1.3 LSB, 74 dB | 0.3 LSB, 86 dB | table noise through the loop |
| pll_refout_cc | 0.06 LSB | 38 LSB, 66 dB | 10 LSB, 79 dB | sine table |
| nbfm_tx | 1.2 LSB | 53 LSB, 64 dB | 16 LSB, 74 dB | table, plus the modulator integrating audio rounding |

The FM modulator is an integrator, so its phase accumulates the rounding of
whatever feeds it; 18 bit audio and a larger table are the two knobs.

## Loops and their transients

PLL and AGC cases compare the float instance with GNU Radio from the first
sample: the recurrences are the same, so the transient is reproduced, not
skipped (GNU Radio agrees with the float instance to 0.06 LSB or better over
all samples). The fixed-point variants of the PLLs are compared with the float
instance from sample 200 on (`"skip": 200`). The measured lock time of the
test loop (`loop_bw = 0.05`, carrier at 0.063 rad per sample) is 43 samples to
5 % and 51 samples to 1 % of the frequency; the fixed-point error against the
float instance is the same with or without the first 200 samples (74.3 dB
for `pll_freqdet_cf` either way), because both start from phase 0 and follow
the same trajectory up to the table noise. AGC cases step the input
amplitude at sample 2500; the gain settles with a time constant of
`1 / (rate * |x|)` samples (about 1700 samples for rate 0.002 at `|x|` = 0.3), and the fixed
variants track the float instance within 1 LSB throughout, transient included.
The cases keep the transient below full scale (a gain overshoot beyond 1.0
saturates in fixed point and is not saturated in the float instance).

## Deviations from GNU Radio

* **Phase is an integer word** (32 bits); GNU Radio's float phase differs by
  rounding, below 1e-6 rad in the tests.
* **`quadrature_demod_cf`** has no history before the first sample, so the
  first output is 0 like GNU Radio's (zero history, `atan2(0, 0) = 0`).
* **AGC gain** is limited to `0..max_gain`; GNU Radio only limits it above. The
  error of the loop uses the unsaturated output up to 16 times full scale
  (like GNU Radio's float), the output itself saturates to the format.
* **`agc3_cc`**: GNU Radio starts from `reference / mean(|x|)` over the first
  `4 * iir_update_decim` samples (it also processes only multiples of that block
  size, dropping the rest of a finite stream) and ignores its `gain`
  argument at the start. A streaming kernel cannot look ahead, so the start
  gain is the `:gain` option: use `ExSdr.Analog.agc3_initial_gain/3` on the first
  block for an exact match. The rule found by experiment (the source is not
  in the container): from sample `4 * decim` on, every `decim` samples
  `g = g (1 - rate) + rate * reference / |x[n]|` with `rate` the attack rate
  when `g |x[n]| > reference`, else the decay rate, and `out = x * g` with the
  updated gain.
* **`agc2`**: GNU Radio picks the attack rate when `e > gain` (with
  `e = |out| - reference`), an apparent typo in its source. Kept, because the
  output must match.
* **Squelch**: `ramp` must be 0 and `gate` false. The power average is of
  `|x|^2` for all three blocks (also `simple_squelch_cc`), the threshold is
  `10^(dB/10)`, and the comparison uses the average after the current sample.
* **`pll_freqdet_cf`** outputs the frequency before the loop update of the
  current sample (found by experiment) in radians per sample times `out_gain`;
  the format must be able to hold it (|value| < 1 for Q15).
* **`feedforward_agc_cc`** is delayed by `n - 1` samples with zeros before that
  (GNU Radio's history of zeros), and its envelope is `max(|re|, |im|) + 0.4
  min(|re|, |im|)`, not `|x|` (found by experiment).
* **Fixed-point tap constants** of the emphasis filters are rounded to 24
  significant bits.
* **GNU Radio on aarch64** computes `complex_to_mag` with a volk NEON kernel
  that is only accurate to about 3e-3. The committed fixtures were made on
  aarch64, so the `am_demod` and `am_demod_cf` bounds include that; on x86 the
  reference is exact and the same bound holds.
* `warmup` of recursive blocks is `:infinity`: the output depends on the
  initial state for ever (decaying). Stateless blocks use a dummy scalar state
  (`NxFixed.Block.run/4` cannot carry `nil` through its scan loop).

## Compositions

`am_demod_cf`, `nbfm_rx` and `wfm_rcv` contain filters. The cases give the
GNU Radio hier block as reference and compose ours from blocks and a real FIR
with the taps GNU Radio computed (firdes and optfir, pasted into the case's
`params`): `:am_demod` then a decimating FIR; `:nbfm_rx_core` then a
decimating FIR; `:quadrature_demod_cf`, decimating FIR, `:fm_deemph`. A
decimating FIR output is the full rate output at every `decim`-th index
starting with the first. `nbfm_tx` with `audio_rate < quad_rate` is the
zero-stuffed input through a FIR with the taps of `interp_fir_filter_fff`,
then `:nbfm_tx`. The FIR helper in `test/analog/helper.exs` stands in for the
filters family; once that exists, the helper's stage is replaced by its block.

## Checking

    mix test test/analog
    scripts/gnuradio_golden.sh test/fixtures/gnuradio/analog test/dsp_cases/analog

Test cost is dominated by compiling each kernel in each format (the PLL
kernels take seconds), not by running it; there are no timing assertions.
