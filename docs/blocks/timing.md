# Timing: symbol timing recovery and arbitrary resampling

Module `ExSdr.Timing` (`block/2`), kernels in `ExSdr.Timing.Kernels`,
interpolators in `ExSdr.Timing.Interp` (GNU Radio's tables in
`ExSdr.Timing.Tables`), loop arithmetic in `ExSdr.Timing.Loop`.

## Coverage

| GNU Radio | ExSdr | Status | Hardware |
|---|---|---|---|
| `filter.mmse_fir_interpolator_cc/ff` (kernel) | `Interp.mmse/0`, `:fractional_delay_cc/ff` | done, GNU Radio's 129 x 8 table | feed_forward |
| `filter.mmse_interp_differentiator_cc/ff` (kernel) | `Interp.mmse/0` (`drows`) | done, used by symbol_sync | - |
| polyphase fractional delay | `:fractional_delay_*` with `interpolator: {:pfb_mf, taps, n}`, `{:pfb_no_mf, n}`, `{:polyphase, taps, n}` | done | feed_forward |
| `filter.mmse_resampler_cc/ff` | `:mmse_resampler_cc/ff` | done; the second input (ratio per item) and the message port are not modelled | feedback |
| `filter.pfb_arb_resampler_ccf/fff` | `:pfb_arb_resampler_ccf/fff` | done (rate up to `filter_size`) | feedback |
| `filter.pfb_arb_resampler_ccc` | - | skipped: complex taps (the interpolator kernel takes real tables) | - |
| `digital.clock_recovery_mm_cc/ff` | `:clock_recovery_mm_cc/ff` | done; the error output is not provided | feedback |
| `digital.symbol_sync_cc/ff` | `:symbol_sync_cc/ff` | done: all 9 TEDs (Mueller-Muller, modified M&M, zero crossing, Gardner, early-late, D'Andrea-Mengali generalized MSK, Mengali-D'Andrea GMSK, signal times slope ML, signum times slope ML), interpolators MMSE, PFB_NO_MF, PFB_MF; partial: no `time_est`/`clock_est` tags, no diagnostic outputs | feedback |
| `digital.pfb_clock_sync_ccf/fff` | `:pfb_clock_sync_ccf/fff` | done; no `time_est` tags, no diagnostic outputs | feedback |
| `digital.msk_timing_recovery_cc` | `:msk_timing_recovery_cc` | done (osps 1 and 2); no `time_est` tags, no error/mu outputs | feedback |

All except `:fractional_delay_*` have `rate: {:variable, max}`: each
input item gives 0..`max` outputs, `NxFixed.Block.run/4` returns the valid
ones in order (`nil` when there are none).

## How the variable rate works

GNU Radio's blocks walk through a buffer with an input index and a
fractional position that advance by a real-valued step per output (or per
interpolation, for symbol_sync). Here the walk is driven by arriving items
(`ExSdr.Timing.Kernels.machine/4`): the state holds the last `L` items and
`wait`, the number of items still missing for the next iteration's window.
Each kernel call pushes one item, decrements `wait`, and runs up to
`:iterations` iterations while `wait <= 0`; an iteration interpolates
from the line, updates the loop, may emit, and adds the integer part of
its step to `wait`. The iterations are unrolled (`max` output lanes with
valid bits in hardware). `:iterations` defaults to the most iterations one
input item can need with the block's parameters (e.g. `floor(1 / ratio) +
1` for mmse_resampler, `ceil(2 * ips / (sps - max_deviation))` for
symbol_sync).

GNU Radio's `set_history(h)` puts `h - 1` zero items in front of the
input; the zero line plays that part and the initial `wait` puts the first
iteration exactly where GNU Radio's is (clock_recovery_mm_cc: two zeros,
pfb_arb_resampler: `taps per arm - 1`, pfb_clock_sync: `taps per arm + 2 sps - 1`).

The state is explicit (line, `wait`, loop state), so input split anywhere
gives the same output as one call, bit for bit, in both instances.

## Number formats

Float instance: the loop arithmetic is GNU Radio's, step by step: f32
where GNU Radio uses `float`, f64 for mmse_resampler's `double` position,
GNU Radio's `branchless_clip` with its roundings, `rint` (ties to even) for
the table row, the loop gains computed with GNU Radio's f32 roundings
(`ExSdr.Timing.clock_loop_gains/3`, pfb_clock_sync's damping of
`2 * nfilts`). Only the interpolator sums differ (GNU Radio uses SIMD dot
products, the order of the f32 additions differs).

Fixed point (`NxFixed.Num.q/3`):

* Positions, fractional delays, periods and loop filter state are s64
  words with 32 fractional bits (`ExSdr.Timing.Loop`, +-2^31 samples).
  Adding and splitting into integer and fraction is exact, so a position
  never drifts. Ratios that are f32 values of at least 2^-8 are exact on
  this grid, so mmse_resampler walks exactly GNU Radio's positions.
* Loop gains (`gain_mu`, `alpha`, `beta`, ...) are constants with 22
  significant bits; each gain times error is rounded once to the word.
* Table row: `round(mu * arms)` with ties up (GNU Radio: ties to even).
* Taps are rounded to `:tap_frac` fractional bits (default the format's),
  not saturated; the window sum is exact and rounded once to the format.
* Products that can exceed full scale (symbol_sync derivatives, the
  GMSK detectors' squares, pfb_clock_sync's derivative arm, msk's
  nonlinearity) use the format with 4 more integer bits; slicer decisions
  (QPSK points are +-sqrt(2)) 3 more.
* pfb_arb_resampler multiplies the derivative arm (rounded to the format)
  by `acc` rounded to `:mu_bits` (18) bits.

Every fixed-point instance passes `NxFixed`'s width check (tested for
Q15, Q17 and Q11 with truncation), so it is bit-exact on every backend.
State values are restated within their ranges at the start of each call
(`bound_state`), which is free in hardware.

## Deviations from GNU Radio

* GNU Radio stops a few outputs before the end of a finite stream (it
  waits for input beyond its windows: up to 6 for the resamplers and
  symbol_sync, 41 for pfb_clock_sync with 45 taps per arm); our output is
  a longer sequence with GNU Radio's as prefix.
* symbol_sync clamps the instantaneous period to
  `[ips / iterations, 2 * (sps + max_deviation)]` (GNU Radio only replaces
  a period `<= 0` by the average). This bounds the work per input item
  and the look-ahead window; it is inactive near lock.
* clock_recovery_mm and msk_timing_recovery: the block raises if the
  gains allow a position step below 0 (GNU Radio clamps the input index
  at 0); pass `:iterations` to accept it.
* pfb_clock_sync: the wraps of the filter phase move a symbol's window;
  the line keeps `:margin` (2) items on each side and windows further out
  are clipped to the line (GNU Radio's buffer is unbounded). Never active
  in the tests.
* Slicers (symbol_sync's M&M, modified M&M, zero crossing): sign
  decisions for BPSK and QPSK shaped constellations, as GNU Radio's
  sector decision makers; nearest point (first point on ties) for other
  constellations, where GNU Radio's decision makers agree except on
  boundaries.
* Kept from GNU Radio on purpose: clock_recovery_mm_cc's 0/1 slicer;
  msk_timing_recovery uses the current sample as "delayed" sample
  (`d_dly_conj_2 = d_dly_conj_1 = in_interp`); pfb_clock_sync adds
  `rate_f` once per output and again `sps` times in the loop update.
* Tags (`time_est`, `clock_est`), message ports and diagnostic outputs
  are not modelled.

## Checks against GNU Radio

Resamplers (`test/dsp_cases/timing/*.json`, run by the shared
`test/gnuradio/golden.py`): tones and noise in Q15, 2048 items.

| Case | max error float vs GNU Radio | Q15 vs float |
|---|---|---|
| mmse_resampler cc 0.75, cc 1.375, ff 2.5, ff 0.3 | 1.2e-7 (ff: 0) | 1.0-1.2 LSB |
| pfb_arb_resampler ccf 1.5059, ccf 0.8, fff 2.56 (32 arms) | 1.8e-7 | 1.6-2.0 LSB |

Symbol timing recovery (`test/dsp_cases/timing/sync/*.json`, run by
`test/gnuradio/timing_golden.py`): QPSK and BPSK with raised cosine
(matched filter output) or root raised cosine pulses (roll-off 0.35), and
MSK, at 4 (and 3.3) samples per symbol, a timing offset and a sample rate
offset of +-100 ppm, noise 0.01 rms, about 880 symbols.

* Float vs GNU Radio: median error 0-6e-8 in every case. In 15 of 25
  cases every output agrees within 3e-7. In the others, single outputs
  differ where a fractional delay lies within f32 rounding of a boundary
  between two table rows (the two pick neighbouring rows), with errors up
  to 0.0027 (128 rows) and 0.011 (32 rows) that the loop then evens out:
  0.3% of the outputs for most, 2% for zero crossing and the PFB_MF cases,
  6% for modified M&M, 10% for msk_timing_recovery (its loop is slow).
  MER after lock equal within 0.01 dB.
* Q15 and Q17 vs float, after 200 outputs: max 0.0019-0.01, rms 1e-4 to
  1.5e-3, MER equal within 0.06 dB, the same number of outputs.

The fixtures are small: inputs as int16, GNU Radio outputs as f32
(780 KB for the family).

## Hardware notes

* An iteration is one table lookup per tap (8 ROM columns of 129 entries
  for MMSE) and a MAC over the window, plus the loop update: one
  `:feedback` path through the position word.
* `:iterations` copies of the iteration per input item (unrolled, `max`
  output lanes). For hardware with one iteration per clock, the
  iterations would run in sequence and the input would stall.
* pfb_clock_sync reads windows at data-dependent offsets (a multiplexer
  over `2 * margin + 1` positions).

## Tests

* `test/timing/resampler_test.exs`: resampler cases vs GNU Radio, Q15 and
  Q17 vs float, split invariance.
* `test/timing/sync_test.exs`: synchronizer cases vs GNU Radio, Q15 (and
  Q17 for five cases) vs float after lock, split invariance (Q15 in three
  pieces, float in small pieces).
* `test/timing/blocks_test.exs`: declarations, width check of every block
  in Q15, Q17 and Q11 truncating, the interpolator tables, the fractional
  delay.
* Regenerating: `mix run test/timing/gen_cases.exs` writes the case files;
  `scripts/gnuradio_golden.sh test/fixtures/gnuradio/timing test/dsp_cases/timing`
  and the podman command in `test/gnuradio/timing_golden.py` write the
  fixtures.
