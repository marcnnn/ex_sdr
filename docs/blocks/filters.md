# gr-filter

Module `ExSdr.Filters` (`block/2`), kernels in `ExSdr.Filters.Kernels`, design
in `ExSdr.Filters.Window` and `ExSdr.Filters.Firdes`.

## Coverage

| GNU Radio | ExSdr | Status | Hardware |
|---|---|---|---|
| `fft.window.*` (hamming, hann, blackman, blackman_harris, rectangular, kaiser) | `Window.build/3` | done, equal to f32 taps | - |
| `firdes.low_pass`, `high_pass`, `band_pass`, `band_reject` | `Firdes.*` | done | - |
| `firdes.complex_band_pass` | `Firdes.complex_band_pass/7` | done | - |
| `firdes.root_raised_cosine`, `gaussian`, `hilbert` | `Firdes.*` | done | - |
| `fir_filter_fff/ccf/ccc/fcc` | `:fir_filter_*` | done, with decimation | feed_forward |
| `fft_filter_fff/ccf/ccc` | `:fft_filter_*` | done as direct convolution (same output) | feed_forward |
| `interp_fir_filter_fff/ccf/ccc/fcc` | `:interp_fir_filter_*` | done | feed_forward |
| `rational_resampler_fff/ccf/ccc/fcc` | `:rational_resampler_*` | done | feed_forward |
| `freq_xlating_fir_filter_ccf/ccc/fcf/fcc` | `:freq_xlating_fir_filter_*` | done, `fcc`/`ccc` not checked against GNU Radio | feed_forward (NCO phase is state) |
| `hilbert_fc` | `:hilbert_fc` | done | feed_forward |
| `dc_blocker_ff/cc` | `:dc_blocker_*` | done, long and short form | feed_forward (see below) |
| `single_pole_iir_filter_ff/cc` | `:single_pole_iir_filter_*` | done | feedback |
| `iir_filter_ffd` | `:iir_filter_ffd` | done, old and new style | feedback |
| `pfb_interpolator_ccf` | `:pfb_interpolator_ccf` | done (the interpolating FIR) | feed_forward |
| `pfb_decimator_ccf` | `:pfb_decimator_ccf` | done, serial input instead of one input per phase | feed_forward |
| `pfb_arb_resampler_*` | - | skipped | - |
| `mmse_resampler_cc/ff` | - | skipped | - |

Skipped blocks produce a data dependent number of outputs per input (the
output position advances by a real-valued ratio). `NxFixed.Block` has
fixed `{in, out}` rates only, and GNU Radio keeps the position in `double`;
a fixed-point version needs an integer phase accumulator and a new
variable-rate block kind, which is a change to nx_fixed.

## Block kinds

Blocks with rate 1:1 are `:stream` (one item per call, checked by
`NxFixed.run_scan_stacked/5` with the width check). Decimating,
interpolating and resampling blocks and the frequency translating filter
are `:frame` with `rate: {in, out}`; the frame length is a multiple of `in`.
All state is explicit: the delay line (and the NCO phase word for the
translating filter). Input in pieces gives the same output as one call
(`test/filters/blocks_test.exs`).

`latency` is the group delay in output items for linear phase taps, else
the centre of the tap energy. `warmup` is the delay line length for FIRs
and, for IIRs, the input count after which the impulse response stays below
2^-17.

## Deviations from GNU Radio

* Windows of one point: GNU Radio gives NaN for cosine windows, we give 1.
* `Firdes` computes in double; GNU Radio returns f32. Differences are below
  3e-7 (complex band pass 5e-6: GNU Radio advances its rotation phase in f32).
* `rational_resampler`: GNU Radio's first output is the full-rate filter
  output at position `(K-1)*I` (K taps per polyphase branch), not at 0. The
  block takes `:phase` for this; with `phase = rem((K-1)*I, D)` our output
  `m + div((K-1)*I, D)` equals GNU Radio output `m` (the case files carry
  `reference_shift`).
* `freq_xlating_fir_filter`: taps are turned by `fc/fs * (i - (n-1)/2)`
  turns, the first output is not rotated, and the rotation advances by
  `-fc*D/fs` per output. The NCO is a u16 phase word (`NxFixed.Num.oscillator`
  style): the shift per output is rounded to 1/65536 turn, so centre
  frequencies with `fc*D/fs` not a multiple of 2^-16 differ slowly from GNU Radio
  (which rotates with an f32 complex recurrence). The cases use exact values.
* `fft_filter_*`: direct convolution, no FFT. GNU Radio drops the last
  partial FFT block of a finite stream (8 items in the case).
* `dc_blocker`: GNU Radio implements it with running sums; the output is a
  FIR (the input delayed by the group delay minus two or four cascaded
  moving averages), which is what we compute, with taps rounded to 8 more
  fractional bits than the data so that the DC gain stays zero to within
  a fraction of an LSB. Declared `feed_forward`: finite impulse response
  and finite warmup. A hardware version would use running sums with wide
  accumulators; the width check cannot prove their bound.
* `iir_filter_ffd`: GNU Radio keeps `double` state; the float instance is f32. Taps of
  `fbtaps[0]` is ignored as in GNU Radio.
* Fixed-point filters round the output to the format after the exact
  accumulation; the feedback of an IIR sees the rounded output. A rounded
  single pole stops within `0.5 / alpha` LSB of a constant input.
* `pfb_decimator_ccf` takes the serial stream (what `stream_to_streams`
  would split) and gives one output per `decimation` items; it is
  one FIR with taps `taps[i] * e^(j 2 pi channel (i mod D) / D)` sampled at
  `k*D + D - 1`.

## Tests

* `test/filters/design_test.exs`: 136 window and firdes parameter sets against GNU Radio taps.
* `test/filters/gnuradio_test.exs`: per case float vs GNU Radio, three fixed formats vs float,
  and the width check. Cases are written by `test/filters/gen_cases.exs`
  (bounds derived from tap rounding, see the file header for the two-step run).
* `test/filters/pfb_test.exs`: polyphase blocks against `test/gnuradio/filters_pfb.py` output.
* `test/filters/blocks_test.exs`: split invariance, declarations, behaviours.
