# Carrier recovery, equalisation, correlation and estimators (`ExSdr.Carrier`)

The carrier and frequency synchronisation, equaliser, correlation and
estimator blocks of GNU Radio's gr-digital as `NxFixed.Block`s:
`ExSdr.Carrier.block(name, opts)` builds a block, `ExSdr.Carrier.run(block,
input, state)` runs it and returns `{output, state}`. The state is explicit, so
a stream cut anywhere (also at a cut that is not a multiple of `sps`) gives the
output of the whole, bit-exact for fixed point. Code in
`lib/ex_sdr/carrier.ex` (blocks), `carrier/loop.ex` (the control loop),
`carrier/kernels.ex` (Costas, FLL, constellation receiver),
`carrier/equalizer.ex` and `carrier/estimators.ex`. Cases in
`test/dsp_cases/carrier/`, GNU Radio output in
`test/fixtures/gnuradio/carrier/`, tests in `test/carrier/`.

## Coverage

| GNU Radio block | ExSdr | Status | Hardware |
|---|---|---|---|
| `costas_loop_cc` | `:costas_loop_cc` | done: orders 2, 4, 8 with GNU Radio's detectors, error clip and `control_loop` gains; the frequency, phase and error ports as `outputs:`. Not done: `use_snr` (the tanh detectors), the noise message port, the `phase_est` input tag | feedback |
| `fll_band_edge_cc` | `:fll_band_edge_cc` | done: band-edge filters as `design_filter`, GNU Radio's history delay of `filter_size` items included; frequency, phase and error ports as `outputs:`. Not done: the setters | feedback |
| `constellation_receiver_cb` | `:constellation_receiver_cb` | done: the decision directed phase loop; decisions are the nearest point; error, phase, frequency and derotated symbol as `outputs:`. Not done: the `set_constellation` and `rotate_phase` message ports, dispatching tags | feedback |
| `linear_equalizer` | `:linear_equalizer` | done: LMS, NLMS, CMA (the `adaptive_algorithm_*` objects are the `:algorithm` option), training sequence with a start flag input, decimation by `sps`, `adapt_after_training`, the taps and state ports as `outputs:`. Not done: `set_taps`, reading the start from a tag (it is a flag stream) | feedback |
| `adaptive_algorithm_lms`, `_nlms`, `_cma` | `:algorithm` `:lms`, `:nlms`, `:cma` of the equalizers | done (the decision directed LMS is `:lms` without training) | feedback |
| `cma_equalizer_cc`, `lms_dd_equalizer_cc` | `:cma_equalizer_cc`, `:lms_dd_equalizer_cc` | GNU Radio 3.10 has neither any more (they became `linear_equalizer` with `adaptive_algorithm_cma` and `_lms`); the names here are `:linear_equalizer` with the algorithm preset | feedback |
| `decision_feedback_equalizer` | `:decision_feedback_equalizer` | done, same options plus `num_taps_forward`, `num_taps_feedback` | feedback |
| `corr_est_cc` | `:corr_est_cc` | partial: `THRESHOLD_ABSOLUTE` only; the tags are output streams (below); no `amp_est` | control |
| `mpsk_snr_est_cc` | `:mpsk_snr_est_cc` | done for `simple`, `skew`, `m2m4`, `svr`: the estimator registers per item, `Estimators.snr_db/3` gives the value GNU Radio puts in its tag | feedback |
| `probe_mpsk_snr_est_c` | `:probe_mpsk_snr_est_c` | done: the same registers; `snr_db/3`, `signal_noise_db/3`; GNU Radio's message timing is the caller's | feedback |
| `snr_est_m2m4` (with `ka`, `kw`) | none | skipped: the M2M4 registers with another formula | |
| `probe_density_b` | `:probe_density_b` | done | feedback |
| `probe_signal_f`, `_c`, `_b` | `:probe_signal` (`item:` `:real`, `:complex`, `:byte`) | done: passes the stream on, the state is the level | feed forward |
| `probe_signal_s`, `_i`, `_vector_*` | none | skipped: wider item types | |
| `symbol_sync_cc/ff`, `clock_recovery_mm_cc/ff`, `pfb_clock_sync_*`, `msk_timing_recovery_cc` | none | skipped: timing recovery, a family of its own | |
| `pn_correlator_cc`, `ofdm_equalizer_*`, `ofdm_frame_equalizer_vcvc`, `ofdm_sync_sc_cfb` | none | skipped: not in the list; the OFDM ones belong with the OFDM blocks | |
| `ber_bf`, `correlate_access_code_*` | none | `correlate_access_code_*` is in `ExSdr.Digital`; `ber_bf` skipped | |

## Using a block

    fmt = NxFixed.Num.q(16, 15, rounding: :nearest, lut_bits: 12)
    b = ExSdr.Carrier.block(:costas_loop_cc, format: fmt, loop_bw: 0.04, order: 4,
                            outputs: [:freq])
    {{y, {freq}}, state} = ExSdr.Carrier.run(b, {i, q})   # i, q: Fixed s16 tensors
    {out2, state} = ExSdr.Carrier.run(b, {i2, q2}, state) # carry the state

`:format` is `NxFixed.Num.float()` (default) or `NxFixed.Num.q/3`. Constants
(gains, taps, points) are options, so they are compile time constants.
Phases and frequencies are 32 bit angle words (a full turn is 2^32) in both
instances, as in `ExSdr.Analog`; the rest of the arithmetic is described per
block below.

| Block | Options (defaults) |
|---|---|
| `costas_loop_cc` | `loop_bw`, `order` (4), `max_freq` (1.0), `min_freq` (-1.0), `phase`, `freq` (initial, 0), `outputs` (`[]`, from `:freq`, `:phase`, `:error`) |
| `fll_band_edge_cc` | `samples_per_symbol`, `rolloff`, `filter_size`, `loop_bw` (GNU Radio's `bandwidth`), `max_freq`, `min_freq` (default +-2 pi 2 / sps), `tap_frac` (20), `outputs` |
| `constellation_receiver_cb` | `constellation` (`ExSdr.Digital.Constellation`, a name or a points list), `loop_bw`, `fmin`, `fmax`, `iterations`, `guard`, `outputs` (`:error`, `:phase`, `:freq`, `:symbol`) |
| `linear_equalizer` | `num_taps`, `algorithm`, `mu`, `sps` (1), `constellation`, `modulus` (CMA, 1.0), `training` (list of `{re, im}`; the input is then `{x, flag}`), `adapt_after_training` (true), `initial_taps`, `weight_bits` (32), `weight_frac` (28), `guard` (8), `outputs` (`:taps`, `:state`) |
| `decision_feedback_equalizer` | as above with `num_taps_forward` and `num_taps_feedback` |
| `corr_est_cc` | `symbols` (`{re, im}`), `sps` (1), `mark_delay` (0), `threshold` (0.9), `tap_frac` (20) |
| `mpsk_snr_est_cc`, `probe_mpsk_snr_est_c` | `type` (`:simple`, `:skew`, `:m2m4`, `:svr`), `alpha` (0.001) |
| `probe_density_b` | `alpha` |
| `probe_signal` | `item` |

Helpers: `ExSdr.Carrier.Kernels.fll_taps/3` (the band-edge taps),
`ExSdr.Carrier.Estimators.snr_db/3`, `signal_noise_db/3`.

## Loops

All three loops share `ExSdr.Carrier.Loop`, GNU Radio's second order
`control_loop` on angle words:

    freq  += beta * error
    phase += freq + alpha * error     (the new frequency)
    freq   = clamp(freq, min_freq, max_freq)

with `alpha`, `beta` from the loop bandwidth at damping sqrt(2) / 2
(`ExSdr.Analog.Kernels.loop_gains/1`, reused). The phase wraps exactly as a
word; GNU Radio wraps in float to +-2 pi, the two agree modulo one turn.

* **Costas**: `y = x exp(-j phase)`, one of GNU Radio's detectors on `y`
  (order 2: `re * im`; order 4: `sign(re) im - sign(im) re`; order 8 as
  `costas_loop_cc_impl.h`, which is not the textbook one), clipped to +-1 the
  way GNU Radio does (`0.5 (|e + 1| - |e - 1|)` in the float instance, a saturation
  at 30 fractional bits in fixed point).
* **FLL**: the input is delayed by `filter_size` items (GNU Radio gives the
  block a history of `filter_size + 1` and indexes it from the start, so its
  output lags by that much; the first outputs are zero), rotated by
  `exp(+j phase)`, and filtered by the lower and the upper band-edge filter;
  the error is the difference of their output powers. The filters are
  complex FIRs with constant taps (`ExSdr.Filters.Kernels.apply_taps/7`,
  reused), taps with `tap_frac` fractional bits and exact accumulators;
  outputs are rounded to 24 fractional bits (28 bits) before squaring so the
  squares stay in 64 bits.
* **Constellation receiver**: `y = x exp(+j phase)` (the other sign than the
  Costas loop), decision = nearest point (`ExSdr.Digital.Kernels.decide/3`,
  reused), error = `-arg(y conj(point))` as an angle word from the CORDIC
  arctangent of `ExSdr.Analog.Math` (16 iterations), unclipped as in GNU Radio.

Sine and cosine are the table of the format (`lut_bits`), an error of up to
pi / 2^lut_bits rad, like every NCO of the analog family.

## Equalisers

`ExSdr.Carrier.Equalizer` documents the step in detail. Items worth knowing:

* `w[i]` multiplies `window[i]` (oldest first): GNU Radio's taps in the order
  its FIR kernel stores them. LMS and NLMS start from zero, CMA from the unit tap
  on the newest sample, as `adaptive_algorithm::initialize_taps`.
* Training: a `u8` flag stream next to the samples replaces the stream tag.
  The training sequence applies from the next output instant, then decision
  directed if `adapt_after_training`. The state machine is GNU Radio's (idle,
  training, decision directed, including that without a training sequence
  and with `adapt_after_training: false` the block never adapts).
* `sps > 1`: `rate: {:variable, 1}`, an output every `sps` inputs. The linear
  equalizer outputs at the first sample of each group (its window ends there),
  so a flag counts for the first output instant *at or after* the flagged
  sample; GNU Radio counts the output at or before it (the same when
  the flag is at a multiple of `sps`, as in the cases). The decision feedback
  equalizer outputs at the last sample of each group (GNU Radio's window), where
  a flag anywhere in the group is seen, exactly as GNU Radio.
* **Weight format** (fixed point): `weight_bits` 32, `weight_frac` 28 by
  default: range +-8, step 3.7e-9, 13 bits finer than a Q15 sample. A step
  size `mu` moves a weight by `mu |x| |e|` per output; with `mu = 2e-4` and an
  error of 1e-2 that is 6e-7, 160 LSBs. Updates below about 2e-9
  (`mu |x| |e|`) are lost to rounding; for steps that small raise
  `weight_frac` (with `weight_bits` up to 40: `window * weight` must stay in 63 bits,
  which limits `weight_bits` + sample bits + log2(taps) to 63).
  The filter output has 8 more fractional bits than the format (and 2 more
  integer bits), error times `mu` is a 30 bit fraction, the NLMS step divides by the
  window energy (floor 2^-16, an integer quotient); all exact integer arithmetic.
  Decisions of the DFE are kept in the sample format.
* CMA error: `y (|y|^2 - modulus)` clipped to +-1 per component as
  `adaptive_algorithm_cma::error`; `modulus` is the target `|y|^2`, as in GNU Radio.

## Correlation and estimators

`corr_est_cc` outputs six values per input item: the input delayed by the sync
word length (GNU Radio's primary output), the correlation, and the tag
information of a detection: `detect` (`u8`) marks that the *previous* sample
`p` was a peak (the decision needs `corr[p + 1]`), with `phase` (an angle word),
`time_est` (22 fractional bits) and `corr_est` (the squared magnitude, own binary
point) of that peak. GNU Radio's tags then are `corr_start` at stream index `p`
and `phase_est`, `time_est`, `corr_est` at `p + mark_delay`. A peak is a sample
with `|corr|^2 > threshold (sum |s|^2)^2` not smaller than its successor and
at least `round(sps)` after the previous one. GNU Radio does this per work call
(its loop stops at the end of a buffer); the streaming rule is the same
everywhere else and the cases show equal detections.

The SNR estimators report registers, not decibels, because the logarithm and
the divisions are for the host at message rate. GNU Radio's blocks have a history of one
item, so their estimators see the input delayed by one with a zero first; this is
reproduced, which also matters for `svr` (it pairs each item with the one before).
`simple` reproduces GNU Radio's recurrence including its off-by-one divisor.
`skew` is GNU Radio's recurrence with `i` the index since the start; GNU Radio
uses the index inside the work call, which depends on its buffer sizes, so `skew` is
compared with an independent double precision implementation of the recurrence
instead of GNU Radio.

## Accuracy

Every case file states its bounds and reasoning (`"rationale"`). A bound is a
number per output port. Measured, in the unit of the port (output in full scale = 1,
frequency and phase in rad):

| Block | float vs GNU Radio | Q15 (lut 12) vs float | Q17 (lut 14) vs float | What limits it |
|---|---|---|---|---|
| `costas_loop_cc` (2, 4, 8) | output 1.5e-6, freq 1e-7, phase 2e-6, error 2e-6, from sample 0 | output 8e-4, freq 2e-5, phase 5e-4 | 2e-4, 5e-6, 1e-4 | sine table |
| `fll_band_edge_cc` | output 3e-5, freq 1e-7, phase 4e-5 (f32 filters) | output 4e-3 (sps 4), 1.5e-2 (sps 2); phase 5e-3, 2e-2 | 2e-3, 5e-3 | table, rounding of the filter outputs |
| `constellation_receiver_cb` | decisions equal, phase 2e-6 | decisions equal (QPSK, 8-PSK), phase 1e-3 | equal | table, CORDIC |
| `linear_equalizer`, `decision_feedback_equalizer` | 3e-7 | output 2.1e-5 (0.7 LSB), taps below 2e-6 | 5e-6 | output rounding only |
| `corr_est_cc` | correlation 1e-6, phase 1e-6, detections equal | corr 1e-5, phase 2.5e-5, time_est 3e-7, detections equal | same | taps in 20 bits, CORDIC |
| SNR estimators | 1e-4 dB (6e-4 dB at 20 dB, M2M4 subtracts nearly equal terms) | below 1e-4 dB | same | 40 bit registers |
| `probe_density_b` | 5e-8 | 4e-10 | same | |

The Costas loop, FLL, receiver and equalizers run the same recurrence as GNU
Radio, so the float instance is compared from the first sample, with no
transient skipped (the f32 recurrences agree to a few 1e-6 through the
acquisition, so the transient is reproduced, not hidden). The fixed-point
variants are compared with the float instance from sample 0 too, except the
16-QAM receiver case, compared from symbol 1000: while the phase error is
still above 0.1 rad the decisions near a boundary are ambiguous and 37
decisions between symbols 270 and 604 differ; after 604 none.

Lock is checked on GNU Radio's own output of each loop case (`"lock"`: the mean
frequency in the tail equals the true offset within the stated tolerance) and
convergence of the equalizer cases (`"evm"`), so the fixtures are meaningful and
not just equal garbage.

## Inputs of the cases

`test/gnuradio/carrier_golden.py` builds the input with GNU Radio blocks
(symbols, `chunks_to_symbols_bc`, RRC `interp_fir_filter_ccf`, multipath
`fir_filter_ccc`, `rotator_cc` for the frequency offset), adds noise, and quantises
to Q15 like every case. Its docstring has the case format (constellation expression,
sync words inserted at given positions, input tags, several output ports, stream tags,
probe methods) and the `podman` command, since `scripts/gnuradio_golden.sh`
runs `golden.py` only.

## Deviations from GNU Radio

* Tags and message ports are streams or host functions (above). The training
  start is a flag stream.
* `corr_est_cc`: no dynamic threshold, no `amp_est`, detections one item late.
* The Costas loop has no `use_snr` variants and no noise message.
* `linear_equalizer` with `sps > 1`: the flag semantics noted above.
* The skew estimator's index (above).
* Constellation decisions are the nearest point; GNU Radio's sector tables
  agree except in sectors it maps differently on purpose.
* `cma_equalizer_cc` and `lms_dd_equalizer_cc` do not exist in the GNU Radio
  3.10 the cases come from; they are presets of `linear_equalizer`.

## Changes needed outside this family

None for the code. `scripts/gnuradio_golden.sh` could take the runner as a
third argument (it runs `golden.py` only; the digital and carrier families call
`podman` directly, see the docstrings of `digital_golden.py` and
`carrier_golden.py`).
