# Sources and channel models

Port of the signal sources of gr-analog, the sources of gr-blocks and gr-channels.
Code in `lib/ex_sdr/sources.ex` and `lib/ex_sdr/sources/`, tests in
`test/sources/`, GNU Radio cases in `test/dsp_cases/sources/` with their output in
`test/fixtures/gnuradio/sources/`.

    block = ExSdr.Sources.block(:sig_source_c, frequency: 1000.0, sample_rate: 48_000.0)
    {samples, state} = ExSdr.Sources.generate(block, 4096)
    {more, state} = ExSdr.Sources.generate(block, 4096, state)

Sources have no input. They are `:stream` blocks whose input sample is ignored
(`generate/3` makes the dummy input), so the same scan loop and the same width
check run them. Every block takes `:format` (float by default, or `NxFixed.Num.q/3`).

## Coverage

| GNU Radio | ExSdr | Status | Hardware |
|---|---|---|---|
| `analog.sig_source_f/c` | `block(:sig_source_f / :sig_source_c, waveform:)`, six waveforms | done | feedback |
| `analog.sig_source_b/s/i` | | skipped: integer outputs are what fixed-point formats are for | |
| (`fxpt_nco`) | `block(:nco)` (complex sine) | done | feedback |
| `blocks.rotator_cc` | `block(:rotator_cc)` | done | feedback |
| `blocks.vco_c` | `block(:vco_c)` | done | feedback |
| `blocks.vco_f` | | skipped: the imaginary part of `vco_c` | |
| `analog.noise_source_f/c` | `block(:noise_source_f / _c, type:)`: uniform, gaussian, laplacian, impulse | done, statistics only | feedback |
| `analog.noise_source_s/i` | | skipped (integer noise) | |
| `analog.fastnoise_source_f/c` | `block(:fastnoise_source_f / _c)` | done, statistics only | memory |
| `analog.random_uniform_source_*` | | skipped: not done | |
| `blocks.vector_source_f/c` | `block(:vector_source_f / _c, data:, repeat:)` | done | memory |
| `blocks.vector_source_b/s/i` | | skipped: use a fixed-point format | |
| `blocks.null_source` | `sig_source` with `waveform: :constant, amplitude: 0.0` | done (as a special case) | stateless |
| `digital.glfsr_source_b` | `block(:glfsr_source_b)`, on `ExSdr.Comm.LFSR` | done, bit exact | feedback |
| `blocks.lfsr_32k_source_s` | | skipped: not done | |
| `blocks.file_source`, `wavfile_source`, `file_descriptor_source`, `file_meta_source` | | skipped: I/O belongs to the runtime | |
| `filter.mmse_resampler_cc` (timing offset) | `block(:mmse_resampler_cc, ratio:)` | done, own taps (agree to 1.2e-5) | feedback |
| `filter.fir_filter_ccc/ccf` | `block(:fir_ccc, taps:)` (complex or real taps) | done, as needed by the channels | feed_forward |
| `channels.channel_model` | `block(:channel_model)` | done | feedback |
| `channels.channel_model2` | | skipped: needs the dynamic parts below | |
| `channels.fading_model` | `block(:fading_model)` | done, statistics only | feedback |
| `channels.selective_fading_model` | `block(:selective_fading_model)` | done, statistics only, own fractional delay filters | feedback |
| `channels.selective_fading_model2`, `dynamic_channel_model` | | skipped: need `sro_model` | |
| `channels.cfo_model` | `block(:cfo_model)` | done, statistics only | feedback |
| `channels.sro_model` | | skipped: a resampler whose ratio changes every output sample is not a fixed-step scan | |
| `channels.iqbal_gen` | `block(:iqbal_gen)` | done, equals GNU Radio | stateless |
| `channels.distortion_2_gen`, `distortion_3_gen` | `block(:distortion_2_gen / _3_gen)` | done, equals GNU Radio | stateless |
| `channels.quantizer` | `block(:quantizer)` | done, bit exact | stateless |
| `channels.phase_noise_gen` | | skipped: not done (filtered noise into a rotator; the parts exist) | |
| `channels.amp_bal`, `phase_bal`, `conj_fs_iqcorr` | | skipped: adaptive correction loops, receiver side | |
| `channels.impairments` | | skipped: a composition of the above | |

## Streams and state

The state of every block is explicit and returned; `generate/3` and
`NxFixed.Block.run/4` continue from it. Input split into any pieces gives the
same output bit for bit (tested for every block, in float and fixed point). The
state is:

* sources and rotators: the phase word (`u32`), see below
* noise: one `u32` xorshift word
* vector source: the read position
* fading: one phase word per component
* FIR: the delay lines; resampler: the unconsumed input and the position
* `channel_model` and other compositions (`ExSdr.Sources.Channel.sequence/2`): the
  list of the states of the parts (`:host` blocks)

`warmup` is `:unbounded` for every block whose state never forgets its initial
value (phase, generator, position). `NxFixed.Block` documents `warmup` as a
non-negative integer; a copy of such a block can only be started mid-stream if
the state is known (for the NCO it is `phase + n * inc`).

The resampler's output length depends on the data, so it returns `nil` instead
of an empty pair when a call produces nothing (Nx has no empty tensors).
`rate` is `{in, out}` reduced from the 32 bit step, exact over long streams.

## Oscillators: phase word, spurs

The phase is a word of `:phase_bits` bits (default 32, at most 32), advanced
modulo `2^bits`. Frequency resolution is `rate / 2^bits` (0.0002 Hz at 1 MHz), and the
phase never drifts because it is an integer. With initial phase `p0`, sample `n`
is the waveform at `p0 + n * inc` (the phase before the increment, as GNU Radio's
`sig_source` and `rotator`). `ddc` in `ExSdr.DSP.Blocks` advances first; the two
differ by one sample.

Waveforms follow GNU Radio (checked against it): square is `amplitude` where the phase
is negative and 0 elsewhere; triangle is `amplitude * (1 - |phase| / pi)`; sawtooth is
`amplitude * (1/2 + phase / (2 pi))`; the offset is added to the real part only;
the imaginary part of the non-sinusoidal waveforms is the same function of
`phase - pi/2`.

Fixed point: a sine table of `2^lut_bits` entries (format option, default 10) addressed by the top phase
bits, amplitude `2^frac - 1`. Dropping the low phase bits makes a sawtooth phase
error of one table step, which is one spur of level

    -6.02 * lut_bits dBc

at the worst frequency (test: 2^16 samples, 4099 turns, Q17: 48.1, 60.2, 72.2 dBc for 8, 10,
12 bits). Rounding the phase (`rounding: :nearest`) removes the bias, so the error power
against the ideal sine is 6 dB lower, but the spur is the same. Other frequencies turn the error
into noise. Float instance: no table, no visible spur (more than 140 dB below the carrier).
Triangle and sawtooth read the top `frac + 1` phase bits directly and have no table.
Waves that peak at exactly 1.0 saturate to `1 - 2^-frac`. Amplitude and offset are
compile time constants and must fit the format.

Against GNU Radio (float instance, 4096 samples, f32 NCO): sine and cosine 0.3 LSB (Q15)
and 102 dB SNR, triangle and sawtooth 0.13 LSB, square and constant exact. GNU Radio
converts the frequency through float32, which shifts the phase by up to 2e-5 rad
after 4096 samples: that is the bound's reasoning (see the case files).

## Noise

The generator is xorshift32 (`x ^= x << 13; x ^= x >> 17; x ^= x << 5`) on a `u32`
state, seeded through an integer hash (never zero). One draw per real sample, two per
complex sample. GNU Radio's random numbers cannot be reproduced bit for bit and we do not try:
the tests compare statistics with theory and with GNU Radio's own output.

* uniform: top `frac + 1` bits, range `[-amplitude, amplitude)` (float: 24 bits)
* gaussian, laplacian: inverse CDF table of `2^14` entries at the midpoints, addressed by the
  top draw bits, amplitude included. Tails stop at 4.0 sigma (gaussian) and 9.7 scale units
  (laplacian); GNU Radio's reach 4.9 sigma in 100000 samples. Variance within 0.1 percent of
  exact.
* complex samples: both components scaled by `1/sqrt(2)` (measured in GNU Radio's
  `noise_source_c`: gaussian components have variance `amplitude^2 / 2`, uniform
  `amplitude^2 / 6`)
* impulse: GNU Radio's `impulse(9)` as measured on 2 million samples: an exponential variate of mean
  10/7 kept when above 9, else 0. One sample in about 560 is nonzero, value `9 + Exp(mean 1.43)`
  times amplitude, never negative. The definition is a fit to these statistics, not GNU
  Radio's source. Table of `2^16` entries.
* `fastnoise_source`: a table of stored noise (`:samples`, a power of two, default 4096;
  GNU Radio uses 2^20) read at a random position each sample. GNU Radio reads a random
  start and then in order, which is not split invariant per sample; here every sample is
  a random read.
* In fixed point choose the amplitude so the peak fits: gaussian 4 sigma, laplacian 9.7
  scale units, impulse about 16 amplitude. Larger values saturate.
* complex laplacian and impulse noise exist here, but GNU Radio 3.10's
  `noise_source_c` hangs on them, so there is no reference.

Checks (`test/sources/noise_test.exs`): mean, variance (theory and GNU Radio), kurtosis,
Kolmogorov-Smirnov distance to the exact distribution, spectral flatness of the
averaged periodogram (0.99 or better), uncorrelated complex components, fixed point
equal to the float values rounded.

## Channel models

`channel_model` is the chain of GNU Radio's hierarchical block: timing offset
(`mmse_resampler_cc`), multipath FIR, frequency offset (rotator), noise (complex gaussian
`noise_voltage`). Without noise it equals GNU Radio to 0.16 LSB. It includes the
resampler even at `epsilon = 1`, as GNU Radio does, which advances the signal by 3 samples
and shortens it by 8; `resample: false` leaves it out.

Resampler: 8 taps, 129 phases, fraction rounded to 1/128, position in 32 bit fraction. GNU Radio's
taps are a table; ours are the least squares fit of a delay over `|w| < pi/2`, computed when the block is built (agreement 1.2e-5 with their
table, which is itself rounded to 5 digits; output difference 0.03 LSB measured). The ratio must
be a float32 value to follow GNU Radio's positions exactly (it stores float32). GNU Radio's scheduler leaves up to a few more input
samples unused at the end of a finite stream than our rule (every output whose window fits);
the comparison uses the common prefix.

Fading: sum of sinusoids (`ExSdr.Sources.Channel` moduledoc). Rayleigh over seeds: exponential
power with mean 1 and correlation `J0(2 pi fDTs m)`; Rician with K: mean power 1, power
variance `(2K + 1) / (K + 1)^2`. This is not GNU Radio's realisation. The `fading_rayleigh`
and `fading_rician` cases hold GNU Radio's output and the test compares its time averaged
power with ours. `selective_fading_model`: paths of
FIR (windowed sinc, `ntaps`, delay realised as `delay + ntaps/2`) and an independent fader each, summed.
GNU Radio's own delay filters are not reproduced.

`cfo_model`: the frequency (a phase word step) takes a gaussian step each sample, clipped to
`+-max_dev_hz`; the sample is rotated by the phase before the update. GNU Radio's exact update
order is not known to us; step statistics and clipping are tested.

## Differences from GNU Radio

* Bit-exact random numbers are not attempted (noise, fading, cfo): statistics only.
* Gaussian and laplacian tails are cut by the table (see Noise).
* `sig_source` takes frequency and rate (or `phase_bits` steps) as constants of the block;
  GNU Radio's setters would be new blocks here.
* The resampler tap table is our own design; `channel_model` always resamples unless told not to.
* `vector_source` without repeat gives zeros after the end (a scan cannot end the stream).
* Fixed-point constants (amplitude, offset, taps, beta) round to the format and must fit it;
  unity in Q15 is `1 - 2^-15`.
* The quantizer keeps GNU Radio's behaviour of saturating at 16 bit integers, not at `bits` bits.

## Regenerating the GNU Radio output

    scripts/gnuradio_golden.sh test/fixtures/gnuradio/sources test/dsp_cases/sources

(rootless podman image `exhdl-gnuradio`, no sudo). Chains that start with a source use input type
`"none"`. Cases of random blocks (`"stats_only": true`: noise, fastnoise, fading)
keep no samples: the script writes `stats.json` (per component n, mean, variance, kurtosis,
spectral flatness, count and moments of nonzero values; for complex output the power mean and
variance), and the tests compare our statistics with those. This keeps the fixtures under 1 MB.
