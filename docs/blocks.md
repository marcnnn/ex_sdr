# Porting GNU Radio blocks

How a block family from GNU Radio becomes part of ex_sdr. Written for the
people (and agents) doing it in parallel, one family per branch.

## Layout per family

| What | Where |
|---|---|
| Code | `lib/ex_sdr/<family>/*.ex`, modules `ExSdr.<Family>.*` |
| Tests | `test/<family>/*_test.exs` |
| GNU Radio cases | `test/dsp_cases/<family>/*.json` |
| GNU Radio output | `test/fixtures/gnuradio/<family>/` (committed) |
| Coverage and notes | `docs/blocks/<family>.md` |

Families work only inside these paths, so branches merge without conflicts.
Changes elsewhere (`c_src/`, `lib/ex_sdr/nif.ex`, `mix.exs`, other
families, nx_fixed) go into the branch's report as a request instead.

## A block

* Generic first: parameters, not a protocol. A block that GNU Radio
  configures with taps, a constellation or a polynomial takes the same here.
* DSP on samples is written once against `NxFixed.Num` (see
  `ExSdr.DSP.Blocks`, `ExSdr.DSP.FFT`): `:format` gives the float reference
  (`NxFixed.Num.float()`) or a fixed-point instance (`NxFixed.Num.q/3`). The
  fixed-point instance must pass the width check (`NxFixed.analyze/3`), so
  it is bit-exact on every backend and can become hardware.
* Bit and byte blocks use integer Nx (`u8` tensors) or Elixir binaries.
* Streams: the state is explicit and returned, so input split into any
  pieces gives the same output as one call (see
  `ExSdr.Comm.Scrambler`, `ExSdr.Comm.ConvInterleaver`).
* Each family module implements the `NxFixed.Block` behaviour:
  `block(name, opts)` returns a `%NxFixed.Block{}` with kind, rate, latency,
  warm-up, item types, hardware class and the GNU Radio name. This is what
  the runtime and ex_hdl read; it must be true.
* No native code in these branches. Speed comes later (nx_simd generated
  code, Highway kernels in `c_src/`), once the Nx version is the reference.

## Checking against GNU Radio

A case file names the GNU Radio chain by import path, the input signal and
the expected agreement; `scripts/gnuradio_golden.sh` runs it (podman image,
no local GNU Radio needed) and writes the fixture:

    scripts/gnuradio_golden.sh test/fixtures/gnuradio/<family> test/dsp_cases/<family>

`scripts/gnuradio_golden.sh` is the only entry point. The runner is
`test/gnuradio/golden.py`; its header has the full case format. In short:

* **Inputs.** `"input"` (one) or `"inputs": [...]` (connected to ports 0..n-1
  of the first block). Types: `"complex"` and `"real"` (tones, steps, noise;
  Q15), `"bytes"` (random, `"modulus"` 2 for bits), `"none"` for chains that
  start with a source (`"samples"` items are taken). Several inputs are drawn in
  order from the case's seed.
* **Outputs.** `"output": {"type": ...}` or `"outputs": [...]` (ports 0..m-1 of the
  last block): `"complex"`, `"real"`, `"bytes"`, `"short"` (int16 sink,
  `gnuradio.s16le`), `"int"` (`gnuradio.s32le`). The default is the first
  input's type. Files after the first are `input1.*`, `gnuradio1.*`.
* **Statistics.** `"stats_only": true` stores `stats.json` instead of samples,
  for random blocks.
* **Arguments.** Strings in a block's `"args"` are Python expressions with `pi`,
  `math`, `np`, the modules `gr`, `blocks`, `filter`, `analog`, `digital`, `fec`,
  `channels`, and the case's `params` as bare names (when they are valid
  identifiers) or as the dict `p` (`p["pass"]`; an argument that is just a param
  name always gives that param). They can build objects
  (`digital.constellation_qpsk().base()`) and name enums (`gr.GR_MSB_FIRST`).
* **Other fixtures.** Taps, tagged streams and PDUs do not fit the case format.
  Scripts listed in `HOOKS` in `golden.py` write them when the family's cases
  are run: `digital/framing.json`, `filters/taps.json`, `filters/pfb.json`.
  A new family adds its script there.
* Some GNU Radio blocks are float-only on one port or take bytes on a control
  port (`sample_and_hold_ff`); a case then lists the input types the block has.

`ExSdr.Test.GnuRadioFixture.family/1` loads all cases of a family:
`[{case, %{input:, reference:}}]`; with several inputs `input` holds one
`[a, b]` per sample, `inputs` the lists, and `references` all outputs.
Look at `test/dsp_cases/fir_ccf_31tap.json` and
`test/gnuradio_reference_test.exs` for the existing pattern, and at the math
cases `add_ff`, `float_to_short` and `packed_to_unpacked_bb_msb` for two inputs,
an int16 output and an enum.

GNU Radio is not always repeatable on the byte: `filters/fft_filter_ccf_decim2`
and `carrier/corr_est_cc_sync32` (FFT filtering) change in float rounding (up
to 1e-7), and `filters/rational_resampler_ccf_5_3` sometimes ends one item
later, from run to run. Regenerating these families can therefore show a diff
there; keep the committed files unless the case changed.

Families whose cases need their own runner are listed in `RUNNERS` in
`golden.py` (carrier); `scripts/gnuradio_golden.sh` dispatches to it.

Per block, as far as it applies:

1. Float instance vs GNU Radio: per-sample bound and SNR, with the
   reasoning in the case file (`"rationale"`). Bit and byte blocks: equal.
2. Fixed-point instances vs float: bound and SNR from the format, with
   reasoning.
3. Split invariance: two chunks with the state carried equal one call
   (StreamData property where cheap).
4. Width check passes for each fixed format used in the tests.

GNU Radio sometimes differs by design (start-up transients, different tap
conventions, scaling). Match it where the difference is a convention; where
it is not, document it in `docs/blocks/<family>.md`.

## Coverage table

`docs/blocks/<family>.md` starts with one row per GNU Radio block of the
family: GNU Radio name, ExSdr function, status (done, partial, skipped and
why), hardware class.
