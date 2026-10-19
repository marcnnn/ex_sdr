# ExSdr

Software-defined radio building blocks for Elixir and Nx, each with an
exact Elixir reference and a fast form (native Highway SIMD kernel or a
compiled Nx computation) that property tests prove equal to it.

- **DSP** (`ExSdr.DSP.Blocks`): DDC, complex FIR, one-pole IIR, defined
  once over [nx_fixed](../nx_fixed) number formats; float reference or Q15,
  compared with GNU Radio 3.10.
- **Channel coding and modulation** (`ExSdr.Comm`): Reed-Solomon over
  GF(2^m), LFSRs and scramblers, Forney interleaver, convolutional codes
  with SIMD Viterbi decoding, constellations, permutations, OFDM with a
  compiled receiver front end and acquisition. See `docs/comm.md`.
- **DVB-T** (`ExSdr.DVBT`): a complete transmitter and receiver built from
  those blocks, bit-identical to GNU Radio's gr-dtv stage by stage and
  ahead of it end to end. See `docs/dvbt.md`.

The SIMD kernels are built with Google Highway 1.4.0 (fetched by the
Makefile, pinned hash) and verified on x86_64, aarch64 and ARMv7 (Zynq
Cortex-A9 under qemu). See `docs/cpu_backend.md`.

## Development

```sh
mix test                      # unit and property tests
mix test --only dvbt          # DVB-T against gr-dtv reference data
mix test --only gnuradio_live # DSP cases against live GNU Radio (native or rootless podman)
mix test --only dvbt_live     # DVB-T end to end against gr-dtv's receiver (podman)
scripts/gnuradio_golden.sh    # regenerate test/fixtures/gnuradio/ after changing a case
MIX_ENV=test mix run scripts/dvbt_realtime.exs   # receiver speed
```

## DSP cases

Each file in `test/dsp_cases/` describes one test once: input signal,
parameters, the `ExSdr.DSP.Blocks` block, the GNU Radio block chain, and
expectations with their reasoning for two layers:

1. `float_vs_gnuradio`: the block's float instance against GNU Radio
   (is the algorithm right?)
2. `fixed_vs_float`: each fixed-point variant (`bits`, `frac`, `rounding`,
   `lut_bits`) against the float instance (what does the format cost?)

`test/gnuradio/golden.py` runs the GNU Radio side, `test/gnuradio_reference_test.exs`
the ExSdr side; outputs are compared sample by sample. Adding a case or a
format variant is editing JSON and running `scripts/gnuradio_golden.sh`;
stale reference data fails the tests.

Requires Elixir 1.18+, a C++17 compiler, `make` and `curl` (the NIF
fetches Google Highway on first build).
