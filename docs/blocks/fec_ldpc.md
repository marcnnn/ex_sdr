# LDPC codes (`ExSdr.FEC.LDPC`)

GNU Radio's LDPC blocks from gr-fec and gr-dtv, plus soft decoders for
hardware. `ExSdr.FEC.LDPC.block(name, opts)` returns the
`NxFixed.Block`; `ExSdr.FEC.LDPC.run(block, input)` runs it compiled.

| GNU Radio | ExSdr | Status | Hardware |
|---|---|---|---|
| `fec.ldpc_H_matrix(alist, gap)` | `LDPC.h_matrix/2` | done | (host) |
| `fec.ldpc_G_matrix(alist)` | `LDPC.g_matrix/1` | done | (host) |
| `fec.ldpc_par_mtrx_encoder` (`_make`, `_make_H`) | `:ldpc_par_mtrx_encoder` | done, equal | transform |
| `fec.ldpc_gen_mtrx_encoder` | `:ldpc_gen_mtrx_encoder` | done, equal | transform |
| `fec.ldpc_encoder` (alist, `cldpc`) | `:ldpc_encoder`, `LDPC.cldpc/1` | done, equal | transform |
| `fec.ldpc_bit_flip_decoder` (H or G matrix) | `:ldpc_bit_flip_decoder` | done, equal | transform |
| `fec.ldpc_decoder` (alist, sum-product) | `:ldpc_decoder` | done, equal | transform (float only) |
| `dtv.dvb_ldpc_bb` | `:dvb_ldpc_bb`, `LDPC.DVBS2` | done for DVB-S2, S2X and T2 normal and short frames, equal; VL-SNR punctured/shortened codes, medium frames and the 128APSK padding not included | transform |
| (none) | `:ldpc_ira_encoder`, `LDPC.IRA` | structured repeat-accumulate codes from parameters | transform |
| (none) | `:ldpc_bp_decoder` | sum-product, flooding or layered, float reference | transform (float only) |
| (none) | `:ldpc_min_sum_decoder` | min-sum, normalised and offset, flooding or layered, float or integer | transform |

All blocks are `:frame` blocks on whole codewords and run any number of
frames in one call (batched on the last tensor axis inside).

## Codes

`ExSdr.FEC.LDPC.Code` holds the parity-check matrix H (sparse,
`ExSdr.FEC.LDPC.Matrix`), `n`, `k`, the codeword position of every
information bit and the encoder data. Any H works for decoding.

* `h_matrix(alist, gap)`: H = [T A B; E C D] with T upper triangular,
  Richardson-Urbanke encoding as gr-fec (`p2 = phi^-1 (D s + E T^-1 B s)`,
  `p1 = T^-1 (A p2 + B s)`). Codewords `[p1, p2, s]`, information last.
* `g_matrix(alist)`: G = [I P], H = [P' I], information first.
* `cldpc(alist)`: gr-fec's `cldpc` (`ldpc_encoder`/`ldpc_decoder`):
  Gaussian elimination of H with column swaps; `k = n - rank(H)`, the
  information bits where the elimination puts them.
* `IRA.code(n:, k:, q:, table:, group:)`: information bit
  `group * r + c` feeds parity checks `(a + c q) mod (n - k)` for the
  addresses `a` of table row `r`; staircase parity. `DVBS2.code(frame,
  rate, standard)` configures it with the tables of EN 302 307-1/-2 (53
  codes, `lib/ex_sdr/fec/ldpc/dvb_s2_tables.txt`, extracted from gr-dtv).

The host reference encoders (`Code.encode/2`) follow gr-fec's and
gr-dtv's algorithms. The tensor encoders use the code's generator
(`G' s` over GF(2), a dense constant matrix: XOR trees in hardware) or,
for IRA codes, a gather of the accumulator addresses plus a prefix XOR
in `log2(n - k)` steps, which stays small for DVB-S2 sizes.

## Decoders

LLRs at the interface are positive for 1, as GNU Radio's FEC decoders
and `ExSdr.Digital`'s soft decoder. Inside, the sign is flipped (the
usual convention) and a bit is 1 when its posterior is negative; zero
decides 0 (GNU Radio: `llr > 0` is 1).

Every decoder starts from the hard decisions of the input; a frame whose
decisions satisfy all checks is done (also before the first iteration)
and keeps its result while other frames of the batch continue, so the
result per frame is GNU Radio's early stopping. Output: the information
bits, or all `n` decisions with `output: :codeword`.

* Bit flip (gr-fec): flip every bit in the largest number of failed
  checks, at most `max_iter` times.
* Sum-product (`:ldpc_decoder`, `:ldpc_bp_decoder`): gr-fec's
  `awgn_bp`: `R = 2 atanh(prod tanh(clip(Q/2, +-18)))` over the other
  edges in row order, `f64`; posteriors summed in column order.
* Min-sum (`:ldpc_min_sum_decoder`): magnitudes from the two smallest
  inputs of a check (min1/min2), signs by XOR; `scale:` multiplies
  (normalised min-sum), `offset:` subtracts and clips at 0.
* Schedules: `:flooding` (all checks, then all variables) and
  `:layered` (rows grouped greedily into layers without shared
  variables; each layer reads and updates the posteriors).

The graph is index tables computed on the host (`ExSdr.FEC.LDPC.Graph`):
check-to-variable gathers on a padded `m x dc` grid, variable sums
through edge gathers, per-layer gathers for the layered update. All
accesses are `Nx.take` along axis 0, no scatters.

### Integer format

`format: NxFixed.Num.q(bits, frac)`: channel LLRs and check messages are
`bits`-bit integers with `frac` fractional bits, saturated symmetrically
to +-(2^(bits-1) - 1) (so negation is exact); posteriors use
`state_bits` (default `bits + 2`, at most 16) with the same saturation.
Arithmetic is exact in `s32` and saturated where stored. The scale is
rounded to 1/16 (`x * round(16 scale)`, shifted with rounding), the
offset to the input LSB. `LDPC.quantize_llr/2` rounds float LLRs into
the format.

### Width check and hardware

`block.fun` is a pure dataflow graph: `max_iter` iterations unrolled
(`loop: :unrolled`), only operations `NxFixed.WidthCheck` understands
(takes with constant index tables, constant slices, elementwise ops,
sums; maxima by halving). Integer instances pass `NxFixed.analyze/3`;
the tests analyse two iterations, which covers every later one because
everything carried between iterations is saturated to its range.
`LDPC.run/3` compiles the same iteration function in a `while` loop that
stops when all frames are done, with the index tables passed as
arguments (XLA compiles large constants slowly); results are identical
(tested).

## Checks

| Test | What |
|---|---|
| `ldpc_gnuradio_test` | 14 cases (`test/dsp_cases/fec/ldpc/`): every encoder equal, bit flip and sum-product decoders give equal bits; gr-fec's example matrices n = 8-1800 |
| `ldpc_dvb_s2_test` | all 53 DVB-S2/S2X/T2 codes equal to gr-dtv's `dvb_ldpc_bb` (SHA-256 of a frame, parity bits for short frames), random IRA parameters, decoding a short frame |
| `ldpc_code_test` | alist round trip; H c = 0 and information positions for random messages on every code kind and on random matrices (Richardson-Urbanke, G, rank deficient H) |
| `ldpc_decoder_test` | dataflow equals loop; frames independent of batching; error-free input; fixed vs float; seeded BER/FER |
| `ldpc_width_test` | width check of integer min-sum (formats, schedules, corrections), bit flip, encoders |

Seeded BER/FER (n = 300, k = 152, 400 frames per point, 50 iterations,
BPSK in AWGN), frame errors:

| Eb/N0 | sum-product flooding | sum-product layered | normalised min-sum 0.75 | same, 6-bit | offset min-sum 0.5, 6-bit | min-sum | bit flip |
|---|---|---|---|---|---|---|---|
| 1.5 dB | 136 | 126 | 151 | 147 | 155 | 174 | 398 |
| 2.5 dB | 11 | 8 | 10 | 10 | 9 | 15 | 393 |
| 3.5 dB | 0 | 0 | 0 | 0 | 0 | 0 | 352 |

The test bounds normalised and offset min-sum (float and 6-bit) to 0.2 dB
and plain min-sum to 0.4 dB of the sum-product reference at this code's
waterfall slope, and the 6-bit decoder to 0.1 dB of float min-sum.

## Differences from GNU Radio

* GNU Radio's encoders and decoders take a frame size that is a multiple
  of k (or n); here any number of whole frames per call.
* `ldpc_H_matrix` builds T^-1 and phi^-1 with a floating point LU
  inverse and checks the result mod 2, which can fail for large
  matrices; here the inverses are exact over GF(2). Back substitution in
  gr-fec assumes T is upper triangular with a unit diagonal without
  checking it: with gap 1 instead of 2 for `n_0100_k_0042_gap_02.alist`
  gr-fec encodes words that violate a parity check. Here that is
  checked and the gap rejected. A gap larger than needed is accepted by
  both and encodes the same.
* `ldpc_decoder` (`awgn_bp`) checks only the first `rank(H)` rows for
  its stopping rule; here all rows. Equal for full-rank H (all of
  gr-fec's example matrices).
* The sum-product decoder uses XLA's `tanh`/`atanh`, which may differ
  from libm in the last bit; decisions can only differ where a posterior
  is within rounding of 0 (none in the fixtures).
* `dvb_ldpc_bb`: the VL-SNR codes of DVB-S2X (puncturing, shortening),
  medium frames and the 6 zero bits appended for 128APSK are not
  included.

## Regenerating the reference data

    podman run --rm -v "$PWD/test:/work/test:Z" -w /work exhdl-gnuradio \
      python3 test/gnuradio/fec_ldpc_golden.py test/dsp_cases/fec/ldpc \
        test/fixtures/gnuradio/fec/ldpc_alist test/fixtures/gnuradio/fec
    podman run --rm -v "$PWD:/work:Z" -w /work exhdl-gnuradio-dtv \
      python3 test/gnuradio/fec_ldpc_golden.py dvb \
        lib/ex_sdr/fec/ldpc/dvb_s2_tables.txt test/fixtures/gnuradio/fec/ldpc_dvb_s2/dvb.json

The alist files in `test/fixtures/gnuradio/fec/ldpc_alist` are copied
from GNU Radio (`/usr/share/gnuradio/fec/ldpc`, GPL-3.0-or-later like
the rest of GNU Radio; the files carry no header of their own). The
DVB address tables are the standard's (ETSI EN 302 307-1 annexes B and
C, EN 302 307-2), as tabulated in gr-dtv.
