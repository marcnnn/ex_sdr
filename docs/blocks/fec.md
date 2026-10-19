# FEC blocks (`ExSdr.FEC`)

The gr-fec blocks that are not LDPC: the generic encoder / decoder
framework, convolutional codes with a soft Viterbi decoder, repetition,
dummy and polar codes, puncturing and the BER meter, as `NxFixed.Block`s.
`ExSdr.FEC.block(name, opts)` builds a block, `ExSdr.FEC.run(block, input,
state)` runs it and returns `{output, state}`.

Code objects (`ExSdr.FEC.Generic`) are plain data, GNU Radio's
`generic_encoder` / `generic_decoder` interface: `input_size`, `output_size`,
`rate`, `history`, `shift`, the conversions and `work/3` for one frame. Every
code in this family is one implementation module of that behaviour
(`ExSdr.FEC.Conv`, `Repetition`, `Dummy`, `Polar`, `PolarDecoder`); the stream
blocks (`:encoder`, `:decoder`) are `:frame` blocks over `work/3`, the tagged
stream and PDU blocks are `:host` wrappers around the same core.

Items: bits are `u8` tensors with 0 or 1. Soft values for decoders: the Viterbi
decoder takes GNU Radio's 8 bit convention (`u8`, 0 a sure 0, 255 a sure 1, 128
or 127 no information), the others `f32` with positive meaning bit 1.
`ExSdr.FEC.Generic.to_u8/3` converts floats to the 8 bit form like
`blocks.float_to_uchar(scale=48, bias=128)` (checked equal on the fixture).

## Coverage

Status: done (equal to GNU Radio on the fixtures), partial, skipped.
Hardware class as in `NxFixed.Block`.

| GNU Radio | ExSdr.FEC | Status | Hardware |
|---|---|---|---|
| `generic_encoder`, `generic_decoder` | `Generic` behaviour and struct, `set_frame_size/2`, `to_u8/3` | done | |
| `encoder`, `decoder` | `block(:encoder / :decoder, code: object)`, `Generic.encode_frames/3`, `decode_frames/3` | done: frames, look-ahead history, carried state | by code |
| `tagged_encoder`, `tagged_decoder` | `block(:tagged_encoder / :tagged_decoder, code:, mtu:)` (`:host`, packets) | done: sizes and bits equal, also padded terminated | control |
| `async_encoder`, `async_decoder` | `block(:async_encoder / :async_decoder, code:, packed:, rev_unpack:, rev_pack:, mtu:)` (`:host`, PDUs) | done, see deviations | control |
| `extended_encoder`, `extended_decoder` | `ExSdr.FEC.Extended`, `block(:extended_encoder / :extended_decoder)` | partial: conversion, puncturing, one decoder; no `ann` / `conv_bit_corr_bb`, no threading | control |
| `extended_tagged_encoder`, `extended_tagged_decoder`, `extended_async_encoder` | tagged / async blocks plus `Extended` | partial: compose them yourself | control |
| `threaded_*`, `capillary_threaded_*` | none | skipped: scheduler parallelism, not a code | |
| `cc_encoder` | `block(:cc_encoder, ...)`, `Conv.encoder/1` | done: all four modes, start state, padding, negative polynomials, any K 2-31 and rate | feed_forward |
| `cc_decoder` | `block(:cc_decoder, ...)`, `Conv.decoder/1`, `ConvViterbi` | done: decisions equal on the same 8 bit input for K = 7 rate 2 in all four modes, `end_state`, chained start state; any K and rate as a superset | memory |
| `ccsds_encoder`, `encode_ccsds_27_bb`, `decode_ccsds_27_fb` | `cc_*` with `polys: [79, -109]` (the CCSDS inversion is a negative polynomial) | skipped as separate blocks: the old Viterbi decoder uses other metrics | |
| `conv_bit_corr_bb` | none | skipped: blind code detection, not part of this family | |
| `repetition_encoder`, `repetition_decoder` | `block(:repetition_encoder / :repetition_decoder, ...)` | done | stateless |
| `dummy_encoder`, `dummy_decoder` | `block(:dummy_encoder / :dummy_decoder, ...)` | done | stateless |
| `puncture_bb`, `puncture_ff` | `block(:puncture_bb / :puncture_ff, puncsize:, puncpat:, delay:)`, `Puncture` | done: random streams with delay and wide patterns | stateless |
| `depuncture_bb` | `block(:depuncture_bb, ..., symbol:)` | done | stateless |
| `polar_encoder` (packed and unpacked), `polar_encoder_systematic` | `block(:polar_encoder / :polar_encoder_systematic, ...)`, `Polar` | done | transform |
| `polar_decoder_sc`, `polar_decoder_sc_systematic` | `block(:polar_decoder_sc / ..., ...)`, `PolarDecoder` | done: decisions equal, also on hard and zero inputs | transform |
| `polar_decoder_sc_list` | `block(:polar_decoder_sc_list, max_list_size:)` | done including GNU Radio's path bookkeeping flaw (`list_mode: :gnuradio`), plus the exact algorithm (`:exact`) | transform |
| `fec.polar.channel_construction` (BEC, Bhattacharyya) | `Polar.frozen_bit_positions/3`, `z_parameters/2`, `bec_capacities/2` | done: positions equal, parameters within 1e-12 | |
| Tal-Vardy construction (scipy) | none | skipped: needs scipy's solver | |
| (not in gr-fec) | `Polar.frozen_bit_positions(n, k, method: :dega)` | extra: Gaussian approximation density evolution | |
| `ber_bf` | `block(:ber_bf, test_mode:, berminerrors:, ber_limit:)`, `Ber` | done: streaming value equal, test mode by its rules | control |
| `bercurve_generator`, `fec_test` (python) | none | skipped: test benches | |
| `tpc_encoder`, `tpc_decoder` | none | skipped: turbo product codes, not in this part | |

## LDPC

see docs/blocks/fec_ldpc.md

## Convolutional codes

GNU Radio's conventions are kept: polynomials are masks over the K-bit
register with bit 0 the newest input (K = 7: `[79, 109]` is `0o171, 0o133`
read the other way round), a negative polynomial inverts that output,
`start_state` has the oldest input in its top bit. The encoder is a shift
register of inputs (`:feed_forward`, warm-up K - 1 bits in streaming mode);
modes:

* `:streaming`: register carried between frames; the decoder reads
  `rate * (K - 1)` items past the frame (GNU Radio's history), so the
  `:decoder` block holds the last frame back until the next call (output
  `nil` while no frame is complete, `latency` is one frame) and chains the start
  state of the next frame from its traceback. The first frame starts with
  all states equally likely.
* `:terminated`: start in `start_state`, K - 1 tail bits back to it,
  `padded` fills the code frame to whole bytes. Optional `end_state`.
* `:truncated`: start in `start_state`, no tail; the last K - 1 bits come from
  the best end state.
* `:tailbiting`: the register starts with the last K - 1 input bits; the
  decoder wraps the frame around `6 * (K - 1)` steps.

### Soft Viterbi in integers (`ExSdr.FEC.ConvViterbi`)

The trellis is `ExSdr.Comm.Convolutional`'s (output table of the register
values, with the polynomial bits reversed) and is not modified; this module adds
GNU Radio's state numbering and soft branch metrics:

* branch metric `(1 + sum_j (expected_j xor symbol_j)) >>> 3`, expected bits are
  0 or 255 (63 at most for rate 1/2);
* where a branch is the complement of its butterfly's reference branch (first and
  last tap of every polynomial set, as usual) the metric is `max - metric` like
  volk's kernel, otherwise the distance is computed directly;
* path metrics `u8`, saturating add, minimum subtracted every step; ties go to
  the larger predecessor state;
* one decision bit per state and step, traceback from the best state (smallest
  number on ties); the decision at step `t` is input bit `t - (K - 1)`.

This is the data a hardware decoder holds: `2^(K-1)` metrics of 8 bits, one
decision bit per state in block RAM (`:memory` class), no division or float.
The portable C kernel of volk wraps around where the SIMD kernels saturate;
they differ only when the spread between states exceeds 255, which clean
input does reach (K = 7, rate 1/2, noise below sigma 100 on 0/255 symbols).
This implementation follows the SIMD kernels, which is what GNU Radio runs on
x86 and ARM; the fixtures were made on aarch64 (NEON).

## Polar codes

The code is GNU Radio's transform (`Polar.transform/1`, volk's
`encodeframepolar`) with natural order info bit positions. The decoders are
the lockstep tree form of its successive cancellation: a node of size n has two
children (the two halves of the received block), bit `2j` gets `minsum(La_j,
Lb_j)`, bit `2j + 1` gets `Lb_j + (1 - 2 u_2j) La_j`, and the decided
`u_2j xor u_2j+1` and `u_2j+1` go back to the children. All operations are
rounded to `f32`, so decisions equal GNU Radio's. The list decoder has two modes:

* `:gnuradio` (default) reproduces `scl_list.cc` buffer for buffer. That code
  has a flaw: when a surviving path has to take over a buffer it takes the
  first discarded one, which can still be the source another survivor copies
  from, so that survivor continues with a foreign history. In the fixtures the
  exact decoder disagrees with GNU Radio in 2 of 216 list frames (N = 256, L = 4
  and N = 128, L = 16, noise sigma 0.75-0.8, near the failure threshold). It also needs a power of two list size.
* `:exact`: the textbook list decoder. In the seeded statistical test it
  fails on fewer frames than SC and no more often than the `:gnuradio` mode.

Exactly tied path metrics (hard +-1 input with many equal sums) can be ordered
differently than `std::sort` does; use noisy or real valued input when
comparing.

## Deviations from GNU Radio

* `cc_decoder`: supports any K and rate (GNU Radio only K = 7, rate 2).
  Saturating metrics as described above.
* `cc_encoder` padded terminated frames: the padding bits are zeros (GNU Radio does
  not write them). `cc_decoder` padded and the PDU decoder: the decoded frame has
  the length GNU Radio allocates, with the bits it does not write set to 0
  (GNU Radio leaves what was in its buffer).
* `cc_encoder` polynomials above K bits are rejected (GNU Radio would use
  older register bits).
* `async_*` and `tagged_*`: the same size rules (`set_frame_size`, `diff`,
  MTU), including the odd handling of padded terminated codes.
* Decoders with history return `nil` while no frame is complete instead of
  blocking.
* Polar codes need sorted frozen positions (GNU Radio assumes it silently).
  `set_frame_size` of polar codes does nothing and returns `false`.
* `rate/1` follows each GNU Radio object, which is not uniform (`cc_encoder`
  gives the number of outputs, the polar encoder K / N).
* `repetition_decoder` counts `> 0` as a one, `dummy_decoder` slices `>= 0`
  (volk): both as in GNU Radio.
* `ber_bf` in streaming mode gives one value per call; the last value is the
  BER of everything seen so far, the others depend on how the stream is cut
  (also in GNU Radio). Zero errors give `:neg_infinity`.

## Checking against GNU Radio

Objects are constructed in Python and driven frame by frame through
`generic_work` (a capsule around a numpy buffer, `test/gnuradio/fec_common.py`),
so streaming state and look-ahead are part of the data:

    test/gnuradio/fec_conv_golden.py    -> test/fixtures/gnuradio/fec/conv.json
    test/gnuradio/fec_polar_golden.py   -> test/fixtures/gnuradio/fec/polar.json
    test/gnuradio/fec_misc_golden.py    -> test/fixtures/gnuradio/fec/misc.json

each run with `podman run --rm -v "$PWD/test:/work/test:ro,Z" -v
"$PWD/test/fixtures/gnuradio/fec:/out:Z" -w /work exhdl-gnuradio python3
test/gnuradio/<script> /out/<file>` (the image `exhdl-gnuradio` already has
gr-fec). The punctured stream cases are ordinary `test/dsp_cases/fec/punct_*.json`
cases for `scripts/gnuradio_golden.sh test/fixtures/gnuradio/fec test/dsp_cases/fec`.

Tests (`test/fec`): `conv_test.exs`, `polar_test.exs`, `repetition_test.exs`
(with dummy), `puncture_test.exs` (with `ber_bf`), `puncture_cases_test.exs`,
`generic_test.exs` (tagged, PDU, blocks, extended chains, soft gain over
hard decisions). Bit results are compared for equality; the statistical tests are
seeded and small.
