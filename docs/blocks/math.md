# Math family: stream math and type conversion

GNU Radio gr-blocks arithmetic, conversions, decimation and bit packing as
`NxFixed.Block` descriptions: `ExSdr.Math.block(name, opts)`, with `:format`
(`NxFixed.Num.float()` or `NxFixed.Num.q/3`) and the block's parameters.

Code: `lib/ex_sdr/math/` (`math.ex` the registry, `stream.ex` sample kernels,
`vector.ex` integrate, repeat, max/min and keep-N, `bits.ex` bit packing).
Cases: `test/dsp_cases/math/`, GNU Radio output `test/fixtures/gnuradio/math/`.
Tests: `test/math/`.

Status: **done** = GNU Radio case with float and fixed bounds, split
invariance and width check. **partial** = works, but a deviation or a missing
check is listed under notes. **skipped** = not implemented, reason given.

## Coverage

| GNU Radio block | ExSdr name | Status | Hardware | Notes |
|---|---|---|---|---|
| `multiply_const_ff` | `:multiply_const_ff` | done | stateless | constant quantized by `param/2` |
| `multiply_const_cc` | `:multiply_const_cc` | done | stateless | real constant |
| `add_const_ff` | `:add_const_ff` | done | stateless | |
| `add_const_cc` | `:add_const_cc` | done | stateless | `c: [re, im]` |
| `conjugate_cc` | `:conjugate_cc` | done | stateless | |
| `complex_to_mag_squared` | `:complex_to_mag_squared` | done | stateless | |
| `complex_to_mag` | `:complex_to_mag` | partial | stateless | float only (sqrt); see notes |
| `complex_to_arg` | `:complex_to_arg` | partial | stateless | float only (atan2) |
| `complex_to_real` | `:complex_to_real` | done | stateless | |
| `complex_to_imag` | `:complex_to_imag` | done | stateless | |
| `complex_to_float` | - | skipped | - | two outputs; `NxFixed.Block` has one output |
| `float_to_complex` | `:float_to_complex` | partial | stateless | pair input `{re, im}`; no GNU Radio case (two sources) |
| `magphase_to_complex` | `:magphase_to_complex` | partial | stateless | pair input, float only; no GNU Radio case |
| `abs_ff` | `:abs_ff` | done | stateless | most negative value saturates |
| `nlog10_ff` | `:nlog10_ff` | partial | stateless | float only (log10); chained after `abs_ff` in the case |
| `add_ff`, `sub_ff`, `mul_ff`, `div_ff` | same | partial | stateless | pair input; unit tests, no GNU Radio case |
| `add_cc`, `sub_cc`, `mul_cc`, `div_cc` | same | partial | stateless | pair input; unit tests, no GNU Radio case |
| `threshold_ff` | `:threshold_ff` | done | feedback | hysteresis; output is a bit (u8) |
| `moving_average_ff` | `:moving_average_ff` | done | feed-forward | `length`, `scale`; window = taps |
| `moving_average_cc` | `:moving_average_cc` | done | feed-forward | |
| `integrate_ff` | `:integrate_ff` | done | feedback | frame, rate `{decim, 1}`; saturates once |
| `integrate_cc` | `:integrate_cc` | done | feedback | |
| `delay` | `:delay` | done | feed-forward | latency and warm-up = delay; see notes |
| `repeat` | `:repeat` | done | stateless | frame, rate `{1, interp}` |
| `keep_one_in_n` | `:keep_one_in_n` | done | control | host; keeps item n-1 of each group |
| `keep_m_in_n` | `:keep_m_in_n` | done | control | host; keeps the first m of each n |
| `max_ff` | `:max_ff` | done | stateless | frame, rate `{vlen, 1}` |
| `min_ff` | `:min_ff` | done | stateless | frame, rate `{vlen, 1}` |
| `stream_to_vector` | `:stream_to_vector` | partial | stateless | identity on the flat stream; vector length is framing only |
| `vector_to_stream` | `:vector_to_stream` | partial | stateless | identity, as above |
| `interleave` | - | skipped | - | N inputs; needs a port model in `NxFixed.Block` |
| `deinterleave` | - | skipped | - | N outputs; needs a port model in `NxFixed.Block` |
| `unpack_k_bits_bb` | `:unpack_k_bits_bb` | done | stateless | frame, rate `{1, k}`; MSB first |
| `pack_k_bits_bb` | `:pack_k_bits_bb` | done | stateless | frame, rate `{k, 1}`; MSB first |
| `packed_to_unpacked_bb` | `:packed_to_unpacked_bb` | partial | stateless | unpack with MSB order; GNU Radio needs an endianness enum that `golden.py` cannot pass |
| `unpacked_to_packed_bb` | `:unpacked_to_packed_bb` | partial | stateless | pack with MSB order; as above |
| `repack_bits_bb` | `:repack_bits_bb` | done | control | host; LSB first; bit state carried |
| `char_to_float` | `:char_to_float` | done | stateless | `scale` divides, as in GNU Radio |
| `float_to_char` | `:float_to_char` | done | stateless | rounds to nearest, saturates to s8 |
| `float_to_short` | - | skipped | - | `golden.py` has no int16 sink; would be `float_to_char` at 16 bits |

Hardware classes follow `NxFixed.Block`. Feed-forward blocks keep a delay
line only; feedback blocks depend on their own state (accumulator,
hysteresis); control blocks count positions or hold a bit buffer.

## Choices

* **Parameters.** `ExSdr.Math.param/2` quantizes constants so they fit the
  format: the fraction keeps as many bits as the integer part allows. In Q15,
  0.7 is Q(16, 15) but 127 is Q(16, 8), instead of saturating.
* **Products and sums.** Exact (`NxFixed.Num`), one `quantize/2` at the
  output (round to nearest or truncate per format, saturate to the width).
  Kernels quantize their inputs first, so the width check bounds products and
  sums by the format, not by the container (s32 for Q17 samples).
* **Pairs.** Two-input blocks take the sample `{a, b}` (complex: `{{ar, ai},
  {br, bi}}`). A two-input block is one `NxFixed.Block` with an item that is a
  pair.
* **Division.** Fixed point is an exact integer quotient of the aligned raw
  values, rounded once; division by zero saturates by the sign of the
  numerator (GNU Radio gives inf). Complex division is `a * conj(b) / |b|^2`
  with one rounding per component.
* **Bit blocks.** Host and frame kernels on `u8` tensors; `repack_bits_bb`
  takes a tensor and returns a binary, since Nx cannot hold empty tensors
  and a chunk may produce no byte.
* **Keep blocks.** The state is the position in the cycle, so any split gives
  the same output. A chunk with no kept item raises `ArgumentError` (the
  empty output cannot be an Nx tensor in this version); chunks of at least n
  items always keep one.

## Deviations from GNU Radio

* **Delay.** GNU Radio flushes the samples still in the delay line when the
  flow graph stops, so its output is `delay` items longer. The case records
  this (`reference_extra_items`); the test checks that the block's final
  state holds exactly those samples.
* **complex_to_mag.** GNU Radio uses VOLK's approximate square root. On the
  case input the largest difference to the exact magnitude is 0.0028 (reference
  0.609375, exact 0.6122). The float instance is exact; the case bound is
  0.005.
* **nlog10_ff.** GNU Radio's log is an approximation; worst difference 2.9e-5
  dB. Float only: no fixed-point log yet.
* **complex_to_arg.** Worst difference 1e-6 rad against GNU Radio.
* **char_to_float.** GNU Radio divides by `scale`; the block does too.
* **Fixed-point mag and arg, nlog10, magphase.** Not implemented (a LUT or
  CORDIC is the next step). The block refuses a fixed format with an
  `ArgumentError`.
* **Integrate.** Sums need headroom: `decim` times the input range. The tests
  use Q(20, 15) for Q15 input and decimation 8.

## Not yet checked against GNU Radio

The two-input blocks (`add_*`, `sub_*`, `mul_*`, `div_*`, `float_to_complex`,
`magphase_to_complex`), the packing aliases, and `stream_to_vector` as a
chain: the unit tests compare with float arithmetic and with the fixed-point
rounding bounds. They need a second input source in
`test/gnuradio/golden.py` (or an endianness enum) to be checked against GNU
Radio.

## Requests outside this branch

* `NxFixed.Block`: `warmup` for the threshold is not bounded (the hysteresis
  forgets its start only after the input leaves the band). The block sets
  `warmup: nil`; the type says `non_neg_integer`. Either allow `nil` (not
  bounded) or a separate field.
* `test/gnuradio/golden.py`: a second source for two-input blocks, and enum
  arguments (`endianness`), so the pair blocks and packing aliases get GNU
  Radio cases.
* `NxFixed.Dataflow` does not support `Nx.sign/1`; the divider uses
  comparisons instead.
