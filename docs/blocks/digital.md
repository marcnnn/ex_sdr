# Digital blocks (`ExSdr.Digital`)

The gr-digital blocks for modulation, symbol mapping, scrambling and
framing, as `NxFixed.Block`s: `ExSdr.Digital.block(name, opts)` builds a
block, `ExSdr.Digital.run(block, input, state)` runs it and returns
`{output, state}`. Synchronisation (timing, carrier, equalisers) and FEC are
not part of this family.

Constellations are plain data (`ExSdr.Digital.Constellation`), built with
GNU Radio's exact point order and `pre_diff_code`. The generic pieces are
parameterised, GNU Radio's protocol specific classes are configurations:
`psk_mod` is `generic_mod` with `Constellation.psk/3`, `crc32_bb` is the CRC
module with GNU Radio's CRC-32 parameters.

Bits, symbols and bytes are `u8` tensors in stream blocks; complex samples
are `{re, im}` of the block's `:format` (float: `f32`; fixed point: `Fixed`).
Packet blocks (`:host`) work on lists of binaries, bit streams on binaries
with one bit per byte, as GNU Radio's unpacked streams.

## Coverage

Status: done (equal to GNU Radio on the cases, or within a stated bound for
samples), partial, skipped. Hardware class as in `NxFixed.Block`.

| GNU Radio | ExSdr.Digital | Status | Hardware |
|---|---|---|---|
| `constellation_bpsk`, `_qpsk`, `_dqpsk`, `_8psk`, `_8psk_natural`, `_16qam` | `Constellation.bpsk/0`, `qpsk/0`, `dqpsk/0`, `psk8/0`, `psk8_natural/0`, `qam16/0` | done: points, pre_diff_code, apply flag, symmetry equal | data |
| `constellation_rect`, `_psk`, `_sector` (point lists) | `Constellation.new/2` (normalisations `:amplitude`, `:power`, `:none`) | partial: points equal, decision is always nearest point | data |
| `psk_constellation`, `qam_constellation` (Python) | `Constellation.psk/3`, `qam/3` | done, equal for the listed sizes and codes | data |
| `constellation_expl_rect`, `large_ampls_to_corners` | none | skipped: sector tables for sync loops | |
| `chunks_to_symbols_bf`, `_bc` | `:chunks_to_symbols_bf`, `:chunks_to_symbols_bc` | done (dimension 1, `u8` symbols); fixed point | stateless |
| `chunks_to_symbols_sf`, `_sc`, `_if`, `_ic` | none | skipped: same kernel with wider input items | |
| `constellation_decoder_cb` | `:constellation_decoder_cb` | done; fixed point | stateless |
| `constellation_soft_decoder_cf` | `:constellation_soft_decoder_cf` | done for `npwr = 1`; float only | stateless |
| `map_bb` | `:map_bb` | done | stateless |
| `diff_encoder_bb`, `diff_decoder_bb` | `:diff_encoder_bb`, `:diff_decoder_bb` | done | feedback, feed forward |
| `diff_phasor_cc` | `:diff_phasor_cc` | done; fixed point | feed forward |
| `binary_slicer_fb` | `:binary_slicer_fb` | done; fixed point | stateless |
| `scrambler_bb`, `descrambler_bb` | `:scrambler_bb`, `:descrambler_bb` | done | feedback |
| `additive_scrambler_bb` | `:additive_scrambler_bb` | partial: `count` and `bits_per_byte` yes, reset tag no | feedback |
| `additive_scrambler_cc/ff/ii/ss` | none | skipped: sign flip variants, not asked | |
| `crc` | `ExSdr.Digital.CRC` (any width 1-64) | done, equal for 8, 16, 32, 64 bits | |
| `crc32_bb` | `:crc32_bb` | done, packed and unpacked, append and check | control |
| `crc_append`, `crc_check` | `:crc_append`, `:crc_check` | done | control |
| `crc32_async_bb`, `crc16_async_bb` | `:crc32_async_bb`, `:crc16_async_bb` | done | control |
| `correlate_access_code_bb` | `:correlate_access_code_bb` | done | control |
| `correlate_access_code_tag_bb` | `:correlate_access_code_tag_bb` | done (tags as `{offset, wrong bits}`) | control |
| `correlate_access_code_ff_ts`, `_tag_ff`, `_bb_ts` | none | skipped: float input and tagged stream framing | |
| `header_format_default`, `_counter`, `_crc` via `protocol_formatter_bb` | `:protocol_formatter_bb` (`kind: :default`, `:counter`, `:crc`) | done (formatting; parsing only `Framing.parse_length/1`) | control |
| `header_format_ofdm`, `packet_headergenerator_bb`, `protocol_parser_b`, `packet_headerparser_b` | none | skipped | |
| `hdlc_framer_pb` | `:hdlc_framer_pb` | done | control |
| `hdlc_deframer_bp` | `:hdlc_deframer_bp` | done | control |
| `generic_mod` | `:generic_mod` | done (no `truncate`) | feedback |
| `psk_mod` | `:psk_mod` | done | feedback |
| `qam_mod` (no such class in 3.10: `generic_mod` with `qam_constellation`) | `:qam_mod` | done | feedback |
| `gfsk_mod` | `:gfsk_mod` | done | feedback |
| `gmsk_mod` | `:gmsk_mod` | done | feedback |
| `interp_fir_filter_fff`, `_ccf` (pulse shaping) | `:interp_fir_filter_fff`, `:interp_fir_filter_ccf` | done, local small version; fixed point | feed forward |
| `analog.frequency_modulator_fc` | `:frequency_modulator_fc` | done; fixed point | feedback |
| `firdes.root_raised_cosine`, `firdes.gaussian` | `Modulation.rrc_taps/5`, `gaussian_taps/4` (and the pulses `rrc_pulse/3`, `gaussian_pulse/2`) | done, equal to float32 precision | |
| `blocks.multiply_const_cc` | `:multiply_const_cc` | local helper for `gfsk_mod`'s output gain; fixed point | stateless |
| `glfsr_source_b/f`, `cpmmod_bc`, `gmskmod_bc`, `burst_shaper`, `mpsk_snr_est`, `ofdm_*` | none | skipped: sources, CPM, tagged bursts, estimators, OFDM belong elsewhere | |

## Checks against GNU Radio

Stream blocks are run by `test/dsp_cases/digital/*.json` through
`test/gnuradio/digital_golden.py` (which drives `golden.py` but evaluates
GNU Radio object arguments such as `digital.constellation_qpsk().base()`).
The rest (tagged streams, PDUs, constellation objects, taps) is produced by
`test/gnuradio/digital_framing_golden.py` into
`test/fixtures/gnuradio/digital/framing.json`. Regenerating, in the podman
image (no sudo):

    podman run --rm -v "$PWD/test:/work/test:ro,Z" \
      -v "$PWD/test/dsp_cases/digital:/cases:ro,Z" \
      -v "$PWD/test/fixtures/gnuradio/digital:/out:Z" -w /work exhdl-gnuradio \
      python3 test/gnuradio/digital_golden.py /cases /out
    podman run --rm -v "$PWD/test:/work/test:ro,Z" \
      -v "$PWD/test/fixtures/gnuradio/digital:/out:Z" -w /work exhdl-gnuradio \
      python3 test/gnuradio/digital_framing_golden.py /out/framing.json

Bit and byte results are equal to GNU Radio's, including the scramblers with
reset, HDLC on noisy streams and all packet formats. Sample results have the
bound and its reasoning in each case file. What was measured, float
instance against GNU Radio (largest error, complex magnitude where
complex): chunks_to_symbols 3.5e-7, soft decoder 1e-6, diff_phasor 3e-8,
interpolating FIR 4e-7, `frequency_modulator_fc` 6e-6, `psk_mod` and
`generic_mod` 8e-7, `gfsk_mod` and `gmsk_mod` 2e-5.

Fixed-point instances (`format: NxFixed.Num.q(bits, frac, ...)`) go through
the width check every time they run; they are compared with the float
instance with the bounds from their format (tap and constant rounding, the
sine table of the modulator). Tests with the stream cut in pieces check that
the state is complete.

## Deviations and conventions worth knowing

* GNU Radio's QPSK and DQPSK constellations are not at unit power: the
  points are +-sqrt(2) on each axis (power 4), BPSK and 8-PSK have unit
  amplitude, 16-QAM unit power. The soft decoder uses these raw points, so
  its log-likelihood ratios scale with them. `Constellation.new/2` normalises
  to mean amplitude 1 like `constellation_rect` does.
* The hard decision is the nearest point, ties to the smaller index.
  `constellation_decoder_cb` returns that index, not `pre_diff_code[index]`
  (GNU Radio does the same, the code is applied by `map_bb` in the
  modulator). The soft decoder labels the bits of a point with the index, or
  with `pre_diff_code[index]` when the constellation applies the code
  (`dqpsk`), as GNU Radio does.
* Soft decisions are the exact log-likelihood ratio
  `ln(sum_{bit=1} exp(-d^2/npwr) / sum_{bit=0} exp(-d^2/npwr))`. That is
  GNU Radio's result without a lookup table (`npwr` 1). With `npwr > 0`
  GNU Radio regenerates a quantised lookup table; ours stays exact, so it can
  differ from GNU Radio there by the table's quantisation.
* Symbols outside a symbol table are clipped to the last entry (GNU Radio
  reads out of bounds).
* `diff_decoder_bb` and `diff_encoder_bb` are modular for any byte input
  (`mod modulus`); GNU Radio only defines inputs below the modulus.
* Frames. `hdlc_framer_pb` does not stuff a 0 after five 1s that end the
  data (the closing flag follows), exactly as GNU Radio, so strictly
  speaking such frames are not standard HDLC; the deframer accepts both
  forms. `length_min` and `length_max` of `hdlc_deframer_bp` count the
  payload and the two CRC bytes. GNU Radio aborts the process for
  `length_min` 0; ours accepts it. GNU Radio's scheduler only emits what its
  buffers flushed, the tests append zero bits to the stream like the
  reference run does.
* `crc32_bb` unpacked packs bits into bytes least significant bit first and
  appends the CRC as 32 bits, least significant bit first; packed it appends
  the CRC little endian. `crc_check` strips the CRC from failing PDUs too
  when `discard_crc` is set. `crc_append` writes the CRC big endian unless
  `swap_endianness`.
* `header_format_default` writes only the last `8 * floor(len / 8)` bits of
  an access code that is not a whole number of bytes long (GNU Radio's
  header buffer does), the length fields 16 bits twice; the counter variant
  adds `bps` and the packet number. `header_format_crc` is length (12 bits),
  packet number (12 bits), both least significant bit first, and a CRC-8
  (polynomial 7, init 0xFF) over both as two 16-bit little-endian words.
* The modulators take `packed bytes`, symbols are cut from the bit stream
  most significant bit first (also 3 bits per symbol, a partial symbol is
  kept in the state). The pulse is an interpolating FIR (`:taps`). For
  `generic_mod`, GNU Radio uses a polyphase arbitrary resampler over 32
  arms, which for integer `samples_per_symbol` dividing 32 is the same FIR
  with every `32 / sps`-th tap; `Modulation.rrc_pulse/3` gives them. The
  pulse peaks above 1 (gain 32 prototype): fixed-point formats need an
  integer bit (Q2.14). `gfsk_mod` has GNU Radio's output gain 0.999,
  `generic_mod` none (it multiplies the taps if given `:gain`).
* The fixed-point frequency modulator keeps the phase in 32 bits and reads a
  sine table with the top 16: the `lut_bits` of the format set the accuracy
  (1024 entries: 3e-3, 4096 entries: 8e-4 against the float instance).
