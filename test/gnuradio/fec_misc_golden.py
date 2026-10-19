#!/usr/bin/env python3
"""GNU Radio reference data for the small gr-fec blocks and the generic
framework: repetition and dummy codes, puncture_bb / puncture_ff /
depuncture_bb, ber_bf, and the tagged stream and PDU (async) wrappers of the
encoders and decoders.

Usage: fec_misc_golden.py <output file>

  podman run --rm -v "$PWD/test:/work/test:ro,Z" \
    -v "$PWD/test/fixtures/gnuradio/fec:/out:Z" -w /work exhdl-gnuradio \
    python3 test/gnuradio/fec_misc_golden.py /out/misc.json
"""
import json
import os
import sys

import numpy as np
import pmt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fec_common import bitstr, encode_frames, hexs, run_chain, run_pdu, run_tagged, work  # noqa: E402
from gnuradio import blocks, fec, gr  # noqa: E402

rng = np.random.default_rng(5)


def f32hex(a):
    return np.asarray(a, dtype="<f4").tobytes().hex()


def repetition_cases():
    out = []
    for f, rep, ap in [(16, 3, 0.5), (10, 4, 0.5), (24, 5, 0.3), (12, 2, 0.5), (20, 6, 0.75), (8, 1, 0.5)]:
        enc = fec.repetition_encoder_make(f, rep)
        dec = fec.repetition_decoder_make(f, rep, ap)
        frames = [rng.integers(0, 2, f).astype(np.uint8) for _ in range(3)]
        encs = encode_frames(enc, frames)
        # decoder input: noisy repetitions, plus exact zeros and ties
        ins, outs = [], []
        for e in encs:
            soft = (e.astype(float) * 2 - 1) + rng.normal(0, 1.1, len(e))
            soft[rng.integers(0, len(e), 3)] = 0.0
            soft = soft.astype(np.float32)
            o = np.zeros(f, dtype=np.uint8)
            work(dec, soft, o)
            ins.append(f32hex(soft))
            outs.append(bitstr(o))
        out.append({"frame_size": f, "rep": rep, "ap_prob": ap, "enc_input_size": enc.get_input_size(),
                    "enc_output_size": enc.get_output_size(), "enc_rate": enc.rate(),
                    "dec_input_size": dec.get_input_size(), "dec_output_size": dec.get_output_size(),
                    "dec_rate": dec.rate(),
                    "frames": [bitstr(x) for x in frames], "encoded": [bitstr(x) for x in encs],
                    "soft": ins, "decoded": outs})
    return out


def dummy_cases():
    out = []
    for pack, packed_bits in [(False, False), (True, False), (False, True), (True, True)]:
        enc = fec.dummy_encoder_make(24, pack, packed_bits)
        frame = rng.integers(0, 256, 24).astype(np.uint8)
        o = np.zeros(24, dtype=np.uint8)
        work(enc, frame, o)
        out.append({"pack": pack, "packed_bits": packed_bits, "frame_size": 24,
                    "input_size": enc.get_input_size(), "output_size": enc.get_output_size(),
                    "input_conversion": enc.get_input_conversion(),
                    "output_conversion": enc.get_output_conversion(), "rate": enc.rate(),
                    "input": hexs(frame), "output": hexs(o)})
    dec = fec.dummy_decoder_make(16)
    soft = np.array([0.0, -0.0, 1e-30, -1e-30, 1.0, -1.0, 3.5, -3.5, 0.5, -0.25, 7.0, -7.0, 0.1, -0.1, 2.0, -2.0],
                    dtype=np.float32)
    o = np.zeros(16, dtype=np.int8)
    work(dec, soft, o)
    dec_case = {"input": f32hex(soft), "output": [int(v) for v in o],
                "input_conversion": dec.get_input_conversion(), "rate": dec.rate()}
    return out, dec_case


def puncture_cases():
    out = []
    pats = [(8, 0xEF, 0), (8, 0xEF, 3), (4, 0b1011, 1), (6, 0b110101, 3), (3, 0b101, 0), (4, 0b0110, 7),
            (5, 0b10111, 2), (4, 0b11111, 1), (7, 0b1101011, 6), (8, 0b11010110, 8)]
    for size, pat, delay in pats:
        kept = bin(pat & ((1 << size) - 1)).count("1")
        # the delay rotates the unmasked pattern: let GR tell the kept count
        groups = 6
        data = rng.integers(0, 256, size * groups).astype(np.uint8)
        pb = run_chain([fec.puncture_bb(size, pat, delay)], data.tolist(), "b", "b")
        fl = rng.normal(0, 1, size * groups).astype(np.float32)
        pf = run_chain([fec.puncture_ff(size, pat, delay)], fl.tolist(), "f", "f")
        kept = len(pb) // groups
        # depuncture: feed what the puncturer kept
        sym_default = run_chain([fec.depuncture_bb(size, pat, delay)], pb, "b", "b")
        sym_custom = run_chain([fec.depuncture_bb(size, pat, delay, 55)], pb, "b", "b")
        out.append({"puncsize": size, "puncpat": pat, "delay": delay, "kept": kept,
                    "data": hexs(data), "punctured": hexs(pb), "float_in": f32hex(fl),
                    "float_out": f32hex(pf), "depunctured": hexs(sym_default),
                    "depunctured_55": hexs(sym_custom)})
    return out


def ber_cases():
    out = []
    for n, flip in [(100, 0.05), (400, 0.2), (64, 0.5)]:
        a = rng.integers(0, 256, n).astype(np.uint8)
        mask = np.where(rng.random(n) < flip, rng.integers(1, 256, n), 0).astype(np.uint8)
        b = a ^ mask
        src_a = blocks.vector_source_b(a.tolist(), False)
        src_b = blocks.vector_source_b(b.tolist(), False)
        ber = fec.ber_bf(False, 100, -7.0)
        snk = blocks.vector_sink_f()
        tb = gr.top_block()
        tb.connect(src_a, (ber, 0))
        tb.connect(src_b, (ber, 1))
        tb.connect(ber, snk)
        tb.run()
        res = list(snk.data())
        errors = int(sum(bin(int(x)).count("1") for x in mask))
        out.append({"a": hexs(a), "b": hexs(b), "errors": errors, "last": float(res[-1]),
                    "count": len(res)})
    return out


def tagged_cases():
    out = []
    mtu = 16
    # terminated, tailbiting and truncated convolutional codes, repetition
    objs = [
        ("cc_terminated", lambda: fec.cc_encoder_make(mtu * 8, 7, 2, [79, 109], 0, fec.CC_TERMINATED, False),
         lambda: fec.cc_decoder.make(mtu * 8, 7, 2, [79, 109], 0, -1, fec.CC_TERMINATED, False)),
        ("cc_terminated_padded",
         lambda: fec.cc_encoder_make(mtu * 8, 7, 2, [79, 109], 0, fec.CC_TERMINATED, True),
         lambda: fec.cc_decoder.make(mtu * 8, 7, 2, [79, 109], 0, -1, fec.CC_TERMINATED, True)),
        ("cc_truncated", lambda: fec.cc_encoder_make(mtu * 8, 7, 2, [79, 109], 0, fec.CC_TRUNCATED, False),
         lambda: fec.cc_decoder.make(mtu * 8, 7, 2, [79, 109], 0, -1, fec.CC_TRUNCATED, False)),
        ("cc_tailbiting", lambda: fec.cc_encoder_make(mtu * 8, 7, 2, [79, 109], 0, fec.CC_TAILBITING, False),
         lambda: fec.cc_decoder.make(mtu * 8, 7, 2, [79, 109], 0, -1, fec.CC_TAILBITING, False)),
        ("repetition", lambda: fec.repetition_encoder_make(mtu * 8, 3),
         lambda: fec.repetition_decoder_make(mtu * 8, 3, 0.5)),
    ]
    for name, mk_enc, mk_dec in objs:
        lens = [40, 64, 100] if name != "cc_tailbiting" else [48, 64, 100]
        packets = [rng.integers(0, 2, n).astype(np.uint8) for n in lens]
        enc = fec.tagged_encoder(mk_enc(), 1, 1, "packet_len", mtu)
        encoded = run_tagged(enc, [p.tolist() for p in packets])
        rec = {"name": name, "mtu": mtu, "lengths": lens,
               "packets": [bitstr(p) for p in packets],
               "encoded": [bitstr(e) for e in encoded]}
        if name.startswith("cc"):
            soft = [np.clip(np.rint(np.array(e) * 255.0 + rng.normal(0, 70, len(e))), 0, 255).astype(np.uint8)
                    for e in encoded]
            dec = fec.tagged_decoder(mk_dec(), 1, 1, "packet_len", mtu)
            decoded = run_tagged(dec, [s.tolist() for s in soft])
            rec["soft"] = [hexs(s) for s in soft]
        else:
            soft = [((np.array(e) * 2 - 1) + rng.normal(0, 1.0, len(e))).astype(np.float32) for e in encoded]
            dec = fec.tagged_decoder(mk_dec(), 4, 1, "packet_len", mtu)
            decoded = run_tagged(dec, [s.tolist() for s in soft], src="f", snk="b")
            rec["soft"] = [f32hex(s) for s in soft]
        rec["decoded"] = [bitstr(d) for d in decoded]
        out.append(rec)
    return out


def u8vec(x):
    return pmt.init_u8vector(len(x), x)


def f32vec(x):
    return pmt.init_f32vector(len(x), x)


def async_cases():
    out = []
    mtu = 16
    makers = [
        ("cc_terminated_padded",
         lambda: fec.cc_encoder_make(mtu * 8, 7, 2, [79, 109], 0, fec.CC_TERMINATED, True),
         lambda: fec.cc_decoder.make(mtu * 8, 7, 2, [79, 109], 0, -1, fec.CC_TERMINATED, True)),
        ("cc_terminated", lambda: fec.cc_encoder_make(mtu * 8, 7, 2, [79, 109], 0, fec.CC_TERMINATED, False),
         lambda: fec.cc_decoder.make(mtu * 8, 7, 2, [79, 109], 0, -1, fec.CC_TERMINATED, False)),
        ("repetition", lambda: fec.repetition_encoder_make(mtu * 8, 3),
         lambda: fec.repetition_decoder_make(mtu * 8, 3, 0.5)),
    ]
    for name, mk_enc, mk_dec in makers:
        for packed, rev_unpack, rev_pack in [(False, True, True), (True, True, True), (True, False, False),
                                             (True, False, True)]:
            if packed:
                pdus = [rng.integers(0, 256, n).astype(np.uint8) for n in (5, 9, 12)]
            else:
                pdus = [rng.integers(0, 2, n).astype(np.uint8) for n in (40, 64, 100)]
            enc = fec.async_encoder(mk_enc(), packed, rev_unpack, rev_pack, mtu)
            res = run_pdu(enc, [p.tolist() for p in pdus], u8vec)["out"]
            out.append({"kind": "encoder", "code": name, "packed": packed, "rev_unpack": rev_unpack,
                        "rev_pack": rev_pack, "mtu": mtu, "pdus": [hexs(p) for p in pdus],
                        "out": [hexs(r) for r in res]})
        # decoders: soft input from the unpacked encoder output
        enc = fec.async_encoder(mk_enc(), False, True, True, mtu)
        bits = [rng.integers(0, 2, n).astype(np.uint8) for n in (40, 64, 96)]
        coded = run_pdu(enc, [b.tolist() for b in bits], u8vec)["out"]
        soft = [((np.array(c) * 2 - 1) + rng.normal(0, 0.45, len(c))).astype(np.float32) for c in coded]
        for packed, rev_pack in [(False, True), (True, True), (True, False)]:
            if packed and name == "cc_terminated":
                bits_used = bits[:1] + bits[1:]
            dec = fec.async_decoder(mk_dec(), packed, rev_pack, mtu)
            try:
                res = run_pdu(dec, [s.tolist() for s in soft], f32vec)["out"]
            except Exception as e:  # pragma: no cover
                res = []
            out.append({"kind": "decoder", "code": name, "packed": packed, "rev_pack": rev_pack,
                        "mtu": mtu, "soft": [f32hex(s) for s in soft],
                        "out": [hexs(r) for r in res]})
    return out


if __name__ == "__main__":
    dummy_enc, dummy_dec = dummy_cases()
    data = {"repetition": repetition_cases(), "dummy_encoders": dummy_enc, "dummy_decoder": dummy_dec,
            "puncture": puncture_cases(), "ber": ber_cases(), "tagged": tagged_cases(),
            "async": async_cases()}
    with open(sys.argv[1], "w") as fh:
        json.dump(data, fh)
