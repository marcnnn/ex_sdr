#!/usr/bin/env python3
"""GNU Radio reference data for the digital family's framing blocks, which
golden.py's plain stream chain cannot drive: tagged-stream blocks
(crc32_bb, protocol_formatter_bb), PDU blocks (crc_append, crc_check,
hdlc_framer_pb, hdlc_deframer_bp) and the access code correlators.

Usage: digital_framing_golden.py <output file>

Writes one JSON file with the inputs (hex strings for bytes, 0/1 strings
for bits) and what GNU Radio produced. Run it in the podman image like
golden.py:

  podman run --rm -v "$PWD/test:/work/test:ro,Z" \
    -v "$PWD/test/fixtures/gnuradio/digital:/out:Z" -w /work exhdl-gnuradio \
    python3 test/gnuradio/digital_framing_golden.py /out/framing.json
"""
import json
import sys
import time

import numpy as np
import pmt
from gnuradio import blocks, digital, gr

rng = np.random.default_rng(2024)


def rand_bytes(n):
    return bytes(rng.integers(0, 256, n).tolist())


def hexs(b):
    return bytes(b).hex()


def bitstr(bits):
    return "".join(str(int(b) & 1) for b in bits)


def tag(offset, key, value):
    t = gr.tag_t()
    t.offset = offset
    t.key = pmt.intern(key)
    t.value = pmt.from_long(value)
    return t


def run_tagged(blk, packets, key="packet_len"):
    data, tags, off = [], [], 0
    for p in packets:
        tags.append(tag(off, key, len(p)))
        data += list(p)
        off += len(p)
    src = blocks.vector_source_b(data, False, 1, tags)
    snk = blocks.vector_sink_b()
    tb = gr.top_block()
    tb.connect(src, blk, snk)
    tb.run()
    out = list(snk.data())
    lens = [pmt.to_python(t.value) for t in snk.tags() if pmt.symbol_to_string(t.key) == key]
    res, pos = [], 0
    for n in lens:
        res.append(out[pos : pos + n])
        pos += n
    return res


def run_pdu(blk, in_port, out_ports, pdus, wait=0.4):
    dbgs = {}
    tb = gr.top_block()
    for p in out_ports:
        d = blocks.message_debug()
        dbgs[p] = d
        tb.msg_connect(blk, p, d, "store")
    tb.start()
    for data in pdus:
        blk.to_basic_block()._post(
            pmt.intern(in_port), pmt.cons(pmt.make_dict(), pmt.init_u8vector(len(data), list(data)))
        )
    time.sleep(wait)
    tb.stop()
    tb.wait()
    out = {}
    for p, d in dbgs.items():
        out[p] = [
            bytes(pmt.u8vector_elements(pmt.cdr(d.get_message(i)))) for i in range(d.num_messages())
        ]
    return out


def crc_cases():
    cases = []
    params = [
        (8, 0x07, 0x00, 0x00, False, False),
        (8, 0x31, 0xFF, 0x00, True, True),
        (16, 0x1021, 0xFFFF, 0x0000, False, False),
        (16, 0x1021, 0x0000, 0x0000, False, False),
        (16, 0x1021, 0xFFFF, 0xFFFF, True, True),
        (16, 0x8005, 0x0000, 0x0000, True, True),
        (16, 0x8005, 0xFFFF, 0xFFFF, False, True),
        (32, 0x04C11DB7, 0xFFFFFFFF, 0xFFFFFFFF, True, True),
        (32, 0x04C11DB7, 0xFFFFFFFF, 0x00000000, False, False),
        (32, 0x1EDC6F41, 0xFFFFFFFF, 0xFFFFFFFF, True, True),
        (64, 0x42F0E1EBA9EA3693, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, True, True),
    ]
    msgs = [b"123456789", b"", rand_bytes(1), rand_bytes(37), rand_bytes(300)]
    for p in params:
        c = digital.crc(*p)
        cases.append(
            {
                "params": list(p[:4]) + [bool(p[4]), bool(p[5])],
                "messages": [
                    {"data": hexs(m), "crc": c.compute(list(m))} for m in msgs
                ],
            }
        )
    return cases


def crc32_bb_cases():
    out = []
    pk = [rand_bytes(n) for n in (1, 11, 40, 255, 6)]
    packed = run_tagged(digital.crc32_bb(False, "packet_len", True), [list(p) for p in pk])
    out.append({"check": False, "packed": True, "in": [hexs(p) for p in pk], "out": [hexs(p) for p in packed]})
    # check: good packets, one corrupted, one too short
    good = [bytes(p) for p in packed]
    bad = bytearray(good[1])
    bad[3] ^= 0x10
    inp = [good[0], bytes(bad), good[2], good[3]]
    chk = run_tagged(digital.crc32_bb(True, "packet_len", True), [list(p) for p in inp])
    out.append({"check": True, "packed": True, "in": [hexs(p) for p in inp], "out": [hexs(p) for p in chk]})
    # unpacked: bits, 8 per byte, LSB first
    def to_bits(b):
        return [(x >> k) & 1 for x in b for k in range(8)]
    bitpk = [to_bits(p) for p in pk[:3]]
    unp = run_tagged(digital.crc32_bb(False, "packet_len", False), bitpk)
    out.append({"check": False, "packed": False, "in": [bitstr(p) for p in bitpk], "out": [bitstr(p) for p in unp]})
    inp = [unp[0], unp[1]]
    inp[1] = list(inp[1])
    inp[1][5] ^= 1
    chk = run_tagged(digital.crc32_bb(True, "packet_len", False), inp)
    out.append({"check": True, "packed": False, "in": [bitstr(p) for p in inp], "out": [bitstr(p) for p in chk]})
    return out


def crc_pdu_cases():
    out = []
    configs = [
        (32, 0x04C11DB7, 0xFFFFFFFF, 0xFFFFFFFF, True, True, True, 0),
        (32, 0x04C11DB7, 0xFFFFFFFF, 0xFFFFFFFF, True, True, False, 0),
        (16, 0x1021, 0xFFFF, 0x0000, False, False, False, 0),
        (16, 0x1021, 0xFFFF, 0x0000, False, False, True, 3),
        (8, 0x07, 0x00, 0x00, False, False, False, 0),
        (64, 0x42F0E1EBA9EA3693, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, True, True, False, 0),
    ]
    pdus = [rand_bytes(n) for n in (5, 12, 33, 100)]
    for cfg in configs:
        a = run_pdu(digital.crc_append(*cfg), "in", ["out"], pdus)["out"]
        # check: the appended ones (ok), plus corrupted copies (fail)
        corrupt = []
        for p in a[:2]:
            q = bytearray(p)
            q[len(q) // 2] ^= 0x01
            corrupt.append(bytes(q))
        entry = {
            "config": [cfg[0], cfg[1], cfg[2], cfg[3], bool(cfg[4]), bool(cfg[5]), bool(cfg[6]), cfg[7]],
            "in": [hexs(p) for p in pdus],
            "appended": [hexs(p) for p in a],
            "check": [],
        }
        for discard in (True, False):
            chk = digital.crc_check(*(cfg[:7] + (discard,) + cfg[7:]))
            order = [a[0], corrupt[0], a[3], corrupt[1]]
            r = run_pdu(chk, "in", ["ok", "fail"], order)
            entry["check"].append(
                {
                    "discard_crc": discard,
                    "in": [hexs(p) for p in order],
                    "ok": [hexs(p) for p in r["ok"]],
                    "fail": [hexs(p) for p in r["fail"]],
                }
            )
        out.append(entry)
    return out


def async_crc_cases():
    out = []
    pdus = [rand_bytes(n) for n in (5, 12, 33)]
    for name in ("crc32_async_bb", "crc16_async_bb"):
        for check in (False,):
            blk = getattr(digital, name)(check)
            r = run_pdu(blk, "in", ["out"], pdus)["out"]
            out.append({"block": name, "check": check, "in": [hexs(p) for p in pdus], "out": [hexs(p) for p in r]})
    return out


def run_bits(blk, bits):
    src = blocks.vector_source_b([int(x) for x in bits], False)
    snk = blocks.vector_sink_b()
    tb = gr.top_block()
    tb.connect(src, blk, snk)
    tb.run()
    return list(snk.data()), [
        (t.offset, pmt.to_python(t.value)) for t in snk.tags()
    ]


def correlate_cases():
    out = []
    for ac, thr in [("1011010001110010", 0), ("1011010001110010", 2), ("0001" * 8, 3), ("101", 0), ("1" * 64, 5)]:
        code = [int(c) for c in ac]
        bits = list(rng.integers(0, 2, 70))
        bits += code
        bits += list(rng.integers(0, 2, 50))
        noisy = list(code)
        noisy[1] ^= 1
        if len(noisy) > 6:
            noisy[5] ^= 1
        bits += noisy
        bits += list(rng.integers(0, 2, 90)) + code + list(rng.integers(0, 2, 150))
        bits = [int(b) for b in bits]
        o, _ = run_bits(digital.correlate_access_code_bb(ac, thr), bits)
        o2, tags = run_bits(digital.correlate_access_code_tag_bb(ac, thr, "sync"), bits)
        out.append(
            {
                "access_code": ac,
                "threshold": thr,
                "in": bitstr(bits),
                "bb": "".join(str(v) for v in o),
                "tag_out": bitstr(o2),
                "tags": [[int(off), int(v)] for off, v in tags],
            }
        )
    return out


def formatter_cases():
    out = []
    for kind, ctor in (("default", digital.header_format_default), ("counter", digital.header_format_counter)):
        for ac, thr, bps in [("1010101011110000", 2, 1), ("101011110000", 1, 2), ("1010101011110000101", 0, 4), ("0" * 64, 0, 1), ("1", 0, 1)]:
            f = ctor(ac, thr, bps)
            lens = [1, 10, 255, 256, 1500, 3]
            hdrs = run_tagged(digital.protocol_formatter_bb(f, "packet_len"), [[7] * n for n in lens])
            out.append(
                {
                    "kind": kind,
                    "access_code": ac,
                    "threshold": thr,
                    "bps": bps,
                    "payload_lengths": lens,
                    "headers": [hexs(h) for h in hdrs],
                }
            )
    return out


def hdlc_cases():
    pkts = [rand_bytes(n) for n in (1, 2, 5, 10, 29, 30, 31, 32, 33, 38, 39, 40, 41, 42, 64, 200)]
    pkts += [b"\xff" * 12, b"\x00" * 9, b"\x7e\x7d\x7e" * 4, bytes([0xF8, 0x1F, 0xFF, 0x3F, 0xF8])]
    fr = digital.hdlc_framer_pb("frame")
    snk = blocks.vector_sink_b()
    tb = gr.top_block()
    tb.connect(fr, snk)
    tb.start()
    for p in pkts:
        fr.to_basic_block()._post(
            pmt.intern("in"), pmt.cons(pmt.make_dict(), pmt.init_u8vector(len(p), list(p)))
        )
    time.sleep(0.6)
    tb.stop()
    tb.wait()
    data = list(snk.data())
    lens = [pmt.to_python(t.value) for t in snk.tags()]
    frames, pos = [], 0
    for n in lens:
        frames.append(data[pos : pos + n])
        pos += n
    framer = {"payloads": [hexs(p) for p in pkts], "frames": [bitstr(f) for f in frames]}

    def deframe(bits, lmin, lmax):
        de = digital.hdlc_deframer_bp(lmin, lmax)
        dbg = blocks.message_debug()
        src = blocks.vector_source_b([int(x) for x in bits] + [0] * 4000, False)
        tb = gr.top_block()
        tb.connect(src, de)
        tb.msg_connect(de, "out", dbg, "store")
        tb.run()
        return [hexs(bytes(pmt.u8vector_elements(pmt.cdr(dbg.get_message(i))))) for i in range(dbg.num_messages())]

    deframer = []
    stream = [b for f in frames for b in f]
    # the same stream with noise between frames, corrupted frames, shared flags
    noise = [int(b) for b in rng.integers(0, 2, 200)]
    corrupted = [list(f) for f in frames]
    corrupted[3][20] ^= 1
    corrupted[9][40] ^= 1
    shared = list(frames[0])
    for f in frames[1:]:
        shared += f[8:]  # drop the second flag's duplicate: one flag between frames
    streams = {
        "plain": stream,
        "noise": noise + stream[:300] + noise + stream[300:],
        "corrupted": [b for f in corrupted for b in f],
        "shared_flags": shared,
        "abort": frames[3][:50] + [1] * 9 + frames[7],
        "idle_flags": [0, 1, 1, 1, 1, 1, 1, 0] * 3 + stream,
    }
    for name, bits in streams.items():
        for lmin, lmax in [(32, 500), (1, 500), (10, 40), (3, 100)]:
            deframer.append(
                {
                    "stream": name,
                    "bits": bitstr(bits),
                    "length_min": lmin,
                    "length_max": lmax,
                    "frames": deframe(bits, lmin, lmax),
                }
            )
    return {"framer": framer, "deframer": deframer}


def main(path, only=None):
    sections = {
        "crc": crc_cases,
        "crc32_bb": crc32_bb_cases,
        "crc_pdu": crc_pdu_cases,
        "crc_async": async_crc_cases,
        "correlate": correlate_cases,
        "formatter": formatter_cases,
        "hdlc": hdlc_cases,
    }
    out = {"gnuradio": gr.version()}
    for name, fun in sections.items():
        if only is None or name in only:
            print(name, flush=True)
            out[name] = fun()
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or None)
