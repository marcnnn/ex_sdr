#!/usr/bin/env python3
"""GNU Radio reference data for gr-fec's polar codes: the encoders
(polar_encoder packed and unpacked, polar_encoder_systematic), the decoders
(polar_decoder_sc, polar_decoder_sc_list, polar_decoder_sc_systematic) and
the frozen bit constructions of fec.polar.channel_construction.

Usage: fec_polar_golden.py <output file>

  podman run --rm -v "$PWD/test:/work/test:ro,Z" \
    -v "$PWD/test/fixtures/gnuradio/fec:/out:Z" -w /work exhdl-gnuradio \
    python3 test/gnuradio/fec_polar_golden.py /out/polar.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fec_common import bitstr, hexs, work  # noqa: E402
from gnuradio import fec  # noqa: E402
from gnuradio.fec.polar import channel_construction as cc  # noqa: E402
from gnuradio.fec.polar.channel_construction_bec import bhattacharyya_bounds  # noqa: E402
from gnuradio.fec.polar.channel_construction_bec import calculate_bec_channel_capacities  # noqa: E402
from gnuradio.fec.polar.channel_construction_bec import design_snr_to_bec_eta  # noqa: E402

rng = np.random.default_rng(11)


def constructions():
    out = []
    for n, k, snr in [(8, 4, 0.0), (16, 8, 0.0), (32, 16, 1.0), (64, 32, 0.0), (128, 64, 2.5),
                      (256, 128, 0.0), (256, 100, -1.0), (1024, 512, 0.0), (1024, 700, 3.0)]:
        pos = cc.frozen_bit_positions(n, k, snr)
        z = bhattacharyya_bounds(snr, n)
        cap = calculate_bec_channel_capacities(design_snr_to_bec_eta(snr), n)
        zpos = np.sort(np.argsort(-z, kind="stable")[: n - k])
        out.append({"n": n, "k": k, "design_snr": snr, "frozen": [int(p) for p in pos],
                    "z": [float(v) for v in z], "capacities": [float(v) for v in cap]})
    return out


def positions(n, k, snr=0.0):
    return [int(p) for p in cc.frozen_bit_positions(n, k, snr)]


def encoder_cases():
    cases = []
    for n, k, packed, systematic, vals in [
        (8, 4, False, False, False), (16, 8, False, False, False), (32, 16, False, False, True),
        (64, 32, False, False, True), (128, 64, False, False, False), (256, 128, False, False, True),
        (1024, 512, False, False, False), (16, 8, True, False, False), (64, 32, True, False, True),
        (256, 128, True, False, False), (1024, 512, True, False, True),
        (16, 8, False, True, False), (64, 40, False, True, False), (256, 128, False, True, False),
        (1024, 600, False, True, False),
    ]:
        pos = positions(n, k)
        fv = [int(v) for v in rng.integers(0, 2, n - k)] if vals else []
        if systematic:
            enc = fec.polar_encoder_systematic_make(n, k, pos)
        else:
            enc = fec.polar_encoder_make(n, k, pos, fv, packed)
        insize, outsize = enc.get_input_size(), enc.get_output_size()
        frames, outs = [], []
        for _ in range(3):
            if packed:
                fr = rng.integers(0, 256, insize).astype(np.uint8)
            else:
                fr = rng.integers(0, 2, insize).astype(np.uint8)
            o = np.zeros(outsize, dtype=np.uint8)
            work(enc, np.ascontiguousarray(fr), o)
            frames.append(fr)
            outs.append(o)
        cases.append({"n": n, "k": k, "packed": packed, "systematic": systematic, "frozen": pos,
                      "values": fv, "input_size": insize, "output_size": outsize,
                      "input_conversion": enc.get_input_conversion(),
                      "output_conversion": enc.get_output_conversion(), "rate": enc.rate(),
                      "inputs": [hexs(f) if packed else bitstr(f) for f in frames],
                      "outputs": [hexs(o) if packed else bitstr(o) for o in outs]})
    return cases


def f32hex(a):
    return np.asarray(a, dtype='<f4').tobytes().hex()


def noisy_codeword(x, sigma):
    s = (np.asarray(x, dtype=float) * 2 - 1) + rng.normal(0, sigma, len(x))
    return s.astype(np.float32)


def decoder_cases():
    cases = []
    plan = []
    for n, k in [(8, 4), (16, 8), (32, 16), (64, 32), (128, 64), (256, 128), (1024, 512)]:
        for sigma in (0.5, 0.9):
            plan.append(("sc", n, k, sigma, 0, False))
    plan += [("sc", 64, 32, 0.8, 0, True), ("sc", 256, 100, 0.7, 0, True)]
    for l in (1, 2, 4, 8, 16):
        for n, k, sigma in [(32, 16, 0.8), (128, 64, 0.8), (256, 128, 0.75)]:
            plan.append(("list", n, k, sigma, l, False))
    plan += [("list", 64, 32, 0.85, 4, True)]
    for n, k, sigma in [(16, 8, 0.6), (64, 32, 0.8), (256, 128, 0.8), (1024, 600, 0.7)]:
        plan.append(("systematic", n, k, sigma, 0, False))
    for kind, n, k, sigma, l, vals in plan:
        pos = positions(n, k)
        fv = [int(v) for v in rng.integers(0, 2, n - k)] if vals else []
        if kind == "sc":
            dec = fec.polar_decoder_sc.make(n, k, pos, fv)
            enc = fec.polar_encoder_make(n, k, pos, fv, False)
        elif kind == "list":
            dec = fec.polar_decoder_sc_list.make(l, n, k, pos, fv)
            enc = fec.polar_encoder_make(n, k, pos, fv, False)
        else:
            dec = fec.polar_decoder_sc_systematic.make(n, k, pos)
            enc = fec.polar_encoder_systematic_make(n, k, pos)
        nframes = (16 if n <= 128 else 8) if kind == 'list' else (2 if n >= 1024 else 4)
        sent, inputs, outputs = [], [], []
        for _ in range(nframes):
            info = rng.integers(0, 2, k).astype(np.uint8)
            x = np.zeros(n, dtype=np.uint8)
            work(enc, np.ascontiguousarray(info), x)
            soft = noisy_codeword(x, sigma)
            o = np.zeros(k, dtype=np.uint8)
            work(dec, np.ascontiguousarray(soft), o)
            sent.append(bitstr(info))
            inputs.append(f32hex(soft))
            outputs.append(bitstr(o))
        cases.append({"kind": kind, "n": n, "k": k, "sigma": sigma, "list": l, "frozen": pos,
                      "values": fv, "input_size": dec.get_input_size(),
                      "output_size": dec.get_output_size(), "rate": dec.rate(),
                      "sent": sent, "inputs": inputs, "outputs": outputs})
    # hard +-1 input and exact zeros: the sign conventions
    for kind in ("sc", "systematic"):
        n, k = 32, 16
        pos = positions(n, k)
        dec = (fec.polar_decoder_sc.make(n, k, pos, []) if kind == "sc"
               else fec.polar_decoder_sc_systematic.make(n, k, pos))
        soft = rng.choice([-1.0, 0.0, 1.0], n).astype(np.float32)
        o = np.zeros(k, dtype=np.uint8)
        work(dec, soft, o)
        cases.append({"kind": kind, "n": n, "k": k, "sigma": None, "list": 0, "frozen": pos,
                      "values": [], "input_size": n, "output_size": k, "rate": dec.rate(),
                      "sent": None, "inputs": [f32hex(soft)], "outputs": [bitstr(o)]})
    return cases


if __name__ == "__main__":
    data = {"constructions": constructions(), "encoders": encoder_cases(),
            "decoders": decoder_cases()}
    with open(sys.argv[1], "w") as fh:
        json.dump(data, fh)
