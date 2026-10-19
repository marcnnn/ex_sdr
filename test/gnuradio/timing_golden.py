#!/usr/bin/env python3
"""GNU Radio reference for the symbol timing recovery blocks (test/dsp_cases/timing/sync/*.json).

The shared runner (golden.py) only makes tones and noise; timing recovery needs
shaped symbol streams with a timing and sample rate offset. This script makes
those signals (seeded numpy), quantises them to Q15, runs each case's GNU Radio
block and writes, under <output dir>:

  signals/<signal>.s16le      the input, int16 (interleaved I/Q for complex)
  signals/<signal>.json       the transmitted symbols and generation parameters
  <case>/gnuradio.f32le       GNU Radio's output, float32 (interleaved I/Q)
  <case>/manifest.json        versions, SHA-256 of the case file and this
                              script, and the taps GNU Radio was given

Usage: timing_golden.py <cases dir> <output dir>. Run through
scripts/gnuradio_golden.sh for the timing family (golden.py HOOKS):

    scripts/gnuradio_golden.sh test/fixtures/gnuradio/timing test/dsp_cases/timing

A case file has "signal" (a name below), "gnuradio" (a Python expression
building the block; it sees gnuradio's digital and filter modules, numpy as
np, math, and "taps"), optionally "taps" (a firdes design, see make_taps) and
the ExSdr side ("block") and bounds used by test/timing/sync_test.exs.
"""
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np
from gnuradio import blocks, digital, gr
from gnuradio import filter as grfilter

FULL_SCALE = 32768.0

# name: (modulation, pulse, sps, timing offset in samples, sample rate offset, symbols, seed, real?)
SIGNALS = {
    "qpsk_rc_sps4": ("qpsk", "rc", 4.0, 0.37, 1.0e-4, 900, 101, False),
    "bpsk_rc_sps4_real": ("bpsk", "rc", 4.0, 0.61, -1.0e-4, 900, 102, True),
    "qpsk_rrc_sps4": ("qpsk", "rrc", 4.0, 0.29, 1.0e-4, 900, 103, False),
    "bpsk_rrc_sps4_real": ("bpsk", "rrc", 4.0, 0.83, -1.0e-4, 900, 104, True),
    "msk_sps4": ("msk", "msk", 4.0, 0.45, 1.0e-4, 900, 105, False),
    "qpsk_rc_sps3p3": ("qpsk", "rc", 3.3, 0.2, 0.0, 900, 106, False),
}

ALPHA = 0.35  # roll-off of the RC and RRC pulses
SPAN = 8  # pulse truncated to +-SPAN symbols
AMP = 0.45  # symbol amplitude per axis
NOISE = 0.01  # noise rms per component


def q15(x):
    return np.clip(np.round(np.asarray(x) * FULL_SCALE), -32768, 32767).astype(np.int16)


def rc(t):
    t = np.asarray(t, dtype=np.float64)
    out = np.sinc(t) * np.cos(np.pi * ALPHA * t)
    den = 1.0 - (2.0 * ALPHA * t) ** 2
    sing = np.abs(den) < 1e-10
    out = np.where(sing, np.pi / 4 * np.sinc(1 / (2 * ALPHA)), out / np.where(sing, 1.0, den))
    return out


def rrc(t):
    t = np.asarray(t, dtype=np.float64)
    a = ALPHA
    out = np.empty_like(t)
    zero = np.abs(t) < 1e-12
    sing = np.abs(np.abs(t) - 1 / (4 * a)) < 1e-10
    reg = ~(zero | sing)
    tr = t[reg]
    out[reg] = (np.sin(np.pi * tr * (1 - a)) + 4 * a * tr * np.cos(np.pi * tr * (1 + a))) / (
        np.pi * tr * (1 - (4 * a * tr) ** 2)
    )
    out[zero] = 1 - a + 4 * a / np.pi
    out[sing] = a / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * a)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * a)))
    return out


def make_signal(name):
    mod, pulse, sps, tau, eps, nsym, seed, real = SIGNALS[name]
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, (nsym, 2))
    if mod == "qpsk":
        sym = (2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)
    else:
        sym = (2 * bits[:, 0] - 1).astype(np.complex128)

    n = int((nsym - 2 * SPAN - 4) * sps)
    # time of sample n in symbols: a sample clock that is off by eps and tau
    t = (np.arange(n) + tau) * (1 + eps) / sps + SPAN

    if mod == "msk":
        a = 2 * bits[:, 0] - 1
        cum = np.concatenate([[0], np.cumsum(a)])
        k = np.floor(t).astype(int)
        phase = np.pi / 2 * (cum[k] + a[k] * (t - k))
        x = 0.7 * np.exp(1j * phase)
    else:
        p = rc if pulse == "rc" else rrc
        x = np.zeros(n, dtype=np.complex128)
        base = np.floor(t).astype(int)
        for d in range(-SPAN, SPAN + 1):
            k = base + d
            x += sym[k] * p(t - k)
        x *= AMP

    noise = NOISE * (rng.standard_normal(n) + (0 if real else 1j * rng.standard_normal(n)))
    x = x + noise
    if real:
        data = q15(x.real)
    else:
        data = np.stack([q15(x.real), q15(x.imag)], axis=1).reshape(-1)
    meta = {
        "name": name,
        "modulation": mod,
        "pulse": pulse,
        "alpha": ALPHA,
        "sps": sps,
        "timing_offset": tau,
        "rate_offset": eps,
        "amplitude": AMP if mod != "msk" else 0.7,
        "noise_rms": NOISE,
        "real": real,
        "samples": n,
        "symbols_re": sym.real.astype(int).tolist(),
        "symbols_im": sym.imag.astype(int).tolist(),
    }
    return data, meta


def make_taps(spec):
    if spec is None:
        return []
    if spec["kind"] == "rrc":
        return list(
            grfilter.firdes.root_raised_cosine(
                spec["gain"], spec["fs"], spec["symbol_rate"], spec["alpha"], spec["ntaps"]
            )
        )
    raise ValueError(spec)


def run_case(case, data, real):
    taps = make_taps(case.get("taps"))
    env = {"digital": digital, "filter": grfilter, "np": np, "math": math, "taps": taps}
    blk = eval(case["gnuradio"], env)
    if real:
        src = blocks.vector_source_f((data.astype(np.float32) / FULL_SCALE).tolist())
        snk = blocks.vector_sink_f()
    else:
        c = (data[0::2] + 1j * data[1::2]).astype(np.complex64) / FULL_SCALE
        src = blocks.vector_source_c(c.tolist())
        snk = blocks.vector_sink_c()
    tb = gr.top_block()
    tb.connect(src, blk, snk)
    tb.run()
    out = np.asarray(snk.data())
    if not real:
        out = np.stack([out.real, out.imag], axis=1).reshape(-1)
    return out.astype("<f4"), taps


def main(cases_dir, out_dir):
    with open(__file__, "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()
    os.makedirs(os.path.join(out_dir, "signals"), exist_ok=True)
    signals = {}
    for path in sorted(glob.glob(os.path.join(cases_dir, "*.json"))):
        with open(path, "rb") as f:
            raw = f.read()
        case = json.loads(raw)
        name = case["signal"]
        if name not in signals:
            data, meta = make_signal(name)
            data.astype("<i2").tofile(os.path.join(out_dir, "signals", name + ".s16le"))
            with open(os.path.join(out_dir, "signals", name + ".json"), "w") as f:
                json.dump(meta, f)
                f.write("\n")
            signals[name] = (data, meta)
        data, meta = signals[name]
        out, taps = run_case(case, data, meta["real"])
        case_dir = os.path.join(out_dir, case["name"])
        os.makedirs(case_dir, exist_ok=True)
        out.tofile(os.path.join(case_dir, "gnuradio.f32le"))
        manifest = {
            "case": case["name"],
            "case_sha256": hashlib.sha256(raw).hexdigest(),
            "script_sha256": script_sha,
            "gnuradio": gr.version(),
            "numpy": np.__version__,
            "taps": [float(t) for t in taps],
        }
        with open(os.path.join(case_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, sort_keys=True)
            f.write("\n")
        print(f"{case['name']}: {len(out) // (1 if meta['real'] else 2)} outputs")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
