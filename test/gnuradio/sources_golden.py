#!/usr/bin/env python3
"""GNU Radio runner for the sources family (test/dsp_cases/sources/*.json).

Usage: sources_golden.py <cases dir> <output dir>
(see docs/blocks/sources.md for the podman command; scripts/gnuradio_golden.sh
runs golden.py, which needs an input signal and cannot start a chain with a
source.)

Same case format and output files as golden.py (cases with "stats_only": true
write stats.json, the statistics of the output, instead of the samples), with one more input type:
{"type": "none"} for chains that start with a source block. Then "output"
("real", "complex" or "bytes") is required, "samples" is the number of
items taken from the chain, and input.s16le is empty. Block arguments are
Python expressions; besides "pi" they can use `analog.GR_SIN_WAVE`,
`analog.GR_GAUSSIAN` etc. Cases with an input signal run through
golden.run_case unchanged.
"""
import glob
import hashlib
import importlib
import json
import math
import os
import sys

import numpy as np
from gnuradio import analog, blocks, gr

import golden


def make_block(spec, params):
    module, name = spec["block"].rsplit(".", 1)
    cls = getattr(importlib.import_module("gnuradio." + module), name)
    env = {"__builtins__": {}, "pi": math.pi, "analog": analog}
    env.update({k: v for k, v in params.items() if isinstance(v, (int, float, list))})
    args = [eval(a, env) if isinstance(a, str) else a for a in spec.get("args", [])]
    return cls(*args)


golden.make_block = make_block

SINKS = {"bytes": blocks.vector_sink_b, "complex": blocks.vector_sink_c, "real": blocks.vector_sink_f}
ITEM_SIZE = {"bytes": 1, "complex": 8, "real": 4}


def run_case(case):
    if case["input"]["type"] != "none":
        return golden.run_case(case)
    out_type = case["output"]["type"]
    chain = [make_block(b, case["params"]) for b in case["gnuradio"]]
    head = blocks.head(ITEM_SIZE[out_type], case["samples"])
    sink = SINKS[out_type]()
    tb = gr.top_block()
    tb.connect(*chain, head, sink)
    tb.run()
    out = np.asarray(sink.data())
    if out_type == "bytes":
        return np.zeros(0, dtype=np.int16), out.astype(np.uint8)
    if out_type == "complex":
        out = np.stack([out.real, out.imag], axis=1).reshape(-1)
    return np.zeros(0, dtype=np.int16), out.astype(np.float32)


def flatness(x, seg=256):
    """Spectral flatness of the averaged Hann periodogram (same as the Elixir test helper)."""
    k = len(x) // seg
    segs = np.asarray(x[: k * seg], dtype=np.float64).reshape(k, seg)
    n = np.arange(seg)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * n / seg)
    p = (np.abs(np.fft.fft(segs * win, axis=1)) ** 2).mean(axis=0)[1 : seg // 2]
    return float(np.exp(np.log(p).mean()) / p.mean())


def component_stats(x, impulse):
    x = np.asarray(x, dtype=np.float64)
    m, v = x.mean(), x.var()
    st = {"n": len(x), "mean": m, "variance": v, "kurtosis": float(((x - m) ** 4).mean() / v**2),
          "flatness": flatness(x)}
    nz = x[x != 0.0]
    st.update({"nonzero": len(nz), "nonzero_mean": float(nz.mean()) if len(nz) else 0.0,
               "nonzero_mean_square": float((nz**2).mean()) if len(nz) else 0.0})
    return st


def stats(case, ref):
    """Statistics of a random block's output; the raw samples are not kept."""
    if case.get("output", {}).get("type", case["input"]["type"]) == "complex":
        comps = [ref[0::2], ref[1::2]]
    else:
        comps = [ref]
    out = {"components": [component_stats(c, False) for c in comps]}
    if comps[0] is not None and len(comps) == 2:
        p = (comps[0].astype(np.float64) ** 2 + comps[1].astype(np.float64) ** 2)
        # power relative to the constant input amplitude (fading cases), when given
        a = case["input"].get("tones", [{"amp": 1.0}])[0]["amp"] if case["input"]["type"] != "none" else 1.0
        p = p / (a * a)
        out["power_mean"] = float(p.mean())
        out["power_variance"] = float(p.var())
    return out


def main(cases_dir, out_dir):
    for path in sorted(glob.glob(os.path.join(cases_dir, "*.json"))):
        with open(path, "rb") as f:
            raw = f.read()
        case = json.loads(raw)
        x16, ref = run_case(case)

        case_dir = os.path.join(out_dir, case["name"])
        os.makedirs(case_dir, exist_ok=True)
        if case.get("stats_only"):
            with open(os.path.join(case_dir, "stats.json"), "w") as f:
                json.dump(stats(case, ref), f, indent=1, sort_keys=True)
                f.write("\n")
            for old in ("input.s16le", "gnuradio.f32le"):
                if os.path.exists(os.path.join(case_dir, old)):
                    os.remove(os.path.join(case_dir, old))
            manifest = {"case": case["name"], "case_sha256": hashlib.sha256(raw).hexdigest(),
                        "gnuradio": gr.version(), "numpy": np.__version__}
            with open(os.path.join(case_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2, sort_keys=True)
                f.write("\n")
            print(f"{case['name']}: statistics of {len(ref)} values")
            continue
        if x16.dtype == np.uint8:
            x16.tofile(os.path.join(case_dir, "input.u8"))
        else:
            x16.astype("<i2").tofile(os.path.join(case_dir, "input.s16le"))
        if ref.dtype == np.uint8:
            ref.tofile(os.path.join(case_dir, "gnuradio.u8"))
        else:
            ref.astype("<f4").tofile(os.path.join(case_dir, "gnuradio.f32le"))
        manifest = {
            "case": case["name"],
            "case_sha256": hashlib.sha256(raw).hexdigest(),
            "gnuradio": gr.version(),
            "numpy": np.__version__,
        }
        with open(os.path.join(case_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"{case['name']}: {len(ref)} values")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
