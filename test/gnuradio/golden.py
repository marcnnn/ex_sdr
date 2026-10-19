#!/usr/bin/env python3
"""GNU Radio runner for the DSP cases in test/dsp_cases/**/*.json.

Usage: golden.py <cases dir> <output dir> [family]
(normally via scripts/gnuradio_golden.sh, the single entry point).

For every case it generates the input signal(s) (seeded numpy), quantises
them to Q15 exactly as ExSdr sees them, runs the case's GNU Radio block chain
and writes <output dir>/<case>/:

  input.s16le     little-endian int16 (interleaved I/Q for complex input);
                  input.u8 for bytes; input1.*, input2.* for more inputs
  gnuradio.f32le  little-endian float32, 1.0 = 32768 (interleaved I/Q for
                  complex); gnuradio.u8 (bytes), gnuradio.s16le ("short"),
                  gnuradio.s32le ("int"); gnuradio1.*, ... for more outputs
  stats.json      instead of the two files above for "stats_only" cases
  manifest.json   versions and the SHA-256 of the case file it was made from

Case fields:

  "input": {"type": "complex" | "real" | "bytes" | "none", ...}
      Q15 tones, steps and noise; bytes are random below "modulus" (default
      256). "none" is for chains that start with a source block: "output" is
      then required, "samples" is the number of items taken from the chain
      and input.s16le is empty.
  "inputs": [ {input spec}, ... ]
      Several inputs, connected to ports 0..n-1 of the first block. They are
      drawn in order from the case's seeded generator, so the first one is
      the same as the single "input" would be.
  "output": {"type": ...} or "outputs": [ {...}, ... ]
      "complex", "real", "bytes", "short" (int16 sink), "int" (int32 sink).
      Defaults to the type of the first input. Several outputs are read from
      ports 0..m-1 of the last block.
  "stats_only": true
      Store statistics of the output (stats.json) instead of the samples,
      for random blocks.
  "gnuradio": [ {"block": "blocks.add_ff", "args": [...]}, ... ]
      The chain, by module and class. String args are Python expressions
      evaluated with: pi, math, np/numpy, the modules gr, blocks, filter,
      analog, digital, fec, channels (those the image has), the case's
      params as bare names when they are valid identifiers, and the dict `p`
      with all of them (p["pass"] for names that are Python keywords; an arg
      that is just a param name gives the param, whatever its name).
      Args can construct objects, e.g. "digital.constellation_qpsk().base()"
      or "blocks.GR_MSB_FIRST".

Cases in family directories may need more than a chain (tagged streams,
PDUs, taps); those are produced by the scripts listed in HOOKS, which
this runner calls after the cases of that family.

GNU Radio computes in float; the ExSdr runner (test/gnuradio_reference_test.exs)
compares its Q15 results against this per sample and by SNR.
"""
import glob
import hashlib
import importlib
import json
import keyword
import math
import os
import subprocess
import sys

import numpy as np
from gnuradio import blocks, gr

FULL_SCALE = 32768.0

# Scripts for fixtures that are not sample streams: family -> [(script, file in the output dir)].
# Families whose cases need their own runner (same arguments: cases dir, output dir).
RUNNERS = {"carrier": "carrier_golden.py"}

HOOKS = {
    "digital": [("digital_framing_golden.py", "framing.json")],
    "filters": [("filters_taps.py", "taps.json"), ("filters_pfb.py", "pfb.json")],
    # Scripts with their own case directory get <cases>/<dir> <out>/<dir>.
    "timing": [("timing_golden.py", "sync", "sync")],
}

SAFE_BUILTINS = {
    n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
    for n in ("abs", "bool", "complex", "dict", "float", "int", "len", "list", "max", "min",
              "range", "round", "str", "sum", "tuple", "zip", "sorted")
}

ITEM_SIZE = {"bytes": 1, "short": 2, "int": 4, "real": 4, "complex": 8}
SINKS = {
    "bytes": blocks.vector_sink_b,
    "short": blocks.vector_sink_s,
    "int": blocks.vector_sink_i,
    "real": blocks.vector_sink_f,
    "complex": blocks.vector_sink_c,
}
SOURCES = {"bytes": blocks.vector_source_b, "complex": blocks.vector_source_c, "real": blocks.vector_source_f}


def q15(x):
    """Rounds floats in [-1, 1) to int16, saturating."""
    return np.clip(np.round(np.asarray(x) * FULL_SCALE), -32768, 32767).astype(np.int16)


def make_input(spec, samples, rng):
    """The input signal described by an "input" section: Q15 ints, or
    uint8 for "bytes" (random values below "modulus", default 256), or an
    empty array for "none"."""
    if spec["type"] == "none":
        return np.zeros(0, dtype=np.int16)
    if spec["type"] == "bytes":
        return rng.integers(0, spec.get("modulus", 256), samples).astype(np.uint8)

    n = np.arange(samples)
    complex_input = spec["type"] == "complex"
    x = np.zeros(samples, dtype=np.complex128 if complex_input else np.float64)

    for tone in spec.get("tones", []):
        angle = 2 * np.pi * tone["freq"] * n + tone.get("phase", 0.0)
        x = x + tone["amp"] * (np.exp(1j * angle) if complex_input else np.sin(angle))

    for step in spec.get("steps", []):
        x = x + step["amp"] * (n >= step["at"])

    noise = spec.get("noise_rms", 0.0)
    if complex_input:
        x = x + noise * (rng.standard_normal(samples) + 1j * rng.standard_normal(samples))
        return np.stack([q15(x.real), q15(x.imag)], axis=1).reshape(-1)
    return q15(x + noise * rng.standard_normal(samples))


def expression_env(params):
    """Names visible to the "args" expressions. Globals (not locals), so
    comprehensions like "[t / 32768 for t in taps_q15]" see them."""
    env = {"__builtins__": SAFE_BUILTINS, "pi": math.pi, "math": math, "np": np, "numpy": np, "gr": gr,
           "blocks": blocks}
    for name in ("filter", "analog", "digital", "fec", "channels"):
        try:
            env[name] = importlib.import_module("gnuradio." + name)
        except ImportError:
            pass
    env["p"] = dict(params)
    env.update({k: v for k, v in params.items() if k.isidentifier() and not keyword.iskeyword(k)})
    return env


def evaluate(arg, env, params):
    """A non-string arg as is; a string that is a param name (also "pass") as
    that param; any other string as a Python expression."""
    if not isinstance(arg, str):
        return arg
    return params[arg] if arg in params else eval(arg, env)


def make_block(spec, params):
    """Instantiates e.g. {"block": "blocks.rotator_cc", "args": ["2 * pi * phase_inc / 65536"]}."""
    module, name = spec["block"].rsplit(".", 1)
    cls = getattr(importlib.import_module("gnuradio." + module), name)
    env = expression_env(params)
    args = [evaluate(a, env, params) for a in spec.get("args", [])]
    return cls(*args)


def input_specs(case):
    return case["inputs"] if "inputs" in case else [case["input"]]


def output_specs(case):
    if "outputs" in case:
        return case["outputs"]
    return [case.get("output", {"type": input_specs(case)[0]["type"]})]


def source_block(spec, x16):
    if spec["type"] == "bytes":
        return blocks.vector_source_b(x16.tolist())
    if spec["type"] == "complex":
        return blocks.vector_source_c((x16[0::2] + 1j * x16[1::2]).astype(np.complex64) / FULL_SCALE)
    return blocks.vector_source_f(x16.astype(np.float32) / FULL_SCALE)


def sink_data(sink, out_type):
    out = np.asarray(sink.data())
    if out_type == "bytes":
        return out.astype(np.uint8)
    if out_type == "short":
        return out.astype(np.int16)
    if out_type == "int":
        return out.astype(np.int32)
    if out_type == "complex":
        out = np.stack([out.real, out.imag], axis=1).reshape(-1)
    return out.astype(np.float32)


def run_case(case):
    """Returns ([inputs as Q15 / uint8 arrays], [outputs: float32, uint8, int16 or int32 arrays])."""
    rng = np.random.default_rng(case["seed"])
    specs = input_specs(case)
    xs = [make_input(s, case["samples"], rng) for s in specs]
    outs = output_specs(case)
    chain = [make_block(b, case["params"]) for b in case["gnuradio"]]
    sinks = [SINKS[o["type"]]() for o in outs]
    tb = gr.top_block()

    if specs[0]["type"] == "none":
        # a source chain: take "samples" items
        head = blocks.head(ITEM_SIZE[outs[0]["type"]], case["samples"])
        tb.connect(*chain, head, sinks[0])
    else:
        sources = [source_block(s, x) for s, x in zip(specs, xs)]
        for i, src in enumerate(sources):
            tb.connect(src, (chain[0], i))
        if len(chain) > 1:
            tb.connect(*chain)
        for j, sink in enumerate(sinks):
            tb.connect((chain[-1], j), sink)
    tb.run()
    return xs, [sink_data(s, o["type"]) for s, o in zip(sinks, outs)]


def flatness(x, seg=256):
    """Spectral flatness of the averaged Hann periodogram (same as the Elixir test helper)."""
    k = len(x) // seg
    segs = np.asarray(x[: k * seg], dtype=np.float64).reshape(k, seg)
    n = np.arange(seg)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * n / seg)
    p = (np.abs(np.fft.fft(segs * win, axis=1)) ** 2).mean(axis=0)[1 : seg // 2]
    return float(np.exp(np.log(p).mean()) / p.mean())


def component_stats(x):
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
    in0 = input_specs(case)[0]
    if output_specs(case)[0]["type"] == "complex":
        comps = [ref[0::2], ref[1::2]]
    else:
        comps = [ref]
    out = {"components": [component_stats(c) for c in comps]}
    if len(comps) == 2:
        p = comps[0].astype(np.float64) ** 2 + comps[1].astype(np.float64) ** 2
        # power relative to the constant input amplitude (fading cases), when given
        a = in0.get("tones", [{"amp": 1.0}])[0]["amp"] if in0["type"] != "none" else 1.0
        p = p / (a * a)
        out["power_mean"] = float(p.mean())
        out["power_variance"] = float(p.var())
    return out


def write_array(path_base, a, kind):
    """kind "input" or "gnuradio"; the file extension follows the dtype."""
    if a.dtype == np.uint8:
        a.tofile(path_base + ".u8")
    elif kind == "input" or a.dtype == np.int16:
        a.astype("<i2").tofile(path_base + ".s16le")
    elif a.dtype == np.int32:
        a.astype("<i4").tofile(path_base + ".s32le")
    else:
        a.astype("<f4").tofile(path_base + ".f32le")


def write_case(case, raw, xs, refs, out_dir):
    case_dir = os.path.join(out_dir, case["name"])
    os.makedirs(case_dir, exist_ok=True)
    if case.get("stats_only"):
        with open(os.path.join(case_dir, "stats.json"), "w") as f:
            json.dump(stats(case, refs[0]), f, indent=1, sort_keys=True)
            f.write("\n")
        for old in ("input.s16le", "input.u8", "gnuradio.f32le"):
            if os.path.exists(os.path.join(case_dir, old)):
                os.remove(os.path.join(case_dir, old))
    else:
        for i, x in enumerate(xs):
            write_array(os.path.join(case_dir, "input" + (str(i) if i else "")), x, "input")
        for j, ref in enumerate(refs):
            write_array(os.path.join(case_dir, "gnuradio" + (str(j) if j else "")), ref, "gnuradio")
    manifest = {
        "case": case["name"],
        "case_sha256": hashlib.sha256(raw).hexdigest(),
        "gnuradio": gr.version(),
        "numpy": np.__version__,
    }
    with open(os.path.join(case_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    what = "statistics of " if case.get("stats_only") else ""
    print(f"{case['name']}: {what}{len(refs[0])} values")


def main(cases_dir, out_dir, family=""):
    here = os.path.dirname(os.path.abspath(__file__))
    if family in RUNNERS:
        subprocess.run([sys.executable, os.path.join(here, RUNNERS[family]), cases_dir, out_dir], check=True)
        return

    for path in sorted(glob.glob(os.path.join(cases_dir, "*.json"))):
        with open(path, "rb") as f:
            raw = f.read()
        case = json.loads(raw)
        xs, refs = run_case(case)
        write_case(case, raw, xs, refs, out_dir)

    for script, target, *cases in HOOKS.get(family, []):
        args = [os.path.join(cases_dir, c) for c in cases] + [os.path.join(out_dir, target)]
        subprocess.run([sys.executable, os.path.join(here, script)] + args, check=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
