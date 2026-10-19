#!/usr/bin/env python3
"""GNU Radio reference data for the convolutional codes of gr-fec
(cc_encoder, cc_decoder) in all four modes, and for the soft input
conversion of the extended decoder.

Usage: fec_conv_golden.py <output file>

The objects are driven frame by frame through generic_work (as the
fec.encoder / fec.decoder blocks do), so streaming state, chained start
states and the decoder history are part of the data. Run in the podman
image, see test/gnuradio/digital_framing_golden.py:

  podman run --rm -v "$PWD/test:/work/test:ro,Z" \
    -v "$PWD/test/fixtures/gnuradio/fec:/out:Z" -w /work exhdl-gnuradio \
    python3 test/gnuradio/fec_conv_golden.py /out/conv.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fec_common import bitstr, encode_frames, hexs, work  # noqa: E402
from gnuradio import blocks, fec, gr  # noqa: E402

rng = np.random.default_rng(7)
MODES = {
    "streaming": fec.CC_STREAMING,
    "terminated": fec.CC_TERMINATED,
    "truncated": fec.CC_TRUNCATED,
    "tailbiting": fec.CC_TAILBITING,
}


def encoder_cases():
    cases = []
    codes = [
        (7, [79, 109]),
        (7, [109, 79]),
        (7, [-79, 109]),
        (3, [7, 5]),
        (5, [25, 27]),
        (7, [79, 109, 89]),
        (9, [491, 369]),
    ]
    for k, polys in codes:
        for mode in MODES:
            for start in sorted({0, (1 << (k - 1)) - 1, 5 % (1 << (k - 1))}):
                if mode == "tailbiting" and start != 0:
                    continue
                for padded in ([False, True] if mode == "terminated" else [False]):
                    f = int(rng.integers(max(k, 8), 40))
                    enc = fec.cc_encoder_make(f, k, len(polys), polys, start, MODES[mode], padded)
                    frames = [rng.integers(0, 2, f).astype(np.uint8) for _ in range(3)]
                    outs = encode_frames(enc, frames)
                    cases.append(
                        {
                            "k": k,
                            "polys": polys,
                            "mode": mode,
                            "start_state": start,
                            "padded": padded,
                            "frame_size": f,
                            "input_size": enc.get_input_size(),
                            "output_size": enc.get_output_size(),
                            "rate": enc.rate(),
                            "frames": [bitstr(x) for x in frames],
                            "outputs": [bitstr(x[: enc.get_output_size()]) for x in outs],
                        }
                    )
    return cases


def noisy(bits, sigma):
    """u8 soft symbols for code bits: 0 / 255 plus Gaussian noise."""
    x = np.asarray(bits, dtype=float) * 255.0 + rng.normal(0, sigma, len(bits))
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def decoder_cases():
    cases = []
    k, polys = 7, [79, 109]
    plan = []
    for mode in MODES:
        for sigma in (60, 110, 160):
            plan.append((mode, sigma, 0, -1, False))
    plan += [
        ("truncated", 90, 9, -1, False),
        ("terminated", 90, 9, -1, False),
        ("terminated", 90, 9, 9, False),
        ("terminated", 90, 0, 0, False),
        ("terminated", 120, 0, -1, True),
        ("streaming", 100, 17, -1, False),
        ("tailbiting", 100, 0, -1, False),
    ]
    for mode, sigma, start, end, padded in plan:
        f = int(rng.integers(40, 90))
        enc = fec.cc_encoder_make(f, k, 2, polys, start, MODES[mode], padded)
        dec = fec.cc_decoder.make(f, k, 2, polys, start, end, MODES[mode], padded)
        nframes = 4
        frames = [rng.integers(0, 2, f).astype(np.uint8) for _ in range(nframes)]
        coded = encode_frames(enc, frames)
        insize, hist, outsize = dec.get_input_size(), dec.get_history(), dec.get_output_size()
        # the decoder reads insize + hist items per frame; in streaming mode
        # the frames are contiguous (the look-ahead is the next frame)
        if mode == "streaming":
            code = np.concatenate([c[: enc.get_output_size()] for c in coded])
            stream = noisy(code, sigma)
            # a few junk symbols after the end so the last frame has a look-ahead
            stream = np.concatenate([stream, noisy(rng.integers(0, 2, hist), sigma)])
            chunks = [stream[i * insize : i * insize + insize + hist] for i in range(nframes)]
        else:
            chunks = []
            for c in coded:
                s = noisy(c[: enc.get_output_size()], sigma)
                buf = np.zeros(insize + hist, dtype=np.uint8)
                n = min(len(s), insize)
                buf[:n] = s[:n]
                chunks.append(buf)
        outs = []
        for ch in chunks:
            out = np.zeros(outsize, dtype=np.uint8)
            outs.append(work(dec, np.ascontiguousarray(ch), out).copy())
        cases.append(
            {
                "k": k,
                "polys": polys,
                "mode": mode,
                "sigma": sigma,
                "start_state": start,
                "end_state": end,
                "padded": padded,
                "frame_size": f,
                "input_size": insize,
                "output_size": outsize,
                "history": hist,
                "shift": dec.get_shift(),
                "input_conversion": dec.get_input_conversion(),
                "sent": [bitstr(x) for x in frames],
                "inputs": [hexs(c) for c in chunks],
                "outputs": [bitstr(o) for o in outs],
            }
        )
    # pure junk symbols: decisions follow the metric rules alone (ties, wrap)
    for mode in MODES:
        f = 50
        dec = fec.cc_decoder.make(f, k, 2, polys, 0, -1, MODES[mode], False)
        insize, hist = dec.get_input_size(), dec.get_history()
        chunks = [rng.integers(0, 256, insize + hist).astype(np.uint8) for _ in range(3)]
        outs = [work(dec, c, np.zeros(f, dtype=np.uint8)).copy() for c in chunks]
        cases.append(
            {
                "k": k,
                "polys": polys,
                "mode": mode,
                "sigma": None,
                "start_state": 0,
                "end_state": -1,
                "padded": False,
                "frame_size": f,
                "input_size": insize,
                "output_size": f,
                "history": hist,
                "shift": 128.0,
                "input_conversion": "uchar",
                "sent": None,
                "inputs": [hexs(c) for c in chunks],
                "outputs": [bitstr(o) for o in outs],
            }
        )
    return cases


def uchar_conversion():
    """blocks.float_to_uchar(scale=48, bias=128) as the extended decoder uses it."""
    x = np.concatenate(
        [
            np.array([0.0, 0.5 / 48, -0.5 / 48, 1.0, -1.0, 2.6, -2.7, 3.0, -3.0, 10.0, -10.0]),
            rng.normal(0, 1.2, 200),
        ]
    ).astype(np.float32)
    src = blocks.vector_source_f(x.tolist(), False)
    cv = blocks.float_to_uchar(1, 48.0, 128.0)
    snk = blocks.vector_sink_b()
    tb = gr.top_block()
    tb.connect(src, cv, snk)
    tb.run()
    return {"input": x.tolist(), "output": [int(v) for v in snk.data()]}


if __name__ == "__main__":
    data = {
        "encoders": encoder_cases(),
        "decoders": decoder_cases(),
        "float_to_uchar": uchar_conversion(),
    }
    with open(sys.argv[1], "w") as fh:
        json.dump(data, fh)
