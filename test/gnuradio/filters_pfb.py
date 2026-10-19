#!/usr/bin/env python3
"""Reference for gr-filter blocks that the case format of golden.py cannot
build: pfb_decimator_ccf has one input per decimation phase, fed here by
stream_to_streams from one serial stream.

Usage: filters_pfb.py <output file>. Called by scripts/gnuradio_golden.sh for the
filters family (golden.py HOOKS), which writes test/fixtures/gnuradio/filters/pfb.json.

Writes {"gnuradio": version, "cases": [{name, decim, channel, taps, input, output}]};
input and output are lists of [re, im] with input values on the Q15 grid.
"""
import json
import sys

import numpy as np
from gnuradio import blocks, gr
from gnuradio import filter as flt

rng = np.random.default_rng(3030)


def q15(x):
    return np.round(np.clip(x, -1, 32767 / 32768) * 32768) / 32768


cases = []
for name, decim, channel, ntaps in (("pfb_decimator_4_ch0", 4, 0, 40), ("pfb_decimator_4_ch1", 4, 1, 40), ("pfb_decimator_5_ch3", 5, 3, 55)):
    taps = (np.hamming(ntaps) * np.sinc((np.arange(ntaps) - (ntaps - 1) / 2) / decim) / decim).tolist()
    n = 400
    x = q15(0.3 * rng.standard_normal(n)) + 1j * q15(0.3 * rng.standard_normal(n))
    tb = gr.top_block()
    src = blocks.vector_source_c(x.astype(np.complex64))
    s2s = blocks.stream_to_streams(gr.sizeof_gr_complex, decim)
    blk = flt.pfb_decimator_ccf(decim, taps, channel, False, False)
    snk = blocks.vector_sink_c()
    tb.connect(src, s2s)
    for m in range(decim):
        tb.connect((s2s, m), (blk, m))
    tb.connect(blk, snk)
    tb.run()
    y = np.array(snk.data())
    cases.append({
        "name": name, "decim": decim, "channel": channel, "taps": taps,
        "input": [[float(v.real), float(v.imag)] for v in x],
        "output": [[float(v.real), float(v.imag)] for v in y],
    })

# pfb_interpolator_ccf: one input, a plain chain
interp = []
for name, i, ntaps in (("pfb_interpolator_3", 3, 30), ("pfb_interpolator_8", 8, 80)):
    taps = (np.hamming(ntaps) * np.sinc((np.arange(ntaps) - (ntaps - 1) / 2) / i)).tolist()
    n = 200
    x = q15(0.3 * rng.standard_normal(n)) + 1j * q15(0.3 * rng.standard_normal(n))
    tb = gr.top_block()
    src = blocks.vector_source_c(x.astype(np.complex64))
    blk = flt.pfb_interpolator_ccf(i, taps)
    snk = blocks.vector_sink_c()
    tb.connect(src, blk, snk)
    tb.run()
    y = np.array(snk.data())
    interp.append({
        "name": name, "interp": i, "taps": taps,
        "input": [[float(v.real), float(v.imag)] for v in x],
        "output": [[float(v.real), float(v.imag)] for v in y],
    })

with open(sys.argv[1], "w") as f:
    json.dump({"gnuradio": gr.version(), "cases": cases, "interpolators": interp}, f)
    f.write("\n")
print(len(cases), "cases")
