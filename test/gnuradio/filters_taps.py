#!/usr/bin/env python3
"""Filter design reference: GNU Radio's windows and firdes taps for a fixed
list of parameter sets, written as JSON for test/filters/design_test.exs.

Usage: filters_taps.py <output file>. Called by scripts/gnuradio_golden.sh for the
filters family (golden.py HOOKS), which writes test/fixtures/gnuradio/filters/taps.json.

Each entry is {"fun": name, "args": [...], "taps": [...]}; complex taps are
[re, im] pairs. Window types are given by name ("hamming", ...).
"""
import json
import sys

from gnuradio import fft
from gnuradio.filter import firdes

W = fft.window
TYPES = {
    "hamming": W.WIN_HAMMING,
    "hann": W.WIN_HANN,
    "blackman": W.WIN_BLACKMAN,
    "blackman_harris": W.WIN_BLACKMAN_HARRIS,
    "rectangular": W.WIN_RECTANGULAR,
    "kaiser": W.WIN_KAISER,
}

entries = []


def add(fun, args, taps):
    if taps and isinstance(taps[0], complex):
        out = [[float(t.real), float(t.imag)] for t in taps]
    else:
        out = [float(t) for t in taps]
    entries.append({"fun": fun, "args": args, "taps": out})


for n in (2, 3, 5, 8, 31, 64):
    for name, f in (
        ("hamming", W.hamming),
        ("hann", W.hann),
        ("blackman", W.blackman),
        ("blackman_harris", W.blackman_harris),
        ("rectangular", W.rectangular),
    ):
        add("window", [name, n, 0.0], f(n))
    for beta in (0.0, 2.5, 6.76, 9.0):
        add("window", ["kaiser", n, beta], W.kaiser(n, beta))

for wname in ("hamming", "hann", "blackman", "blackman_harris", "rectangular", "kaiser"):
    t = TYPES[wname]
    beta = 6.76
    add("low_pass", [1.0, 1.0, 0.1, 0.05, wname, beta], firdes.low_pass(1.0, 1.0, 0.1, 0.05, t, beta))
    add("low_pass", [2.5, 48000.0, 6000.0, 2000.0, wname, beta], firdes.low_pass(2.5, 48000.0, 6000.0, 2000.0, t, beta))
    add("high_pass", [1.0, 1.0, 0.2, 0.05, wname, beta], firdes.high_pass(1.0, 1.0, 0.2, 0.05, t, beta))
    add("high_pass", [1.0, 8000.0, 1000.0, 500.0, wname, beta], firdes.high_pass(1.0, 8000.0, 1000.0, 500.0, t, beta))
    add("band_pass", [1.0, 1.0, 0.1, 0.25, 0.04, wname, beta], firdes.band_pass(1.0, 1.0, 0.1, 0.25, 0.04, t, beta))
    add("band_pass", [3.0, 44100.0, 300.0, 3400.0, 400.0, wname, beta], firdes.band_pass(3.0, 44100.0, 300.0, 3400.0, 400.0, t, beta))
    add("band_reject", [1.0, 1.0, 0.1, 0.25, 0.04, wname, beta], firdes.band_reject(1.0, 1.0, 0.1, 0.25, 0.04, t, beta))
    add("complex_band_pass", [1.0, 1.0, 0.05, 0.25, 0.04, wname, beta], firdes.complex_band_pass(1.0, 1.0, 0.05, 0.25, 0.04, t, beta))
    add("complex_band_pass", [1.0, 1.0, -0.3, -0.1, 0.05, wname, beta], firdes.complex_band_pass(1.0, 1.0, -0.3, -0.1, 0.05, t, beta))
    add("hilbert", [19, wname, beta], firdes.hilbert(19, t, beta))
    add("hilbert", [7, wname, beta], firdes.hilbert(7, t, beta))

for kb in (1.0, 4.0, 8.5):
    add("low_pass", [1.0, 1.0, 0.1, 0.03, "kaiser", kb], firdes.low_pass(1.0, 1.0, 0.1, 0.03, TYPES["kaiser"], kb))

for alpha, spb, ntaps in ((0.35, 4.0, 33), (0.35, 4.0, 32), (0.5, 8.0, 65), (1.0, 2.0, 21), (0.25, 4.0, 41), (0.2, 3.0, 31), (0.35, 2.5, 25), (0.1, 5.0, 51)):
    add("root_raised_cosine", [1.0, spb, 1.0, alpha, ntaps], firdes.root_raised_cosine(1.0, spb, 1.0, alpha, ntaps))
add("root_raised_cosine", [2.0, 1000000.0, 250000.0, 0.35, 41], firdes.root_raised_cosine(2.0, 1000000.0, 250000.0, 0.35, 41))

for gain, spb, bt, ntaps in ((1.0, 4.0, 0.35, 9), (1.0, 4.0, 0.35, 33), (2.0, 8.0, 0.5, 64), (1.0, 2.0, 0.3, 17)):
    add("gaussian", [gain, spb, bt, ntaps], firdes.gaussian(gain, spb, bt, ntaps))

with open(sys.argv[1], "w") as f:
    json.dump({"gnuradio": __import__("gnuradio.gr", fromlist=["gr"]).version(), "entries": entries}, f)
    f.write("\n")
print(len(entries), "entries")
