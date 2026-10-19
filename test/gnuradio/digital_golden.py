#!/usr/bin/env python3
"""GNU Radio runner for the digital family's stream cases (test/dsp_cases/digital/*.json).

Usage: digital_golden.py <cases dir> <output dir>

Same case format and output files as golden.py (it is imported and driven),
but the "args" of a GNU Radio block may be Python expressions that build
objects, e.g. "digital.constellation_qpsk().base()" or
"digital.psk_constellation(4, 'gray', True)", and "params" values may be
any JSON value. The expression environment has pi, math, numpy as np and
the modules blocks, filter, analog, digital, gr, plus the case's params.

Run it like golden.py (rootless podman, image exhdl-gnuradio):

  podman run --rm -v "$PWD/test:/work/test:ro,Z" \
    -v "$PWD/test/dsp_cases/digital:/cases:ro,Z" \
    -v "$PWD/test/fixtures/gnuradio/digital:/out:Z" -w /work exhdl-gnuradio \
    python3 test/gnuradio/digital_golden.py /cases /out
"""
import importlib
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import golden  # noqa: E402
import numpy as np  # noqa: E402
from gnuradio import analog, blocks, digital, filter, gr  # noqa: E402


def make_block(spec, params):
    module, name = spec["block"].rsplit(".", 1)
    cls = getattr(importlib.import_module("gnuradio." + module), name)
    env = {
        "__builtins__": {"list": list, "range": range, "len": len, "int": int, "float": float, "tuple": tuple},
        "pi": math.pi,
        "math": math,
        "np": np,
        "blocks": blocks,
        "filter": filter,
        "analog": analog,
        "digital": digital,
        "gr": gr,
    }
    env.update(params)
    args = [eval(a, env) if isinstance(a, str) else a for a in spec.get("args", [])]
    return cls(*args)


if __name__ == "__main__":
    golden.make_block = make_block
    golden.main(sys.argv[1], sys.argv[2])
