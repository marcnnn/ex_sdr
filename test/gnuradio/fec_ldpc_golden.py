#!/usr/bin/env python3
"""GNU Radio reference data for the LDPC blocks of gr-fec.

Usage: fec_ldpc_golden.py <cases dir> <alist dir> <output dir>

Reads test/dsp_cases/fec/ldpc_*.json and writes, per case, a directory
<output dir>/<name>/ with

  input.u8 (encoders: information bits, one per byte) or
  input.f32le (decoders: LLRs, positive for 1),
  info.u8 (decoders: the information bits that were sent),
  gnuradio.u8 (the block's output, one bit per byte),
  manifest.json (case file hash, versions).

Run it in the podman image (gr-fec ships the alist files used here, they
are copied to test/fixtures/gnuradio/fec/ldpc_alist):

  podman run --rm -v "$PWD/test:/work/test:Z" -w /work exhdl-gnuradio \
    python3 test/gnuradio/fec_ldpc_golden.py test/dsp_cases/fec \
      test/fixtures/gnuradio/fec/ldpc_alist test/fixtures/gnuradio/fec
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
from gnuradio import blocks, fec, gr


def run(coder, data, in_size, out_size, decoder=False):
    if decoder:
        src = blocks.vector_source_f([float(x) for x in data], False)
        blk = fec.decoder(coder, in_size, out_size)
    else:
        src = blocks.vector_source_b([int(x) for x in data], False)
        blk = fec.encoder(coder, in_size, out_size)
    snk = blocks.vector_sink_b()
    tb = gr.top_block()
    tb.connect(src, blk, snk)
    tb.run()
    return np.array(snk.data(), dtype=np.uint8)


def encoder_for(case, alist):
    b = case["block"]
    if b == "ldpc_par_mtrx_encoder" or (b == "ldpc_bit_flip_decoder" and case["matrix"] == "H"):
        m = fec.ldpc_H_matrix(alist, case["gap"])
        return m, fec.ldpc_par_mtrx_encoder_make_H(m)
    if b == "ldpc_gen_mtrx_encoder" or (b == "ldpc_bit_flip_decoder" and case["matrix"] == "G"):
        m = fec.ldpc_G_matrix(alist)
        return m, fec.ldpc_gen_mtrx_encoder_make(m)
    enc = fec.ldpc_encoder_make(alist)
    return None, enc


def main(cases_dir, alist_dir, out_root):
    for path in sorted(glob.glob(os.path.join(cases_dir, "ldpc_*.json"))):
        raw = open(path, "rb").read()
        case = json.loads(raw)
        rng = np.random.default_rng(case["seed"])
        alist = os.path.join(alist_dir, case["alist"])
        mtrx, enc = encoder_for(case, alist)
        k = enc.get_input_size()
        n = enc.get_output_size()
        frames = case["frames"]
        info = rng.integers(0, 2, k * frames).astype(np.uint8)
        code = run(enc, info, gr.sizeof_char, gr.sizeof_char)
        assert len(code) == n * frames

        out = os.path.join(out_root, case["name"])
        os.makedirs(out, exist_ok=True)
        if case["block"].endswith("encoder"):
            info.tofile(os.path.join(out, "input.u8"))
            code.tofile(os.path.join(out, "gnuradio.u8"))
        else:
            points = case["ebn0_db"]
            per = frames // len(points)
            sigmas = []
            for i in range(frames):
                ebn0 = 10 ** (points[min(i // per, len(points) - 1)] / 10)
                sigmas.append(np.sqrt(1 / (2 * (k / n) * ebn0)))
            sigma = np.repeat(np.array(sigmas), n)
            y = (2.0 * code.astype(np.float64) - 1.0) + sigma * rng.standard_normal(n * frames)
            llr = (2.0 * y / sigma**2).astype(np.float32)

            if case["block"] == "ldpc_bit_flip_decoder":
                dec = fec.ldpc_bit_flip_decoder.make(mtrx.get_base_sptr(), case["max_iter"])
            else:
                dec = fec.ldpc_decoder.make(alist, case["max_iter"])
            decoded = run(dec, llr, gr.sizeof_float, gr.sizeof_char, decoder=True)
            assert len(decoded) == k * frames
            llr.astype("<f4").tofile(os.path.join(out, "input.f32le"))
            info.tofile(os.path.join(out, "info.u8"))
            decoded.tofile(os.path.join(out, "gnuradio.u8"))
            errors = int(np.sum(decoded != info))
            print(case["name"], "bit errors after decoding:", errors)

        manifest = {
            "case": case["name"],
            "case_sha256": hashlib.sha256(raw).hexdigest(),
            "gnuradio": gr.version(),
            "numpy": np.__version__,
            "n": n,
            "k": k,
        }
        with open(os.path.join(out, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        print(case["name"], "n", n, "k", k, "frames", frames)


if __name__ == "__main__":
    main(*sys.argv[1:4])
