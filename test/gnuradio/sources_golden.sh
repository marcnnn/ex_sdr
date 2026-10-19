#!/usr/bin/env bash
# Regenerates test/fixtures/gnuradio/sources from test/dsp_cases/sources
# (rootless podman image exhdl-gnuradio, or a native GNU Radio).
set -euo pipefail
cd "$(dirname "$0")/../.."
out=test/fixtures/gnuradio/sources
cases=test/dsp_cases/sources
mkdir -p "$out"
out=$(cd "$out" && pwd)
cases=$(cd "$cases" && pwd)
if /usr/bin/python3 -c "import gnuradio" 2>/dev/null; then
  PYTHONPATH=test/gnuradio /usr/bin/python3 test/gnuradio/sources_golden.py "$cases" "$out"
else
  podman build -q -t exhdl-gnuradio test/gnuradio >/dev/null
  podman run --rm -v "$PWD/test:/work/test:ro,Z" -v "$cases:/cases:ro,Z" -v "$out:/out:Z" -w /work/test/gnuradio \
    exhdl-gnuradio python3 sources_golden.py /cases /out 2>&1 | grep -v vmcircbuf
fi
