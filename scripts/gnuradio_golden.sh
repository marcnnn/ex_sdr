#!/usr/bin/env bash
# Runs the shared DSP cases (test/dsp_cases/*.json) through GNU Radio.
#
#   scripts/gnuradio_golden.sh            # regenerate test/fixtures/gnuradio/
#   scripts/gnuradio_golden.sh <out dir>  # write elsewhere (used by the live tests)
#   scripts/gnuradio_golden.sh <out dir> <cases dir>
#       # another case directory, e.g. test/dsp_cases/filters
#       #   -> test/fixtures/gnuradio/filters
#
# The family (the case directory's name below test/dsp_cases) selects extra
# scripts for fixtures that are not sample streams (see HOOKS in
# test/gnuradio/golden.py): digital/framing.json, filters/taps.json and
# filters/pfb.json. The case format is described in test/gnuradio/golden.py
# and docs/blocks.md.
#
# Uses a native GNU Radio if /usr/bin/python3 can import it, otherwise the
# podman image from test/gnuradio/Containerfile (rootless, no sudo).
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-test/fixtures/gnuradio}
cases=${2:-test/dsp_cases}
mkdir -p "$out"
out=$(cd "$out" && pwd)
cases=$(cd "$cases" && pwd)
family=""
if [ "$(basename "$(dirname "$cases")")" = dsp_cases ]; then family=$(basename "$cases"); fi

if /usr/bin/python3 -c "import gnuradio" 2>/dev/null; then
  /usr/bin/python3 test/gnuradio/golden.py "$cases" "$out" "$family"
else
  podman build -q -t exhdl-gnuradio test/gnuradio >/dev/null
  podman run --rm -v "$PWD/test:/work/test:ro,Z" -v "$cases:/cases:ro,Z" -v "$out:/out:Z" -w /work \
    exhdl-gnuradio python3 test/gnuradio/golden.py /cases /out "$family" 2>&1 | grep -v vmcircbuf_prefs
fi
