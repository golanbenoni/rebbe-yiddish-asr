#!/usr/bin/env bash
# Run remotely on each node:  ssh <host> 'bash -s' < cluster/setup_node.sh
# Per-node copy model: everything lives in $ROOT (default ~/Sichos), pushed by cluster/push.sh.
set -euo pipefail
ROOT=${ROOT:-$HOME/Sichos}
mkdir -p "$ROOT/asr"
cd "$ROOT/asr"
PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3 || true)
[ -n "$PY" ] || { echo "NO PYTHON on $(hostname): install Xcode command line tools (xcode-select --install) or brew install python@3.12"; exit 2; }
echo "node $(hostname -s): $($PY --version 2>&1), $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, $(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.ncpu) performance cores, macOS $(sw_vers -productVersion)"
command -v afconvert >/dev/null || { echo "afconvert missing?!"; exit 2; }
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements-align.txt
python -c "import faster_whisper, soundfile, jiwer, pyluach; print('align env ready on', __import__('socket').gethostname())"
