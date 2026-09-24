#!/usr/bin/env bash
# Training environment on a Mac node without Homebrew or admin rights: uv-managed Python 3.12 in $HOME.
#   ssh <host> 'bash -s' < cluster/setup_train_node.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12 >/dev/null 2>&1 || true
cd "$HOME/Sichos/asr"
[ -d .venv-train ] || uv venv --python 3.12 .venv-train >/dev/null
uv pip install --python .venv-train/bin/python -q -r requirements.txt
.venv-train/bin/python -c "import torch, transformers, peft, faster_whisper; print('train env ready on', __import__('socket').gethostname(), '| torch', torch.__version__, '| mps', torch.backends.mps.is_available(), '| python', __import__('sys').version.split()[0])"
