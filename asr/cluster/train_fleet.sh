#!/usr/bin/env bash
# One experiment per node (embarrassingly parallel): line i of the configs file runs on host i.
# Memory: 256 GB nodes need --gradient-checkpointing (and/or batch 8 x accum 4) for full/LoRA fine-tunes;
# fp32 turbo full FT at batch 16 without checkpointing takes ~211 GB and killed two nodes on 2026-09-17.
# Verify 5 min after launch: the log must show step 1/ progress and no EXIT line.
#   cluster/train_fleet.sh cluster/hosts.txt cluster/train_configs.txt
# Outputs land in $ROOT/asr/runs/<name>/ on the share; logs in asr/reports/train-<name>.log.
set -euo pipefail
HOSTS=${1:?hosts file}; CONFIGS=${2:?configs file}
ROOT=${ROOT:-\$HOME/Sichos}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-train.clean.jsonl}
hosts=(); while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue; hosts+=("$h"); done < "$HOSTS"
LOCAL=$(cd "$(dirname "$0")/../.." && pwd)
for h in "${hosts[@]}"; do   # every node gets the current manifests (train.clean.jsonl) and pipeline first
  rsync -az "$LOCAL/asr/data/manifests/" "$h:~/Sichos/asr/data/manifests/" && rsync -az --delete "$LOCAL/asr/pipeline/" "$h:~/Sichos/asr/pipeline/" && echo "  synced manifests+pipeline to $h"
done
i=0
while IFS='|' read -r name args; do
  name=$(echo "$name" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'); [[ -z "$name" || "$name" == \#* ]] && continue   # (sed, not xargs: an apostrophe in a comment line made xargs fail and set -e abort the launch on 2026-09-22)
  h=${hosts[$((i % ${#hosts[@]}))]}
  if ssh -n -o BatchMode=yes "$h" "test -f ~/Sichos/asr/reports/train-$name.log"; then
    echo "skip $name on $h (log exists; delete ~/Sichos/asr/reports/train-$name.log to rerun)"; i=$((i + 1)); continue
  fi
  # preflight: unified memory actually free on the node (other workloads may hold a lot of it)
  free_gb=$(ssh -n -o BatchMode=yes "$h" "top -l 1 | grep PhysMem | grep -oE '[0-9]+[MG] unused' | awk '{u=\$1; if (u ~ /M/) print int(u)/1024; else print int(u)}'")
  if [ -n "$free_gb" ] && [ "${free_gb%.*}" -lt "${MIN_FREE_GB:-80}" ]; then
    echo "REFUSE $name on $h: only ${free_gb} GB unused (MIN_FREE_GB=${MIN_FREE_GB:-80})"; i=$((i + 1)); continue
  fi
  # PYTORCH_MPS_HIGH_WATERMARK_RATIO caps the MPS allocator below physical RAM (the LOW ratio must be set below it,
  # PyTorch's default low of 1.4 otherwise raises 'invalid low watermark ratio') so an oversized run fails with a
  # Python 'MPS backend out of memory' traceback instead of thrashing the node into a jetsam kill or a panic.
  ssh -n -o BatchMode=yes "$h" "cd $ROOT/asr && source .venv-train/bin/activate && export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=${MPS_RATIO:-0.7} PYTORCH_MPS_LOW_WATERMARK_RATIO=${MPS_LOW_RATIO:-0.6} && ( nohup python -u pipeline/finetune.py $args --train-manifest $TRAIN_MANIFEST --output runs/$name --eval-limit 200 --eval-steps 500 --save-steps 500 --num-workers 2 </dev/null >>reports/train-$name.log 2>&1; echo \"EXIT \$? \$(date)\" >>reports/train-$name.log ) >/dev/null 2>&1 & echo started $name on \$(hostname -s) - ${free_gb:-?} GB unused"
  i=$((i + 1))
done < "$CONFIGS"
