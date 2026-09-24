#!/usr/bin/env bash
# Fan the alignment of all text-only days across the fleet.
#   cluster/align_fleet.sh cluster/hosts.txt            (WORKERS=6 processes per node by default)
# Per-node copy model (cluster/push.sh first): each worker takes a disjoint shard of days and writes
# sync_local into that day's JSON on its node; cluster/pull_days.sh merges them back. Logs on each
# node: ~/Sichos/asr/reports/align-<host>-<w>.log
set -euo pipefail
HOSTS=${1:?hosts file}
ROOT=${ROOT:-\$HOME/Sichos}
WORKERS=${WORKERS:-6}
MODEL=${MODEL:-ivrit-ai/yi-whisper-large-v3-turbo-ct2}
THREADS=${THREADS:-4}
SELECT=${SELECT:---all-missing}     # e.g. SELECT="--site-timed --force" to cross-check site timings
TAG=${TAG:-align}
hosts=(); while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue; hosts+=("$h"); done < "$HOSTS"
N=${#hosts[@]}; TOTAL=$((N * WORKERS))
echo "aligning on $N nodes x $WORKERS workers = $TOTAL shards, model $MODEL"
for i in "${!hosts[@]}"; do
  h=${hosts[$i]}
  cmd="cd $ROOT/asr && mkdir -p reports && source .venv/bin/activate && export OMP_NUM_THREADS=$THREADS && "
  for ((w=0; w<WORKERS; w++)); do
    shard=$((i * WORKERS + w))
    cmd+="nohup python -u pipeline/align_day.py $SELECT --model $MODEL --shard $shard/$TOTAL > reports/$TAG-$h-$w.log 2>&1 & "
  done
  cmd+="echo started $WORKERS workers on \$(hostname)"
  ssh "$h" "$cmd"
done
echo "progress: cluster/progress.sh $HOSTS"
