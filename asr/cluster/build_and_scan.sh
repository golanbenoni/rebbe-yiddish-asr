#!/usr/bin/env bash
# After alignment (cluster/pull_days.sh done on this Mac):
#  1. push the merged day records to every node,
#  2. every node builds the identical dataset locally (deterministic: same days, same splits),
#  3. every node scores its 1/N shard of the TRAIN split with the base model (for filter_manifest),
#  4. pull the shard reports back.
#   cluster/build_and_scan.sh cluster/hosts.txt
set -euo pipefail
HOSTS=${1:?hosts file}
MODEL=${MODEL:-ivrit-ai/yi-whisper-large-v3-turbo-ct2}
YEARS=${YEARS:-"5783 5784 5785 5786 5787"}
LOCAL=$(cd "$(dirname "$0")/../.." && pwd)
hosts=(); while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue; hosts+=("$h"); done < "$HOSTS"
N=${#hosts[@]}
echo "== 1. push merged day records + splits + pipeline"
for h in "${hosts[@]}"; do
  ( rsync -az "$LOCAL/asr/data/days/" "$h:~/Sichos/asr/data/days/" && rsync -az "$LOCAL/asr/data/splits.json" "$h:~/Sichos/asr/data/" \
    && rsync -az --delete "$LOCAL/asr/pipeline/" "$h:~/Sichos/asr/pipeline/" && echo "  pushed $h" ) &
done; wait
echo "== 2. build the dataset on every node (identical output), then 3. scan train shard"
for i in "${!hosts[@]}"; do
  h=${hosts[$i]}
  ssh -n -o BatchMode=yes "$h" "cd ~/Sichos/asr && source .venv/bin/activate && export OMP_NUM_THREADS=4 && rm -rf data/segments data/manifests && nohup bash -c 'python -u pipeline/build_dataset.py --year $YEARS --splits-file data/splits.json > reports/build.log 2>&1 && for w in 0 1 2 3 4; do python -u pipeline/evaluate.py --model $MODEL --name train-scan --splits train --shard \$(( $i*5 + w ))/$(( N*5 )) > reports/scan-\$w.log 2>&1 & done; wait; echo SCAN_DONE >> reports/scan-0.log' > /dev/null 2>&1 & echo started build+scan on \$(hostname -s)"
done
echo "monitor:  for h in \${hosts[@]}; do ssh -n \$h 'tail -1 ~/Sichos/asr/reports/scan-0.log'; done"
echo "when every node shows SCAN_DONE:  cluster/pull_scan.sh $HOSTS"
