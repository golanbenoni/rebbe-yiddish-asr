#!/usr/bin/env bash
# Filter -> push -> rebuild on nodes -> launch v1 matrix -> baseline/v0 scoring on bigmac05.
set -u
cd "$(dirname "$0")/.." && source .venv/bin/activate
HOSTS=$(grep -vE '^\s*(#|$)' cluster/hosts.txt)
echo "== 1. filter train split with the fleet scan reports"
python -u pipeline/filter_manifest.py --manifest data/manifests/train.jsonl --report reports/train-scan-shard*/report.json --out data/manifests/train.clean.jsonl
echo "== 2. push day records, splits, pipeline"
for h in $HOSTS; do ( rsync -az data/days/ "$h:~/Sichos/asr/data/days/" && rsync -az data/splits.json "$h:~/Sichos/asr/data/" && rsync -az --delete pipeline/ "$h:~/Sichos/asr/pipeline/" && echo "  pushed $h" ) & done; wait
echo "== 3. rebuild the dataset on every node (8 workers each)"
for h in $HOSTS; do
  ssh -n -o BatchMode=yes "$h" "cd ~/Sichos/asr && rm -rf data/segments data/manifests reports/build-v2.log && (nohup bash -c 'source .venv/bin/activate; python -u pipeline/build_dataset.py --year 5783 5784 5785 5786 5787 --splits-file data/splits.json --workers 8 > reports/build-v2.log 2>&1; echo BUILD_V2_DONE >> reports/build-v2.log' </dev/null >/dev/null 2>&1 &); echo \"  rebuild started on \$(hostname -s)\""
done
for i in $(seq 1 60); do
  done_n=0
  for h in $HOSTS; do ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$h" 'grep -q BUILD_V2_DONE ~/Sichos/asr/reports/build-v2.log 2>/dev/null' && done_n=$((done_n + 1)); done
  echo "  $(date +%H:%M) rebuilt on $done_n/8 nodes"; [ "$done_n" -eq 8 ] && break; sleep 60
done
for h in $HOSTS; do printf "  %-11s " "$h"; ssh -n -o BatchMode=yes "$h" 'grep -E "^ *all:" ~/Sichos/asr/reports/build-v2.log | tr -s " "'; done
echo "== 4. launch the v1 matrix"
cluster/train_fleet.sh cluster/hosts.txt cluster/train_configs_v1.txt
echo "== 5. baseline + v0 scoring on bigmac05 (dev+test, 5 shards per model)"
V0=""
if [ -f runs/v0-step1000-ct2/model.bin ]; then rsync -az runs/v0-step1000-ct2/ bigmac05:~/Sichos/asr/runs/v0-step1000-ct2/ && V0="runs/v0-step1000-ct2"; fi
ssh -n -o BatchMode=yes bigmac05 "cd ~/Sichos/asr && rm -f reports/devtest-v2.done && (nohup bash -c 'source .venv/bin/activate; export OMP_NUM_THREADS=4; for m in ivrit-ai/yi-whisper-large-v3-turbo-ct2 ivrit-ai/yi-whisper-large-v3-ct2 $V0; do for w in 0 1 2 3 4; do python -u pipeline/evaluate.py --model \$m --name devtest-\$(basename \$m) --splits dev test --shard \$w/5 > reports/devtest-\$(basename \$m)-\$w.log 2>&1 & done; wait; done; echo DEVTEST_V2_DONE > reports/devtest-v2.done' </dev/null >/dev/null 2>&1 &); echo \"  scoring started on bigmac05 (v0 included: ${V0:-no})\""
echo "=== CHAIN DONE ==="
