#!/usr/bin/env bash
# v5 dataset (2026-09-23): after BOTH alignment waves - pull the day records, build the dataset on every node and here
# (deterministic: same days, same splits; data/splits.json unchanged + data/splits_b.json as the second held-out set),
# scan the TRAIN split with the production CT2 model in 5 shards per node, pull the shard reports, filter, and report the
# per-year drop rate (the OCR/decode-quality signal). Stops there: the v5 training launch is a separate decision.
#   cluster/v5_build_and_scan.sh cluster/hosts_align.txt
set -u
cd "$(dirname "$0")/.."
HOSTS=${1:-cluster/hosts_align.txt}; LOG=reports/fleet/v5-stage-wave2.log
MODEL=${MODEL:-runs/turbo-full-lr3e5-4ep-ct2}
YEARS=${YEARS:-"5775 5776 5777 5778 5779 5780 5781 5783 5784 5785 5786 5787"}
hosts=(); while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue; hosts+=("$h"); done < "$HOSTS"
N=${#hosts[@]}
until grep -q 'ALIGN5B LAUNCHED' "$LOG" 2>/dev/null; do sleep 120; done
running() { local n=0; for h in "${hosts[@]}"; do c=$(ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$h" 'pgrep -f align_day.py | wc -l' 2>/dev/null | tr -d ' '); n=$((n + ${c:-1})); done; echo $n; }
sleep 300; until [ "$(running)" -eq 0 ]; do sleep 120; done
echo "=== WAVE 2 FINISHED $(date +%H:%M) ==="
bash cluster/pull_days.sh "$HOSTS" 2>&1 | tail -3
source .venv/bin/activate
python3 - <<'PY'
import json, glob
tot = 0
for y in (5775, 5776, 5777, 5778, 5779, 5780, 5781, 5783, 5784, 5785, 5786, 5787):
    fs = glob.glob(f"data/days/{y}/*.json"); ok = 0
    for f in fs:
        r = json.load(open(f))
        if (r.get("sync_local") or {}).get("segments") or (r.get("sync") or {}).get("segments"): ok += 1
    tot += ok; print(f"  {y}: {ok}/{len(fs)} days aligned")
print(f"  total aligned days: {tot}")
PY
echo "== push day records + splits + pipeline; clear old scan reports"
rm -rf reports/train-scan-shard*
for h in "${hosts[@]}"; do
  ( rsync -az "data/days/" "$h:~/Sichos/asr/data/days/" && rsync -az data/splits.json data/splits_b.json "$h:~/Sichos/asr/data/" \
    && rsync -az --delete pipeline/ "$h:~/Sichos/asr/pipeline/" && ssh -n -o BatchMode=yes "$h" 'rm -rf ~/Sichos/asr/reports/train-scan-shard* ~/Sichos/asr/reports/scan-*.log' && echo "  pushed $h" ) &
done; wait
echo "== build the dataset on every node, then scan its train shards ($(date +%H:%M))"
for i in "${!hosts[@]}"; do
  h=${hosts[$i]}
  ssh -n -o BatchMode=yes "$h" "cd ~/Sichos/asr && source .venv/bin/activate && export OMP_NUM_THREADS=4 && rm -rf data/segments data/manifests && nohup bash -c 'python -u pipeline/build_dataset.py --year $YEARS --splits-file data/splits.json --splits-b-file data/splits_b.json --workers 8 > reports/build.log 2>&1 && for w in 0 1 2 3 4; do python -u pipeline/evaluate.py --model $MODEL --name train-scan --splits train --suppress allow-quotes --shard \$(( $i*5 + w ))/$(( N*5 )) > reports/scan-\$w.log 2>&1 & done; wait; echo SCAN_DONE >> reports/scan-0.log' > /dev/null 2>&1 & echo started build+scan on \$(hostname -s)"
done
echo "== build the dataset here too (manifests for the filter and for training configs)"
rm -rf data/segments data/manifests
python -u pipeline/build_dataset.py --year $YEARS --splits-file data/splits.json --splits-b-file data/splits_b.json --workers 8 > reports/build-v5.log 2>&1
tail -6 reports/build-v5.log
for s in all train dev test test_b; do [ -f data/manifests/$s.jsonl ] && echo "  $s: $(wc -l < data/manifests/$s.jsonl | tr -d ' ') rows"; done
scan_done() { local n=0; for h in "${hosts[@]}"; do ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$h" 'grep -q SCAN_DONE ~/Sichos/asr/reports/scan-0.log 2>/dev/null' && n=$((n + 1)); done; echo $n; }
until [ "$(scan_done)" -eq "$N" ]; do sleep 300; done
echo "=== SCAN DONE on $N nodes $(date +%H:%M) ==="
bash cluster/pull_scan.sh "$HOSTS" 2>&1 | tail -4
python3 - <<'PY'
import json, collections
rows = [json.loads(l) for l in open("data/manifests/train.jsonl", encoding="utf-8")]
kept = {json.loads(l)["audio"] for l in open("data/manifests/train.clean.jsonl", encoding="utf-8")}
by = collections.defaultdict(lambda: [0, 0])
for r in rows:
    y = str(r.get("year") or r.get("day", "")[-4:] or "?")
    by[y][0] += 1; by[y][1] += (r["audio"] not in kept)
print("  per-year train clips / dropped by the scan filter:")
for y in sorted(by): print(f"    {y}: {by[y][0]:6d} clips, dropped {by[y][1]:5d} ({100*by[y][1]/max(1,by[y][0]):.1f}%)")
print(f"  train.clean: {len(kept)} clips")
PY
echo "=== V5 DATASET READY $(date +%H:%M) ==="
