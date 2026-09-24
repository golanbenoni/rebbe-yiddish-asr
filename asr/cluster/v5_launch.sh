#!/usr/bin/env bash
# v5 training launch (2026-09-24), guarded: waits for "V5 DATASET READY" (cluster/v5_build_and_scan.sh), refuses if the
# cleaned train manifest is small or any year lost > 25% of its clips to the scan filter, then starts the runs of
# cluster/train_configs_v5.txt on cluster/hosts_v5.txt and arms a GPU-path scoring chain per run (long-form test with the
# default auto windows + clip dev/test/test_b), pulling the reports into reports/hf/.
set -u
cd "$(dirname "$0")/.."
LOG=reports/fleet/v5-build-scan.log
until grep -q 'V5 DATASET READY' "$LOG" 2>/dev/null; do sleep 300; done
source .venv/bin/activate
python3 - <<'PY' || { echo "!!! GUARD FAILED - v5 not launched (see above)"; exit 1; }
import json, collections, sys
rows = [json.loads(l) for l in open("data/manifests/train.jsonl", encoding="utf-8")]
kept = {json.loads(l)["audio"] for l in open("data/manifests/train.clean.jsonl", encoding="utf-8")}
by = collections.defaultdict(lambda: [0, 0])
for r in rows:
    y = str(r.get("year", "?")); by[y][0] += 1; by[y][1] += (r["audio"] not in kept)
bad = [f"{y}: {d}/{n}" for y, (n, d) in sorted(by.items()) if n > 500 and d / n > 0.25]
print(f"train.clean rows {len(kept)}; per-year drop: " + ", ".join(f"{y} {100*d/max(1,n):.1f}%" for y, (n, d) in sorted(by.items())))
if len(kept) < 45000 or bad:
    print("guard: too few clean clips or a year lost > 25%:", len(kept), bad); sys.exit(1)
PY
echo "=== launching v5 $(date +%H:%M) ==="
TRAIN_MANIFEST=train.clean.jsonl bash cluster/train_fleet.sh cluster/hosts_v5.txt cluster/train_configs_v5.txt 2>&1 | tail -12
# scoring chains: one per (host, run) pair in file order
hosts=(); while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue; hosts+=("$h"); done < cluster/hosts_v5.txt
i=0
while IFS='|' read -r name args; do
  name=$(echo "$name" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'); [[ -z "$name" || "$name" == \#* ]] && continue
  h=${hosts[$((i % ${#hosts[@]}))]}; i=$((i + 1))
  nohup bash -c "
    until ssh -n -o BatchMode=yes -o ConnectTimeout=20 $h \"grep -q '^EXIT 0' ~/Sichos/asr/reports/train-$name.log\" 2>/dev/null; do sleep 600; done
    echo \"=== $name finished \$(date '+%m-%d %H:%M'); scoring on $h ===\"
    ssh -n -o BatchMode=yes $h \"cd ~/Sichos/asr && source .venv-train/bin/activate && nohup bash -c 'python -u pipeline/evaluate_longform.py --backend hf --model runs/$name --name hf-$name --splits-file data/splits.json > reports/early-$name-hflf.log 2>&1; python -u pipeline/evaluate.py --backend hf --model runs/$name --name hf-$name --splits dev test test_b --batch 8 > reports/early-$name-hfclip.log 2>&1; echo done > reports/early-$name.done' >/dev/null 2>&1 </dev/null &\"
    until ssh -n -o BatchMode=yes -o ConnectTimeout=20 $h 'test -f ~/Sichos/asr/reports/early-$name.done' 2>/dev/null; do sleep 300; done
    rsync -az --include='hf-$name*/' --include='hf-$name*/**' --exclude='*' $h:~/Sichos/asr/reports/ reports/hf/
    ssh -n -o BatchMode=yes $h \"grep -h '^{' ~/Sichos/asr/reports/early-$name-hflf.log | cut -c1-220\"
    echo \"=== $name GPU-PATH SCORES PULLED \$(date '+%m-%d %H:%M') ===\"
  " > reports/fleet/v5-score-$name.log 2>&1 &
  echo "  scoring chain armed for $name on $h"
done < cluster/train_configs_v5.txt
echo "=== V5 LAUNCHED $(date +%H:%M) ==="
