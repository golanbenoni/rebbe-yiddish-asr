#!/usr/bin/env bash
# 2026-09-23: stage the 5775-5777 audio to the hub (the first staging rsync dropped the connection and its "done"
# marker was written anyway), push it hub -> alignment nodes, then, once the running align5 wave has finished, pull
# its results and launch a second wave over the days that are still missing (5775-5777 + any failures).
set -u
cd "$(dirname "$0")/.."
HUB=${HUB:-supermac01}; HOSTS=${1:-cluster/hosts_align.txt}
for attempt in 1 2 3 4 5; do
  rsync -az --partial --timeout=180 "../DS 5775 CHUL" "../DS 5776" "../DS 5777" "$HUB:~/Sichos/new_audio/" && break
  echo "  local -> hub rsync failed (attempt $attempt), retrying"; sleep 15
done
ssh -n -o BatchMode=yes "$HUB" 'cd ~/Sichos && for d in "DS 5775 CHUL" "DS 5776" "DS 5777"; do [ -e "$d" ] || ln -s "new_audio/$d" "$d"; echo "hub $d: $(ls "new_audio/$d"/*/*.mp3 2>/dev/null | wc -l | tr -d " ") mp3"; done'
while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue
  ( for attempt in 1 2 3; do ssh -n -o BatchMode=yes "$HUB" "rsync -az --partial --timeout=180 ~/Sichos/new_audio/ $h:~/Sichos/new_audio/" && break; echo "  hub -> $h failed (attempt $attempt)"; sleep 15; done
    ssh -n -o BatchMode=yes "$h" 'cd ~/Sichos && for d in "DS 5775 CHUL" "DS 5776" "DS 5777"; do [ -e "$d" ] || ln -s "new_audio/$d" "$d"; done; echo "  $(hostname): 5775 $(ls "new_audio/DS 5775 CHUL"/*/*.mp3 2>/dev/null | wc -l | tr -d " "), 5776 $(ls "new_audio/DS 5776"/*/*.mp3 2>/dev/null | wc -l | tr -d " "), 5777 $(ls "new_audio/DS 5777"/*/*.mp3 2>/dev/null | wc -l | tr -d " ") mp3"' ) &
done < "$HOSTS"
wait
echo "=== AUDIO STAGED ON NODES $(date +%H:%M) ==="
# wait for the first wave to finish everywhere
running() { local n=0; while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue; c=$(ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$h" 'pgrep -f align_day.py | wc -l' 2>/dev/null | tr -d ' '); n=$((n + ${c:-1})); done < "$HOSTS"; echo $n; }
until [ "$(running)" -eq 0 ]; do sleep 120; done
echo "=== WAVE 1 FINISHED $(date +%H:%M) ==="
bash cluster/pull_days.sh "$HOSTS" 2>&1 | tail -3
source .venv/bin/activate
python3 - <<'PY'
import json, glob
for y in (5775, 5776, 5777, 5778, 5779, 5780, 5781):
    fs = glob.glob(f"data/days/{y}/*.json"); ok = 0
    for f in fs:
        r = json.load(open(f))
        if (r.get("sync_local") or {}).get("segments") or (r.get("sync") or {}).get("segments"): ok += 1
    print(f"  {y}: {ok}/{len(fs)} days aligned")
PY
bash cluster/v5_prepare_nodes.sh "$HOSTS" 2>&1 | tail -10
echo "=== launching wave 2 $(date +%H:%M) ==="
WORKERS=5 THREADS=4 TAG=align5b MODEL=runs/turbo-full-lr3e5-4ep-ct2 bash cluster/align_fleet.sh "$HOSTS" 2>&1 | tail -9
echo "=== ALIGN5B LAUNCHED $(date +%H:%M) ==="
