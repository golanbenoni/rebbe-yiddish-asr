#!/usr/bin/env bash
# Wave 2 of the v5 alignment (2026-09-23). Waits for the audio staging (cluster/v5_stage_and_wave2.sh) and for the
# wave-1 workers to exit, pulls their results, REBUILDS the day records of 5776-5778 from the watermark-free text
# (their wave-1 alignments used text with the diagonal watermark glued into body rows), pushes everything to the
# nodes and launches align5b over all days still missing (5775-5777, 5778-5780 again with clean text, and any failures).
set -u
cd "$(dirname "$0")/.."
HOSTS=${1:-cluster/hosts_align.txt}; LOG=reports/fleet/v5-stage-wave2.log
until grep -q 'AUDIO STAGED ON NODES' "$LOG" 2>/dev/null; do sleep 60; done
pkill -f v5_stage_and_wave2.sh && echo "  (old chain stopped after staging; wave 2 handled here)" >> "$LOG"
running() { local n=0; while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue; c=$(ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$h" 'pgrep -f align_day.py | wc -l' 2>/dev/null | tr -d ' '); n=$((n + ${c:-1})); done < "$HOSTS"; echo $n; }
until [ "$(running)" -eq 0 ]; do sleep 120; done
{
echo "=== WAVE 1 FINISHED $(date +%H:%M) ==="
bash cluster/pull_days.sh "$HOSTS" 2>&1 | tail -3
source .venv/bin/activate
python -u pipeline/build_pdf_days.py --year 5776 5777 5778 5779 5780 --force 2>&1 | tail -6   # clean text (watermark/footer junk removed); their wave-1 alignments are redone
python3 - <<'PY'
import json, glob
for y in (5775, 5776, 5777, 5778, 5779, 5780, 5781):
    fs = glob.glob(f"data/days/{y}/*.json"); ok = 0
    for f in fs:
        r = json.load(open(f))
        if (r.get("sync_local") or {}).get("segments") or (r.get("sync") or {}).get("segments"): ok += 1
    print(f"  {y}: {ok}/{len(fs)} days aligned after wave 1")
PY
bash cluster/v5_prepare_nodes.sh "$HOSTS" 2>&1 | tail -10
echo "=== launching wave 2 $(date +%H:%M) ==="
WORKERS=5 THREADS=4 TAG=align5b MODEL=runs/turbo-full-lr3e5-4ep-ct2 bash cluster/align_fleet.sh "$HOSTS" 2>&1 | tail -9
echo "=== ALIGN5B LAUNCHED $(date +%H:%M) ==="
} >> "$LOG" 2>&1
