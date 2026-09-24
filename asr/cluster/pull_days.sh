#!/usr/bin/env bash
# Pull alignment results back: each node wrote sync_local into disjoint day JSONs, so newer files win.
#   cluster/pull_days.sh cluster/hosts.txt
set -euo pipefail
HOSTS=${1:?hosts file}
ROOT_REMOTE=${ROOT_REMOTE:-Sichos}
LOCAL=$(cd "$(dirname "$0")/../.." && pwd)
hosts=(); while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue; hosts+=("$h"); done < "$HOSTS"
for h in "${hosts[@]}"; do
  rsync -azu "$h:~/$ROOT_REMOTE/asr/data/days/" "$LOCAL/asr/data/days/" && echo "pulled from $h"
  rsync -az "$h:~/$ROOT_REMOTE/asr/reports/" "$LOCAL/asr/reports/fleet/$h/" 2>/dev/null || true
done
cd "$LOCAL" && python3 - <<'PY'
import json; from pathlib import Path
days=[json.loads(p.read_text()) for p in Path("asr/data/days").rglob("*.json")]
print(len(days), "days;", sum(1 for d in days if d.get("sync_local")), "with local alignment;", sum(1 for d in days if (d.get("sync") or {}).get("segments")), "with site timings")
PY
