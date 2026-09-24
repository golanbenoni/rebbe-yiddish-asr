#!/usr/bin/env bash
# Pull the train-scan shard reports back and build the cleaned train manifest here.
#   cluster/pull_scan.sh cluster/hosts.txt
set -euo pipefail
HOSTS=${1:?hosts file}
LOCAL=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$LOCAL/asr/reports"
while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue
  rsync -az --include='train-scan-shard*/' --include='train-scan-shard*/**' --exclude='*' "$h:~/Sichos/asr/reports/" "$LOCAL/asr/reports/" && echo "  pulled scan reports from $h"
done < "$HOSTS"
cd "$LOCAL/asr" && source .venv/bin/activate
python -u pipeline/filter_manifest.py --manifest data/manifests/train.jsonl --report reports/train-scan-shard*/report.json --out data/manifests/train.clean.jsonl
