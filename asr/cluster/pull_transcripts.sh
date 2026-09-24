#!/usr/bin/env bash
# Pull Phase 7 outputs (txt/srt/json) from every node into asr/output/.   cluster/pull_transcripts.sh cluster/hosts.txt
HOSTS=${1:-cluster/hosts.txt}; LOCAL=$(cd "$(dirname "$0")/.." && pwd); mkdir -p "$LOCAL/output"
while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue
  rsync -az "$h:~/Sichos/asr/output/" "$LOCAL/output/" && echo "  pulled from $h"
done < "$HOSTS"
echo "transcripts: $(find "$LOCAL/output" -name '*.txt' | wc -l | tr -d ' ') files"
