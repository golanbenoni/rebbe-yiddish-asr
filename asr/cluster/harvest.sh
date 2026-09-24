#!/usr/bin/env bash
# Harvest finished training runs in place on each node (see harvest_node.sh) and pull the small reports here.
#   cluster/harvest.sh cluster/hosts.txt
set -u
HOSTS=${1:-cluster/hosts.txt}
LOCAL=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$LOCAL/reports/v1"
while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue
  rsync -az "$LOCAL/cluster/harvest_node.sh" "$h:~/Sichos/asr/cluster/harvest_node.sh" 2>/dev/null || { ssh -n -o BatchMode=yes "$h" 'mkdir -p ~/Sichos/asr/cluster'; rsync -az "$LOCAL/cluster/harvest_node.sh" "$h:~/Sichos/asr/cluster/harvest_node.sh"; }
  rsync -az --delete "$LOCAL/pipeline/" "$h:~/Sichos/asr/pipeline/"   # scoring uses the current pipeline (running trainers are unaffected)
  ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$h" 'bash ~/Sichos/asr/cluster/harvest_node.sh'
done < "$HOSTS"
while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue
  for n in $(ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$h" 'cd ~/Sichos/asr && for d in runs/*/; do n=$(basename $d); [ -f $d/.harvest_done ] && [ ! -f $d/.pulled ] && echo $n; done'); do
    mkdir -p "$LOCAL/reports/hf"; rsync -az --include="hf-$n*/" --include="hf-$n*/**" --exclude='*' "$h:~/Sichos/asr/reports/" "$LOCAL/reports/hf/"
    rsync -az --include="v1-$n*/" --include="v1-$n*/**" --exclude='*' "$h:~/Sichos/asr/reports/" "$LOCAL/reports/v1/" && ssh -n -o BatchMode=yes "$h" "touch ~/Sichos/asr/runs/$n/.pulled" && echo "  pulled reports for $n from $h (CT2 rows v1-$n, GPU-path rows hf-$n)"
  done
done < "$HOSTS"
