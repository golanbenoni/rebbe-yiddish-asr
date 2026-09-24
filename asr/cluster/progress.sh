#!/usr/bin/env bash
# Aligned-day count and running processes per node.   cluster/progress.sh cluster/hosts.txt
HOSTS=${1:?hosts file}; TAG=${TAG:-align}   # TAG=align5 for the v5 waves (logs reports/<TAG>-<host>-<w>.log)
total_p=0; total_d=0
while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue
  out=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$h" 'd=$(cat ~/Sichos/asr/reports/'"$TAG"'-*.log 2>/dev/null | grep -c anchors); p=$(pgrep -f align_day.py | wc -l | tr -d " "); f=$(cat ~/Sichos/asr/reports/'"$TAG"'-*.log 2>/dev/null | grep -c FAILED); echo "$d $p $f"' 2>/dev/null || echo "? ? ?")
  set -- $out; printf "  %-11s days done %4s  processes %3s  failed %3s\n" "$h" "$1" "$2" "$3"
  [[ "$1" =~ ^[0-9]+$ ]] && total_d=$((total_d + $1)); [[ "$2" =~ ^[0-9]+$ ]] && total_p=$((total_p + $2))
done < "$HOSTS"
echo "total: $total_d days done, $total_p processes running"
