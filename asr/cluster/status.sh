#!/usr/bin/env bash
# One-screen fleet status: training runs, scoring jobs, memory, disk.   cluster/status.sh cluster/hosts.txt
HOSTS=${1:-cluster/hosts.txt}
while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue
  printf "%-11s " "$h"
  ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$h" 'cd ~/Sichos/asr 2>/dev/null || { echo "no project dir"; exit; }
    for f in $(ls -t reports/train-*.log 2>/dev/null | head -2); do n=$(basename $f .log | sed s/^train-//)
      printf "%s: %s | evals[%s] %s%s | " "$n" "$(tail -c 400 $f | tr "\r" "\n" | grep -oE "[0-9]+/[0-9]+ \[[^]]*\]" | tail -1 | cut -c1-40)" \
        "$(grep -oE "eval_wer.: .?[0-9.]+" $f | grep -oE "[0-9.]+$" | tr "\n" " " | sed "s/ $//")" \
        "$(grep -E "^EXIT" $f | tail -1 | cut -d" " -f1-2)" "$(test -f runs/$n/model.safetensors -o -f runs/$n/adapter_model.safetensors && echo " SAVED")"
    done
    printf "trainers %s | scorers %s | %s | disk %s free\n" "$(pgrep -f pipeline/finetune.py | wc -l | tr -d " ")" "$(pgrep -f "pipeline/evaluate" | wc -l | tr -d " ")" "$(top -l 1 | grep -oE "[0-9]+[MG] unused")" "$(df -h ~ | tail -1 | awk "{print \$4}")"'
done < "$HOSTS"
