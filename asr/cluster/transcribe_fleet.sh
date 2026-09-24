#!/usr/bin/env bash
# Phase 7: transcribe the untranscribed year folders (5778-5781 by default) across the fleet.
#   cluster/transcribe_fleet.sh cluster/hosts.txt <origin-host> <ct2-dir-on-origin>   e.g. supermac01 runs/turbo-frozen-lr1e5-4ep-ct2
#   BACKEND=hf runs the HF fp32 model on each node's GPU instead (one process per node, ~20x realtime; MODEL is then the HF dir).
#   EXTRA_ARGS defaults to "--suppress allow-quotes" for ct2 (2026-09-21 fix: Whisper otherwise blocks the gershayim) and "--backend hf --batch 8" for hf.
#   Outputs overwrite ~/Sichos/asr/output/ on the nodes (transcribe_file.py has no skip-if-exists).
# 1) copies the CT2 model from the origin node to every node (node-to-node, needs enable_node_to_node.sh),
# 2) fans out each node's shard of the mp3s from the origin node (which holds ~/Sichos/new_audio, pushed once from this Mac),
# 3) runs transcribe_file.py in WORKERS processes per node, outputs mirrored under ~/Sichos/asr/output/<year>/.
# Pull results with cluster/pull_transcripts.sh.
set -u
HOSTS=${1:?hosts}; ORIGIN=${2:?origin host}; MODEL=${3:?ct2 dir on origin}
WORKERS=${WORKERS:-5}; YEARS=${YEARS:-"DS 5778|Sicha Yomis 5779 audio|Sicha Yomis 5780 audio|Sicha Yomis 5781 audio"}
BACKEND=${BACKEND:-ct2}   # ct2 = faster-whisper int8, WORKERS CPU processes per node; hf = HF fp32 model on the GPU (pipeline/hf_longform.py), 1 process per node
if [ "$BACKEND" = hf ]; then VENV=.venv-train; WORKERS=${HF_WORKERS:-1}; EXTRA_ARGS=${EXTRA_ARGS:-"--backend hf --batch 8"}; MODEL_FILE=model.safetensors
else VENV=.venv; EXTRA_ARGS=${EXTRA_ARGS:-"--suppress allow-quotes"}; MODEL_FILE=model.bin; fi   # (2026-09-21 fix: Whisper otherwise blocks the gershayim of abbreviations)
LOCAL=$(cd "$(dirname "$0")/../.." && pwd)
hosts=(); while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue; hosts+=("$h"); done < "$HOSTS"
N=${#hosts[@]}
echo "== model $MODEL from $ORIGIN to all nodes"
for h in "${hosts[@]}"; do [ "$h" = "$ORIGIN" ] && continue
  ssh -n -o BatchMode=yes "$ORIGIN" "rsync -az --exclude 'checkpoint-*' ~/Sichos/asr/$MODEL/ $h:~/Sichos/asr/$MODEL/ && ssh -n $h 'touch ~/Sichos/asr/$MODEL/.harvest_started ~/Sichos/asr/$MODEL/.harvest_done ~/Sichos/asr/$MODEL/.pulled' && echo '  model on $h'" &   # never copy checkpoint dirs (9-17 GB each; filled supermac01 on 2026-09-23); mark the copy so harvest skips it
done; wait
echo "== file shards (audio lives on $ORIGIN under ~/Sichos/new_audio; shards fan out node-to-node)"
IFS='|' read -ra dirs <<< "$YEARS"; all=()
for d in "${dirs[@]}"; do while IFS= read -r f; do all+=("$f"); done < <(cd "$LOCAL" && find "$d" -name '*.mp3' | sort); done
echo "  ${#all[@]} files over $N nodes"
for i in "${!hosts[@]}"; do h=${hosts[$i]}; list="$LOCAL/asr/reports/transcribe-shard-$i.txt"; : > "$list"
  for ((k=i; k<${#all[@]}; k+=N)); do echo "${all[$k]}" >> "$list"; done
  rsync -az "$list" "$ORIGIN:~/Sichos/asr/reports/transcribe-shard-$i.txt"
  if [ "$h" = "$ORIGIN" ]; then
    ssh -n -o BatchMode=yes "$ORIGIN" "cp Sichos/asr/reports/transcribe-shard-$i.txt Sichos/asr/reports/transcribe-list.txt"
  else
    ( ssh -n -o BatchMode=yes "$ORIGIN" "cd Sichos && rsync -az --files-from=asr/reports/transcribe-shard-$i.txt new_audio/ $h:Sichos/new_audio/ && rsync -az asr/reports/transcribe-shard-$i.txt $h:Sichos/asr/reports/transcribe-list.txt" && echo "  shard $i ($(wc -l < "$list" | tr -d ' ') files) on $h" ) &
  fi
done; wait
echo "== transcribing (pipeline synced to every node first)"
for h in "${hosts[@]}"; do
  rsync -az --delete "$LOCAL/asr/pipeline/" "$h:~/Sichos/asr/pipeline/" || { echo "  SKIP $h: pipeline sync failed"; continue; }
  ssh -n -o BatchMode=yes "$h" "test -f ~/Sichos/asr/$MODEL/$MODEL_FILE && test -s ~/Sichos/asr/reports/transcribe-list.txt" || { echo "  SKIP $h: model or shard list missing"; continue; }
  ssh -n -o BatchMode=yes "$h" "cd ~/Sichos/asr && source $VENV/bin/activate && export OMP_NUM_THREADS=4 && sed 's#^#'\$HOME'/Sichos/new_audio/#' reports/transcribe-list.txt > reports/transcribe-abs.txt && rm -f reports/transcribe-part-*; split -l \$(( (\$(wc -l < reports/transcribe-abs.txt) + $WORKERS - 1) / $WORKERS )) -a 1 reports/transcribe-abs.txt reports/transcribe-part- && for p in reports/transcribe-part-*; do nohup python -u pipeline/transcribe_file.py --model $MODEL --out output --mirror-root ~/Sichos/new_audio --skip-seconds 7 $EXTRA_ARGS --list \$p > \$p.log 2>&1 </dev/null & done; echo '  started $WORKERS workers on '\$(hostname -s)"
done
echo "progress: for h in ${hosts[*]}; do ssh -n \$h 'find ~/Sichos/asr/output -name \"*.txt\" | wc -l'; done"
