#!/usr/bin/env bash
# Prepare every node in a hosts file for the v5 alignment: current pipeline + day records from this Mac, the archive
# audio years 5775-5781 from the supermac01 hub (~/Sichos/new_audio/<folder> -> symlinked as ~/Sichos/<folder> so
# rec.local.path resolves), and the production CTranslate2 model for anchor transcription.
#   cluster/v5_prepare_nodes.sh cluster/hosts_align.txt
set -u
HOSTS=${1:?hosts file}; HUB=${HUB:-supermac01}; MODEL=${MODEL:-turbo-full-lr3e5-4ep-ct2}
LOCAL=$(cd "$(dirname "$0")/../.." && pwd)
FOLDERS=("DS 5775 CHUL" "DS 5776" "DS 5777" "DS 5778" "Sicha Yomis 5779 audio" "Sicha Yomis 5780 audio" "Sicha Yomis 5781 audio")
# the hub itself: symlinks so its own align/transcribe jobs resolve the paths
for d in "${FOLDERS[@]}"; do ssh -n -o BatchMode=yes "$HUB" "cd ~/Sichos && [ -e \"$d\" ] || ln -s \"new_audio/$d\" \"$d\""; done
for d in "DS 5775 CHUL" "DS 5776" "DS 5777"; do
  n=$(ssh -n -o BatchMode=yes "$HUB" "ls ~/Sichos/new_audio/'$d'/*/*.mp3 2>/dev/null | wc -l | tr -d ' '")
  [ "${n:-0}" -ge 240 ] || { echo "!!! hub $HUB lacks new_audio/$d ($n mp3): stage the audio first"; exit 1; }
done
while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue
  (
    rsync -az --delete "$LOCAL/asr/pipeline/" "$h:~/Sichos/asr/pipeline/"
    rsync -az "$LOCAL/asr/data/days/" "$h:~/Sichos/asr/data/days/"
    if [ "$h" != "$HUB" ]; then
      ssh -n -o BatchMode=yes "$HUB" "rsync -az ~/Sichos/new_audio/ $h:~/Sichos/new_audio/ && rsync -az ~/Sichos/asr/runs/$MODEL/ $h:~/Sichos/asr/runs/$MODEL/"
    fi
    for d in "${FOLDERS[@]}"; do ssh -n -o BatchMode=yes "$h" "cd ~/Sichos && [ -e \"$d\" ] || ln -s \"new_audio/$d\" \"$d\""; done
    ssh -n -o BatchMode=yes "$h" 'cd ~/Sichos && echo "  $(hostname) ready: days $(ls asr/data/days/*/ 2>/dev/null | wc -l | tr -d " "), mp3 5775 $(ls "new_audio/DS 5775 CHUL"/*/*.mp3 2>/dev/null | wc -l | tr -d " ") / 5777 $(ls "new_audio/DS 5777"/*/*.mp3 2>/dev/null | wc -l | tr -d " ") / 5781 $(ls "new_audio/Sicha Yomis 5781 audio"/*/*.mp3 2>/dev/null | wc -l | tr -d " "), model $([ -f asr/runs/'"$MODEL"'/model.bin ] && echo ok || echo MISSING)"'
  ) &
done < "$HOSTS"
wait
echo "=== NODES PREPARED $(date +%H:%M) ==="
