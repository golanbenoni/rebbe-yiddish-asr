#!/usr/bin/env bash
# Push data + pipeline to every node (rsync over SSH, all nodes in parallel).
#   cluster/push.sh cluster/hosts.txt
# Sends: asr/pipeline, requirements, asr/data/days (day records), asr/data/audio (5783-85 site mp3s),
# the local year folders that have paired text (DS 5786, DS_5787), and the cached CT2 models.
set -euo pipefail
HOSTS=${1:?hosts file}
ROOT_REMOTE=${ROOT_REMOTE:-Sichos}          # relative to the remote $HOME
LOCAL=$(cd "$(dirname "$0")/../.." && pwd)  # the Sichos folder
hosts=(); while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue; hosts+=("$h"); done < "$HOSTS"
for h in "${hosts[@]}"; do
  (
    ssh "$h" "mkdir -p ~/$ROOT_REMOTE/asr/data ~/.cache/huggingface/hub"
    rsync -az --delete "$LOCAL/asr/pipeline/" "$h:~/$ROOT_REMOTE/asr/pipeline/"
    rsync -az "$LOCAL/asr/requirements.txt" "$LOCAL/asr/requirements-align.txt" "$h:~/$ROOT_REMOTE/asr/"
    rsync -az "$LOCAL/asr/data/days/" "$h:~/$ROOT_REMOTE/asr/data/days/"
    rsync -az "$LOCAL/asr/data/audio/" "$h:~/$ROOT_REMOTE/asr/data/audio/"
    rsync -az "$LOCAL/DS 5786/" "$h:~/$ROOT_REMOTE/DS\ 5786/"      # remote path is shell-parsed: escape the space
    rsync -az "$LOCAL/DS_5787/" "$h:~/$ROOT_REMOTE/DS_5787/"
    for m in ${MODELS:-models--ivrit-ai--yi-whisper-large-v3-turbo-ct2}; do   # MODELS="a b" to push more
      [ -d "$HOME/.cache/huggingface/hub/$m" ] && rsync -az "$HOME/.cache/huggingface/hub/$m/" "$h:~/.cache/huggingface/hub/$m/"
    done
    echo "pushed to $h"
  ) &
done
wait
echo "push complete"
