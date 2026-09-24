#!/usr/bin/env bash
# Data-parallel training by periodic model averaging (needs only the shared volume).
# Round r: every node trains STEPS steps on its own shard of the clips, starting from the
# previous round's average; the coordinator averages the N checkpoints and scores the average
# on dev. With STEPS=200 and 8 nodes, one round processes 8*200*effective-batch clips.
#   cluster/train_dp_rounds.sh cluster/hosts.txt 6            (6 rounds)
set -euo pipefail
HOSTS=${1:?hosts file}; ROUNDS=${2:-6}
LOCAL=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${ROOT:-\$HOME/Sichos}
BASE=${BASE:-ivrit-ai/yi-whisper-large-v3}
STEPS=${STEPS:-200}; LR=${LR:-1e-5}; BATCH=${BATCH:-8}; ACCUM=${ACCUM:-2}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-train.clean.jsonl}
hosts=(); while IFS= read -r h; do [[ -z "$h" || "$h" == \#* ]] && continue; hosts+=("$h"); done < "$HOSTS"
N=${#hosts[@]}; init=""
for ((r=1; r<=ROUNDS; r++)); do
  echo "=== round $r/$ROUNDS: $N nodes x $STEPS steps ==="
  for i in "${!hosts[@]}"; do
    h=${hosts[$i]}; out="runs/dp/round_$r/node_$i"
    warm=$([ "$r" -eq 1 ] && echo 50 || echo 0)
    ssh "$h" "cd $ROOT/asr && source .venv-train/bin/activate && export PYTORCH_ENABLE_MPS_FALLBACK=1 && nohup python pipeline/finetune.py --base-model $BASE ${init:+--init-from $init} --shard $i/$N --train-manifest $TRAIN_MANIFEST --max-steps $STEPS --warmup-steps $warm --lr $LR --lr-scheduler constant --batch $BATCH --grad-accum $ACCUM --gradient-checkpointing --eval-steps 100000 --save-steps 100000 --eval-limit 16 --num-workers 4 --output $out > reports/dp-round$r-node$i.log 2>&1 & echo started node $i on \$(hostname)"
  done
  # wait for every node's checkpoint
  for i in "${!hosts[@]}"; do
    until ssh -n -o BatchMode=yes "${hosts[$i]}" "test -f ~/Sichos/asr/runs/dp/round_$r/node_$i/model.safetensors -o -f ~/Sichos/asr/runs/dp/round_$r/node_$i/model.safetensors.index.json"; do sleep 60; done
    rsync -az "${hosts[$i]}:~/Sichos/asr/runs/dp/round_$r/node_$i/" "$LOCAL/asr/runs/dp/round_$r/node_$i/"
  done
  init="runs/dp/round_$r/avg"
  ssh "${hosts[0]}" "cd $ROOT/asr && source .venv-train/bin/activate && python pipeline/average_checkpoints.py --out $init runs/dp/round_$r/node_* && ct2-transformers-converter --model $init --output_dir $init-ct2 --quantization int8 --copy_files tokenizer.json preprocessor_config.json >/dev/null && python pipeline/evaluate.py --model $init-ct2 --name dp-round$r --splits dev | tail -4"
done
echo "final model: $ROOT/asr/$init  (convert with ct2-transformers-converter for transcribe_file.py)"
