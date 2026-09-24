#!/usr/bin/env bash
# Runs ON a node: score a CT2 model on dev+test clips (SHARDS parallel shards) and then on the long-form test
# days, with a decoding option (see pipeline/decoding.py). Reports land in ~/Sichos/asr/reports/<name>-shardN[-longform]/.
#   cluster/rescore_node.sh runs/<m>-ct2 aq-<m> allow-quotes
cd ~/Sichos/asr || exit 1
M=${1:?ct2 dir or HF id}; NAME=${2:?report name}; SUP=${3:-allow-quotes}; SH=${SHARDS:-5}
source .venv/bin/activate; export OMP_NUM_THREADS=4
for ((w=0; w<SH; w++)); do python -u pipeline/evaluate.py --model "$M" --name "$NAME" --splits dev test --shard $w/$SH --suppress "$SUP" > "reports/rescore-$NAME-clip$w.log" 2>&1 & done; wait
for ((w=0; w<SH; w++)); do python -u pipeline/evaluate_longform.py --model "$M" --name "$NAME" --splits-file data/splits.json --shard $w/$SH --suppress "$SUP" > "reports/rescore-$NAME-lf$w.log" 2>&1 & done; wait
echo "RESCORE DONE $NAME $(date)" > "reports/rescore-$NAME.done"
