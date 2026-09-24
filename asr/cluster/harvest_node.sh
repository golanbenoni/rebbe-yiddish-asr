#!/usr/bin/env bash
# Runs ON a node: for every finished run (final save present, no trainer on it, not yet harvested) merge LoRA
# if needed, convert to CTranslate2 int8, score dev+test clips and long-form test days (5 shards each), then score
# again through the HF GPU path (reports hf-<name>, the production decoder since 2026-09-21).
# Idempotent via marker files in runs/<name>/. Called by cluster/harvest.sh; safe to run repeatedly.
cd ~/Sichos/asr || exit 0
for d in runs/*/; do
  n=$(basename "$d"); case "$n" in smoke-test|*-ct2|*-merged|v0-step1000*|calib-*) continue;; esac
  final=""; [ -f "$d/model.safetensors" ] && final=full; [ -f "$d/adapter_model.safetensors" ] && final=lora
  [ -z "$final" ] && continue
  [ -f "$d/.harvest_started" ] && continue
  [ -f "reports/train-$n.log" ] || continue   # only runs trained ON this node; staged copies of other nodes' models are skipped
  pgrep -f "runs/$n " >/dev/null && continue
  touch "$d/.harvest_started"; echo "  $(hostname -s): harvesting $n ($final)"
  nohup bash -c '
    n="$1"; final="$2"
    source .venv-train/bin/activate
    if [ "$final" = lora ]; then
      base=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))[\"base_model_name_or_path\"])" "runs/$n/adapter_config.json")
      python -u pipeline/finetune.py --merge-lora "runs/$n" --base-model "$base" > "reports/harvest-$n.log" 2>&1; SRC="runs/$n-merged"
    else SRC="runs/$n"; fi
    if [ ! -f "$SRC/preprocessor_config.json" ]; then
      snap=$(ls -d ~/.cache/huggingface/hub/models--ivrit-ai--yi-whisper-large-v3*/snapshots/*/ | head -1); cp "$snap/preprocessor_config.json" "$SRC/" 2>/dev/null
    fi
    ct2-transformers-converter --model "$SRC" --output_dir "runs/$n-ct2" --quantization int8 --copy_files tokenizer.json preprocessor_config.json >> "reports/harvest-$n.log" 2>&1
    source .venv/bin/activate; export OMP_NUM_THREADS=4
    for w in 0 1 2 3 4; do python -u pipeline/evaluate.py --model "runs/$n-ct2" --name "v1-$n" --splits dev test --shard $w/5 > "reports/harvest-$n-clip$w.log" 2>&1 & done; wait
    for w in 0 1 2 3 4; do python -u pipeline/evaluate_longform.py --model "runs/$n-ct2" --name "v1-$n" --splits-file data/splits.json --shard $w/5 > "reports/harvest-$n-lf$w.log" 2>&1 & done; wait
    # production decoder (2026-09-21): the HF fp32 model on the GPU, VAD windows, no timestamps -> reports hf-<name>[-longform]
    source .venv-train/bin/activate
    python -u pipeline/evaluate.py --backend hf --model "$SRC" --name "hf-$n" --splits dev test --batch 8 > "reports/harvest-$n-hfclip.log" 2>&1
    python -u pipeline/evaluate_longform.py --backend hf --model "$SRC" --name "hf-$n" --splits-file data/splits.json > "reports/harvest-$n-hflf.log" 2>&1
    rm -rf runs/$n/checkpoint-*
    touch "runs/$n/.harvest_done"
  ' _ "$n" "$final" </dev/null >/dev/null 2>&1 &
done
for d in runs/*/; do n=$(basename "$d"); [ -f "$d/.harvest_done" ] && [ ! -f "$d/.pulled" ] && echo "  $(hostname -s): READY $n"; done
exit 0
