#!/usr/bin/env python3
"""Fine-tune a Whisper checkpoint on the Daily Sicha clips (Hugging Face Transformers).

Default base model is ivrit-ai/yi-whisper-large-v3 (Whisper large-v3 already
adapted to Yiddish by ivrit.ai), the best open starting point measured on this
audio. Features are computed on the fly from the FLAC clips, so no feature cache
is needed. Two modes:

  full fine-tune (recommended on one 80 GB GPU):
    python finetune.py --output runs/yi-large-v3-ft --epochs 3 --lr 1e-5 --batch 16 --grad-accum 2 --bf16
  LoRA (fits a 24-48 GB GPU):
    python finetune.py --output runs/yi-large-v3-lora --lora --lr 1e-4 --batch 8 --grad-accum 4 --bf16
  smoke test (CPU/MPS, whisper-tiny, 2 steps) to prove the data path:
    python finetune.py --smoke-test

After training, convert for evaluate.py / faster-whisper:
    ct2-transformers-converter --model runs/yi-large-v3-ft --output_dir runs/yi-large-v3-ft-ct2 --quantization float16
(For LoRA, merge first: see --merge-lora.)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import faulthandler
import jiwer
import numpy as np
import soundfile as sf
import torch

faulthandler.enable()  # a killed/segfaulting run leaves a stack trace in the log

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import scoring_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "asr" / "data" / "manifests"


def read_manifest(path: Path, limit: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return rows[:limit] if limit else rows


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], processor, max_label_len: int = 440):
        self.rows, self.processor, self.max_label_len = rows, processor, max_label_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        audio, sr = sf.read(ROOT / r["audio"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        feats = self.processor.feature_extractor(audio, sampling_rate=sr, return_tensors="np").input_features[0]
        labels = self.processor.tokenizer(r["text"]).input_ids[: self.max_label_len]
        return {"input_features": feats, "labels": labels}


@dataclass
class Collator:
    processor: object
    decoder_start_token_id: int

    def __call__(self, features):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="ivrit-ai/yi-whisper-large-v3")
    ap.add_argument("--manifest-dir", default=str(MANIFESTS))
    ap.add_argument("--train-manifest", default="train.jsonl", help="file name inside --manifest-dir")
    ap.add_argument("--shard", default="0/1", help="i/n: train on every n-th clip starting at i (one shard per machine)")
    ap.add_argument("--init-from", default=None, help="load weights from this checkpoint dir instead of --base-model")
    ap.add_argument("--lr-scheduler", default="linear", help="linear | constant | cosine (constant for periodic-averaging rounds)")
    ap.add_argument("--output", default=str(ROOT / "asr" / "runs" / "yi-large-v3-ft"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--eval-limit", type=int, default=0, help="cap dev clips used during training eval")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--language", default="yi")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--merge-lora", default=None, help="path of a LoRA run to merge into a plain checkpoint, then exit")
    ap.add_argument("--resume", action="store_true", help="resume from the latest checkpoint-* in --output")
    ap.add_argument("--seed", type=int, default=42, help="training seed (Trainer default 42); vary it to measure run-to-run variance")
    args = ap.parse_args()

    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                              WhisperProcessor)

    if args.merge_lora:
        from peft import PeftModel
        base = WhisperForConditionalGeneration.from_pretrained(args.base_model)
        merged = PeftModel.from_pretrained(base, args.merge_lora).merge_and_unload()
        merged.save_pretrained(args.merge_lora + "-merged")
        WhisperProcessor.from_pretrained(args.base_model).save_pretrained(args.merge_lora + "-merged")
        print("merged ->", args.merge_lora + "-merged"); return 0

    if args.smoke_test:
        args.base_model, args.output = "openai/whisper-tiny", str(ROOT / "asr" / "runs" / "smoke-test")
        args.max_steps, args.batch, args.grad_accum, args.num_workers = 2, 2, 1, 0
        args.eval_steps = args.save_steps = 2; args.eval_limit = 4; args.warmup_steps = 0

    processor = WhisperProcessor.from_pretrained(args.base_model, language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.init_from or args.base_model)
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    if args.freeze_encoder:
        model.freeze_encoder()
    if args.lora:
        from peft import LoraConfig, get_peft_model
        if args.gradient_checkpointing:
            model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
        model.print_trainable_parameters()

    train_rows = read_manifest(Path(args.manifest_dir) / args.train_manifest, 16 if args.smoke_test else 0)
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    train_rows = train_rows[shard_i::shard_n]
    dev_rows = read_manifest(Path(args.manifest_dir) / "dev.jsonl", args.eval_limit)
    print(f"train clips {len(train_rows)} ({sum(r['duration'] for r in train_rows)/3600:.2f} h), dev clips {len(dev_rows)}")
    train_ds, dev_ds = ClipDataset(train_rows, processor), ClipDataset(dev_rows, processor)

    def compute_metrics(pred):
        ids = pred.predictions
        ids = np.where(ids == -100, processor.tokenizer.pad_token_id, ids)
        label_ids = np.where(pred.label_ids == -100, processor.tokenizer.pad_token_id, pred.label_ids)
        hyps = [scoring_text(t) for t in processor.tokenizer.batch_decode(ids, skip_special_tokens=True)]
        refs = [scoring_text(t) for t in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
        pairs = [(r, h) for r, h in zip(refs, hyps) if r]
        return {"wer": jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]) if pairs else 1.0}

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output, per_device_train_batch_size=args.batch, per_device_eval_batch_size=max(1, args.batch // 2),
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs, max_steps=args.max_steps, bf16=args.bf16, fp16=args.fp16,
        lr_scheduler_type=args.lr_scheduler,
        gradient_checkpointing=args.gradient_checkpointing, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=3, logging_steps=25,
        predict_with_generate=True, generation_max_length=225, load_best_model_at_end=not args.smoke_test,
        metric_for_best_model="wer", greater_is_better=False, remove_unused_columns=False,
        dataloader_num_workers=args.num_workers, report_to="none", use_cpu=args.smoke_test,
        label_names=["labels"], seed=args.seed,
    )
    from transformers import TrainerCallback

    class MemoryLog(TrainerCallback):
        """Log accelerator memory after the first steps: MPS runs that exceed physical RAM thrash silently otherwise."""
        def on_step_end(self, args, state, control, **kw):
            if state.global_step in (1, 2, 5, 10) or state.global_step % 200 == 0:
                if torch.backends.mps.is_available():
                    print(f"[mem] step {state.global_step}: mps driver {torch.mps.driver_allocated_memory()/2**30:.1f} GB", flush=True)
                elif torch.cuda.is_available():
                    print(f"[mem] step {state.global_step}: cuda max {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)

    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                             data_collator=Collator(processor, model.config.decoder_start_token_id),
                             compute_metrics=compute_metrics, processing_class=processor, callbacks=[MemoryLog()])
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    metrics = trainer.evaluate()
    print("final dev metrics:", metrics)
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    processor.feature_extractor.save_pretrained(args.output)  # writes preprocessor_config.json for ct2-transformers-converter
    processor.tokenizer.save_pretrained(args.output)
    (Path(args.output) / "train_config.json").write_text(json.dumps(vars(args), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
