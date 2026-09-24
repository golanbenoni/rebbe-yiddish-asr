#!/usr/bin/env python3
"""Score an ASR model on manifest clips with the shared normalizer (WER and CER).

Backend: faster-whisper (CTranslate2). --model takes a Hugging Face repo id of a
CT2 conversion (e.g. ivrit-ai/yi-whisper-large-v3-ct2) or a local CT2 directory,
so a fine-tuned checkpoint converted with `ct2-transformers-converter` scores the
same way as the baselines. Writes asr/reports/<name>/report.json and report.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import jiwer
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import scoring_text  # noqa: E402
from decoding import add_suppress_arg, suppress_tokens  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load_rows(manifest_dir: Path, splits: list[str], days: list[str], limit: int) -> list[dict]:
    rows = []
    for split in splits:
        path = manifest_dir / f"{split}.jsonl"
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if days:
        rows = [r for r in rows if r["day"] in days]
    return rows[:limit] if limit else rows


def main() -> int:
    ap = argparse.ArgumentParser()
    add_suppress_arg(ap)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--manifest-dir", default=str(ROOT / "asr" / "data" / "manifests"))
    ap.add_argument("--splits", nargs="*", default=["dev", "test"])
    ap.add_argument("--days", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--language", default="yi")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--initial-prompt", default=None)
    ap.add_argument("--shard", default="0/1", help="i/n: score every n-th clip starting at i; report name gets -shardi")
    ap.add_argument("--backend", default="ct2", choices=["ct2", "hf"], help="ct2 = faster-whisper (CT2 model); hf = transformers generate (HF checkpoint; CUDA/MPS/CPU)")
    ap.add_argument("--batch", type=int, default=8, help="hf backend batch size")
    args = ap.parse_args()

    rows = load_rows(Path(args.manifest_dir), args.splits, args.days, args.limit)
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    if shard_n > 1:
        rows = rows[shard_i::shard_n]; args.name = f"{args.name}-shard{shard_i}"
    if not rows:
        print("no clips selected"); return 1
    t_load = time.time()
    if args.backend == "ct2":
        from faster_whisper import WhisperModel
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

        sup = suppress_tokens(model, args.suppress, args.language)
        def transcribe_batch(audios):
            outs = []
            for audio in audios:
                segs, _ = model.transcribe(audio, language=args.language, task="transcribe", beam_size=args.beam,
                                           vad_filter=False, condition_on_previous_text=False, suppress_tokens=sup,
                                           without_timestamps=True, initial_prompt=args.initial_prompt)
                outs.append(" ".join(s.text.strip() for s in segs))
            return outs
        batch = 1
    else:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        device = args.device if args.device != "cpu" or not torch.cuda.is_available() else "cuda"
        if device == "cpu" and torch.backends.mps.is_available():
            device = "mps"
        dtype = torch.float16 if device == "cuda" else torch.float32
        proc = WhisperProcessor.from_pretrained(args.model, language=args.language, task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

        def transcribe_batch(audios):
            feats = proc.feature_extractor(audios, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
            with torch.no_grad():
                ids = model.generate(feats, language=args.language, task="transcribe", num_beams=args.beam, max_new_tokens=225)
            return [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
        batch = args.batch
    print(f"loaded {args.model} ({args.backend}) in {time.time()-t_load:.0f}s; scoring {len(rows)} clips", flush=True)

    out, audio_sec, t0 = [], 0.0, time.time()
    for b in range(0, len(rows), batch):
        chunk = rows[b:b + batch]
        audios = []
        for r in chunk:
            audio, sr = sf.read(ROOT / r["audio"], dtype="float32")
            audio_sec += len(audio) / sr; audios.append(audio)
        hyps_b = transcribe_batch(audios)
        for r, hyp in zip(chunk, hyps_b):
            ref_n, hyp_n = scoring_text(r["text"]), scoring_text(hyp)
            w = jiwer.process_words(ref_n, hyp_n)
            out.append({"id": r["id"], "day": r["day"], "split": r["split"], "duration": r["duration"],
                        "ref": r["text"], "hyp": hyp, "wer": round(w.wer, 4),
                        "S": w.substitutions, "D": w.deletions, "I": w.insertions, "n_ref": len(ref_n.split())})
        i = len(out)
        if i % 20 < batch or i == len(rows):
            n = sum(o["n_ref"] for o in out); e = sum(o["S"] + o["D"] + o["I"] for o in out)
            print(f"  {i}/{len(rows)} clips  running WER {e/max(1,n):.3f}  ({audio_sec/(time.time()-t0):.2f}x realtime)", flush=True)

    refs = [scoring_text(o["ref"]) for o in out]; hyps = [scoring_text(o["hyp"]) for o in out]
    per_day = defaultdict(lambda: {"n_ref": 0, "err": 0, "clips": 0})
    for o in out:
        d = per_day[o["day"]]; d["n_ref"] += o["n_ref"]; d["err"] += o["S"] + o["D"] + o["I"]; d["clips"] += 1
    summary = {
        "name": args.name, "model": args.model, "backend": args.backend, "language": args.language, "beam": args.beam,
        "splits": args.splits, "suppress": args.suppress, "clips": len(out), "audio_hours": round(audio_sec / 3600, 3),
        "ref_words": sum(o["n_ref"] for o in out),
        "wer": round(jiwer.wer(refs, hyps), 4), "cer": round(jiwer.cer(refs, hyps), 4),
        "substitutions": sum(o["S"] for o in out), "deletions": sum(o["D"] for o in out),
        "insertions": sum(o["I"] for o in out),
        "wall_seconds": round(time.time() - t0, 1), "realtime_factor": round(audio_sec / (time.time() - t0), 2),
        "per_day": {k: {"clips": v["clips"], "wer": round(v["err"] / max(1, v["n_ref"]), 4)} for k, v in sorted(per_day.items())},
    }
    rep = ROOT / "asr" / "reports" / args.name
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "report.json").write_text(json.dumps({"summary": summary, "clips": out}, ensure_ascii=False, indent=1))
    worst = sorted(out, key=lambda o: -o["wer"])[:10]
    md = [f"# {args.name}", "", f"model: `{args.model}`  language: {args.language}  beam: {args.beam}", "",
          f"clips: {summary['clips']}  audio: {summary['audio_hours']} h  ref words: {summary['ref_words']}", "",
          f"**WER {summary['wer']:.3f}   CER {summary['cer']:.3f}**  (S {summary['substitutions']} / D {summary['deletions']} / I {summary['insertions']})",
          f"speed: {summary['realtime_factor']}x realtime on {args.device}/{args.compute_type}", "",
          "| day | clips | WER |", "|---|---:|---:|"]
    md += [f"| {k} | {v['clips']} | {v['wer']:.3f} |" for k, v in summary["per_day"].items()]
    md += ["", "## Worst clips", ""]
    for o in worst:
        md += [f"### {o['id']}  WER {o['wer']:.2f}", f"- ref: {o['ref']}", f"- hyp: {o['hyp']}", ""]
    (rep / "report.md").write_text("\n".join(md))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_day"}, ensure_ascii=False, indent=1))
    print("per day:", {k: v["wer"] for k, v in summary["per_day"].items()})
    print("report ->", rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
