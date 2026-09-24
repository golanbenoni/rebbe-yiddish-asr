#!/usr/bin/env python3
"""Full-file WER per day: the product metric.

evaluate.py scores clips cut on known timings. This script decodes each day's whole local
mp3 the way transcribe_file.py does (VAD chunking, no timings) and scores the result against
the day's entire Yiddish hanacha body. Use it on the test days of data/splits.json for every
model that matters; it catches long-form failure modes (drift, repetition, dropped passages)
that clip scoring cannot see.

  python evaluate_longform.py --model runs/<best>-ct2 --name <best> --splits-file data/splits.json
  python evaluate_longform.py --model ivrit-ai/yi-whisper-large-v3-turbo-ct2 --name turbo-base --days "020 28-Tishrei 5787"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio import SR, decode_16k_mono  # noqa: E402
from normalize import hanacha_paragraphs, scoring_text, training_text  # noqa: E402
from decoding import add_suppress_arg, suppress_tokens  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "asr" / "data"


def find_day(stem: str) -> Path:
    hits = list((DATA / "days").rglob(f"{stem}.json"))
    if not hits:
        raise FileNotFoundError(stem)
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    add_suppress_arg(ap)
    ap.add_argument("--model", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--days", nargs="*", default=[]); ap.add_argument("--splits-file", default=None)
    ap.add_argument("--split", default="test"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--language", default="yi"); ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--intro-seconds", type=float, default=7.0)
    ap.add_argument("--shard", default="0/1", help="i/n over the selected days; report name gets -shardi")
    ap.add_argument("--backend", default="ct2", choices=["ct2", "hf"], help="ct2 = faster-whisper int8 on CPU; hf = HF fp32 model on MPS/CUDA (pipeline/hf_longform.py)")
    ap.add_argument("--batch", type=int, default=8, help="hf backend: windows per batch")
    ap.add_argument("--max-window", type=float, default=28.0, help="hf backend: max seconds of speech per decoded window")
    ap.add_argument("--windows", default="auto", choices=["vad", "energy", "auto", "fixed"], help="hf backend: window strategy (hf_longform.speech_windows_flagged); auto = VAD windows + confidence-gated loud gaps, production default since 2026-09-23")
    args = ap.parse_args()
    stems = list(args.days)
    if args.splits_file:
        stems += json.loads(Path(args.splits_file).read_text())[args.split]
    if args.limit:
        stems = stems[: args.limit]
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    if shard_n > 1:
        stems = stems[shard_i::shard_n]; args.name = f"{args.name}-shard{shard_i}"
    if not stems:
        print("no days"); return 1
    if args.backend == "hf":
        from hf_longform import HFTranscriber
        model = HFTranscriber(args.model, language=args.language, beam=args.beam, batch=args.batch)
        sup = None; print(f"hf backend on {model.device} ({model.dtype})", flush=True)
    else:
        from faster_whisper import WhisperModel
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
        sup = suppress_tokens(model, args.suppress, args.language)
    rows, refs, hyps, t0, audio_sec = [], [], [], time.time(), 0.0
    for stem in stems:
        rec = json.loads(find_day(stem).read_text())
        paras = hanacha_paragraphs(rec["api"]["sicha"].get("contentHtml") or "")
        ref = training_text(" ".join(paras))
        audio = decode_16k_mono(ROOT / rec["local"]["path"])
        intro = float(((rec.get("sync") or {}).get("diagnostics") or {}).get("hiddenAlignmentIntroEnd") or args.intro_seconds)
        skip = float(rec.get("local_offset_seconds") or 0.0) + intro
        audio = audio[int(skip * SR):]; audio_sec += len(audio) / SR
        if args.backend == "hf":
            segs = model.transcribe(audio, max_seconds=args.max_window, min_silence_ms=400, mode=args.windows)
        else:
            segs, _ = model.transcribe(audio, language=args.language, task="transcribe", beam_size=args.beam,
                                       vad_filter=True, vad_parameters={"min_silence_duration_ms": 400},
                                       condition_on_previous_text=False, suppress_tokens=sup)
        hyp = " ".join(training_text(s.text) for s in segs)
        r, h = scoring_text(ref), scoring_text(hyp)
        w = jiwer.process_words(r, h); c = jiwer.cer(r, h)
        rows.append({"day": stem, "source_year": rec["api"]["sicha"].get("deliveredYear"), "audio_seconds": round(len(audio) / SR, 1),
                     "ref_words": len(r.split()), "hyp_words": len(h.split()), "wer": round(w.wer, 4), "cer": round(c, 4),
                     "S": w.substitutions, "D": w.deletions, "I": w.insertions, "hyp": hyp})
        refs.append(r); hyps.append(h)
        print(f"  {stem:<28} WER {w.wer:.3f}  CER {c:.3f}  ref {len(r.split())} hyp {len(h.split())}  ({audio_sec/(time.time()-t0):.2f}x realtime)", flush=True)
    summary = {"name": args.name, "model": args.model, "backend": args.backend, "suppress": args.suppress if args.backend == "ct2" else "none (hf)", "days": len(rows), "audio_hours": round(audio_sec / 3600, 3),
               "wer": round(jiwer.wer(refs, hyps), 4), "cer": round(jiwer.cer(refs, hyps), 4),
               "realtime_factor": round(audio_sec / (time.time() - t0), 2)}
    rep = ROOT / "asr" / "reports" / f"{args.name}-longform"; rep.mkdir(parents=True, exist_ok=True)
    (rep / "report.json").write_text(json.dumps({"summary": summary, "days": rows}, ensure_ascii=False, indent=1))
    md = [f"# {args.name} (long-form, full files)", "", f"model: `{args.model}`", "",
          f"**WER {summary['wer']:.3f}  CER {summary['cer']:.3f}** over {summary['days']} days, {summary['audio_hours']} h, {summary['realtime_factor']}x realtime", "",
          "| day | source year | WER | CER | ref words | hyp words |", "|---|---|---:|---:|---:|---:|"]
    md += [f"| {r['day']} | {r['source_year']} | {r['wer']:.3f} | {r['cer']:.3f} | {r['ref_words']} | {r['hyp_words']} |" for r in rows]
    (rep / "report.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary, ensure_ascii=False)); print("report ->", rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
