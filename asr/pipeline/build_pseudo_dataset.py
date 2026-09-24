#!/usr/bin/env python3
"""Turn Phase 7 transcripts into pseudo-labeled training clips (self-training data).

For every <stem>.json written by transcribe_file.py (segments with start/end/text/avg_logprob), cut the
corresponding audio segments (merged up to --max-seconds) from the source mp3 into FLAC clips and write a
manifest with the model's text as target. Keep only confident segments (--min-logprob) of sane length.
Runs where the audio lives (a node with ~/Sichos/new_audio) or here.

  python build_pseudo_dataset.py --output-dir output --audio-root ../new_audio --out data/manifests/train.pseudo.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio import SR, decode_16k_mono  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(ROOT / "asr" / "output"), help="Phase 7 outputs (mirrored year folders)")
    ap.add_argument("--audio-root", default=str(ROOT), help="root containing the year folders with the mp3s")
    ap.add_argument("--clips-dir", default=str(ROOT / "asr" / "data" / "segments_pseudo"))
    ap.add_argument("--out", default=str(ROOT / "asr" / "data" / "manifests" / "train.pseudo.jsonl"))
    ap.add_argument("--max-seconds", type=float, default=28.0)
    ap.add_argument("--min-seconds", type=float, default=2.0)
    ap.add_argument("--min-logprob", type=float, default=-0.6)
    ap.add_argument("--max-no-speech", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    jsons = sorted(Path(args.output_dir).rglob("*.json"))
    if args.limit:
        jsons = jsons[: args.limit]
    rows = []; kept = dropped = 0
    for jf in jsons:
        d = json.loads(jf.read_text())
        rel = jf.relative_to(args.output_dir).with_suffix(".mp3")
        mp3 = Path(args.audio_root) / rel
        if not mp3.exists():
            print(f"  missing audio for {rel}"); continue
        segs = [s for s in d["segments"] if s["avg_logprob"] >= args.min_logprob and s["no_speech_prob"] <= args.max_no_speech and s["text"].strip()]
        dropped += len(d["segments"]) - len(segs)
        # merge consecutive segments up to max_seconds
        chunks, cur = [], None
        for s in segs:
            if cur and s["end"] - cur["start"] <= args.max_seconds and s["start"] - cur["end"] < 1.5:
                cur["end"] = s["end"]; cur["text"] += " " + s["text"].strip(); cur["n"] += 1; cur["lp"] = min(cur["lp"], s["avg_logprob"])
            else:
                if cur: chunks.append(cur)
                cur = {"start": s["start"], "end": s["end"], "text": s["text"].strip(), "n": 1, "lp": s["avg_logprob"]}
        if cur: chunks.append(cur)
        chunks = [c for c in chunks if c["end"] - c["start"] >= args.min_seconds]
        if not chunks:
            continue
        audio = decode_16k_mono(mp3); total = len(audio) / SR
        out_dir = Path(args.clips_dir).resolve() / rel.parent / rel.stem; out_dir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(chunks):
            a, b = int(c["start"] * SR), int(min(c["end"], total) * SR)
            if b - a < int(args.min_seconds * SR):
                continue
            clip = out_dir / f"{i:03d}_{int(c['start']*100):07d}.flac"
            if not clip.exists():
                sf.write(clip, audio[a:b], SR, format="FLAC")
            rows.append({"id": f"pseudo/{rel.parent}/{rel.stem}/{i:03d}", "audio": str(clip.relative_to(ROOT)) if clip.is_relative_to(ROOT) else str(clip),
                         "start": round(c["start"], 3), "end": round(c["end"], 3), "duration": round((b - a) / SR, 3),
                         "text": c["text"], "n_segments": c["n"], "min_logprob": round(c["lp"], 3),
                         "source_mp3": str(rel), "split": "train", "timing_source": "pseudo"})
            kept += 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    hrs = sum(r["duration"] for r in rows) / 3600
    print(f"pseudo-labeled: {kept} clips, {hrs:.1f} h, from {len(jsons)} files; segments dropped for confidence: {dropped}; -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
