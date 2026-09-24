#!/usr/bin/env python3
"""Audit Phase 7 transcripts: flag files whose output is implausibly short or long for their audio, and (when a
reference text exists in data/pdf_text/<year>/) files whose word count differs from the reference by more than
--ratio. The 2026-09-21 "0 empty files" check missed a 31-word transcript of a 638 s recording.

  python pipeline/audit_transcripts.py                 # reads asr/output/**/*.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = {"DS 5775 CHUL": 5775, "DS 5776": 5776, "DS 5777": 5777, "DS 5778": 5778, "Sicha Yomis 5779 audio": 5779,
             "Sicha Yomis 5780 audio": 5780, "Sicha Yomis 5781 audio": 5781}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "asr" / "output"))
    ap.add_argument("--min-wps", type=float, default=1.0, help="flag below this many words per second of audio (corpus median 2.07)")
    ap.add_argument("--max-wps", type=float, default=3.5)
    ap.add_argument("--ratio", type=float, default=0.15, help="flag when |hyp/ref - 1| exceeds this (reference from data/pdf_text)")
    ap.add_argument("--min-coverage", type=float, default=0.9, help="flag when the decoded windows cover less than this share of the audio (VAD dropped speech)")
    args = ap.parse_args()
    refs = {}
    for f in (ROOT / "asr" / "data" / "pdf_text").glob("*/[0-9]*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        for mp3 in r.get("mp3", []):
            refs[Path(mp3).stem + "|" + Path(mp3).parts[0]] = r.get("n_words", 0)
    flagged = []; n = 0
    for f in sorted(glob.glob(f"{args.output}/**/*.json", recursive=True)):
        d = json.loads(open(f, encoding="utf-8").read()); n += 1
        words = sum(len(s["text"].split()) for s in d["segments"]); secs = d.get("audio_seconds") or 1
        wps = words / secs
        cover = sum(s["end"] - s["start"] for s in d["segments"]) / secs
        rel = Path(f).relative_to(args.output); folder = rel.parts[0]; stem = rel.stem
        why = []
        if wps < args.min_wps: why.append(f"only {wps:.2f} words/s")
        if wps > args.max_wps: why.append(f"{wps:.2f} words/s (loop?)")
        if secs > 120 and cover < args.min_coverage: why.append(f"windows cover {cover:.2f} of the audio")
        ref = refs.get(stem + "|" + folder)
        if ref and abs(words / max(1, ref) - 1) > args.ratio: why.append(f"hyp {words} vs ref {ref} words")
        if why: flagged.append((str(rel), why))
    print(f"audited {n} transcripts; flagged {len(flagged)}")
    for rel, why in flagged: print(f"  {rel}: {'; '.join(why)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
