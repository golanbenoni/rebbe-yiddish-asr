#!/usr/bin/env python3
"""Finalize the pseudo-label manifest and merge it with the human-timed training set.

build_pseudo_dataset.py (run on the node that holds the new audio) writes train.pseudo.jsonl with the model's raw
text. This script, run here after pulling that file: normalizes the text with normalize.training_text (same
convention as the real targets), fixes the audio path prefix (paths must be relative to the Sichos root, i.e.
start with asr/), adds day/source_year, drops rows with fewer than --min-words words or repetitive text (decoder\nloops, hf_longform.is_repetitive), and writes
train.clean_pseudo.jsonl = train.clean.jsonl + pseudo rows.

  python pipeline/merge_pseudo_manifest.py            # data/manifests/train.pseudo.jsonl -> train.clean_pseudo.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_longform import is_repetitive  # noqa: E402
from normalize import training_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data" / "manifests"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo", default=str(MAN / "train.pseudo.jsonl"))
    ap.add_argument("--clean", default=str(MAN / "train.clean.jsonl"))
    ap.add_argument("--out", default=str(MAN / "train.clean_pseudo.jsonl"))
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--min-logprob", type=float, default=None, help="optionally tighten the confidence filter here")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.pseudo) if l.strip()]
    kept, dropped, fixed, loops = [], 0, 0, 0
    for r in rows:
        if not r["audio"].startswith("asr/"):
            r["audio"] = "asr/" + r["audio"]; fixed += 1
        r["text"] = training_text(r["text"])
        r["day"] = r["source_mp3"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        r["source_year"] = r["source_mp3"].split("/")[0]
        if len(r["text"].split()) < args.min_words or (args.min_logprob is not None and r["min_logprob"] < args.min_logprob):
            dropped += 1; continue
        if is_repetitive(r["text"], r["duration"]):   # decoder loops must not become training labels
            dropped += 1; loops += 1; continue
        kept.append(r)
    Path(args.pseudo).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    clean = Path(args.clean).read_text()
    Path(args.out).write_text((clean if clean.endswith("\n") else clean + "\n") + "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    hrs = sum(r["duration"] for r in kept) / 3600
    q = sum(r["text"].count('"') for r in kept)
    print(f"pseudo rows kept {len(kept)} ({hrs:.1f} h, {q:,} gershayim), dropped {dropped} (of which repetitive {loops}), audio paths fixed {fixed}; "
          f"{args.out}: {sum(1 for l in open(args.out) if l.strip())} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
