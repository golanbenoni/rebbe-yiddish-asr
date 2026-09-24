#!/usr/bin/env python3
"""Drop training clips whose baseline transcription disagrees wildly with the reference.

The site's segment timings are machine-aligned; a clip whose audio does not match
its text (shifted boundaries, skipped passages) would teach the model nonsense.
Score the train split with evaluate.py first, then filter on that report:

  python evaluate.py --model ivrit-ai/yi-whisper-large-v3-ct2 --name train-scan --splits train --device cuda --compute-type float16
  python filter_manifest.py --manifest data/manifests/train.jsonl --report reports/train-scan/report.json \
      --out data/manifests/train.clean.jsonl

Rules (defaults are deliberately loose: the base model itself has ~50% WER here):
  keep if CER <= --max-cer (0.60) and the hypothesis/reference length ratio is within --len-ratio (0.5..2.0).
Dropped clips are listed in <out>.dropped.tsv with the reason, for spot checks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import scoring_text  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--report", required=True, nargs="+", help="one or more evaluate.py report.json files (e.g. shards)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-cer", type=float, default=0.60)
    ap.add_argument("--len-ratio", type=float, nargs=2, default=(0.5, 2.0))
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    scored = {}
    for rep in args.report:
        scored.update({c["id"]: c for c in json.loads(Path(rep).read_text())["clips"]})
    kept, dropped = [], []
    for r in rows:
        c = scored.get(r["id"])
        if c is None:
            kept.append(r); continue  # unscored clips pass through
        ref, hyp = scoring_text(c["ref"]), scoring_text(c["hyp"])
        if not ref:
            dropped.append((r["id"], "empty reference", "")); continue
        cer = jiwer.cer(ref, hyp) if hyp else 1.0
        ratio = (len(hyp.split()) / max(1, len(ref.split())))
        if cer > args.max_cer:
            dropped.append((r["id"], f"cer {cer:.2f}", c["hyp"][:80]))
        elif not (args.len_ratio[0] <= ratio <= args.len_ratio[1]):
            dropped.append((r["id"], f"length ratio {ratio:.2f}", c["hyp"][:80]))
        else:
            r = dict(r, baseline_cer=round(cer, 4)); kept.append(r)
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    Path(args.out + ".dropped.tsv").write_text("".join(f"{i}\t{why}\t{h}\n" for i, why, h in dropped))
    hours = sum(r["duration"] for r in kept) / 3600
    print(f"kept {len(kept)} clips ({hours:.2f} h), dropped {len(dropped)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
