#!/usr/bin/env python3
"""Flag site-timed days whose clips almost all failed the baseline scan (the site timing file does
not match the audio, e.g. swapped parts of one farbrengen). Sets `sync_untrusted: true` in the day
JSON so build_dataset.py uses `sync_local` for them (run align_day.py --days ... first).

  python flag_untrusted_site_timings.py --dropped data/manifests/train.clean.jsonl.dropped.tsv --manifest data/manifests/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dropped", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--min-drop-fraction", type=float, default=0.8)
    args = ap.parse_args()
    total = Counter(); dropped = Counter(); source = {}
    for line in Path(args.manifest).read_text().splitlines():
        if line.strip():
            r = json.loads(line); key = f"{r['year']}/{r['day']}"; total[key] += 1; source[key] = r["timing_source"]
    for line in Path(args.dropped).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            dropped[parts[0].rsplit("/", 1)[0]] += 1
    flagged = []
    for key, n in total.items():
        if source.get(key) == "site" and dropped[key] / n >= args.min_drop_fraction:
            year, stem = key.split("/", 1); path = ROOT / "asr" / "data" / "days" / year / f"{stem}.json"
            rec = json.loads(path.read_text()); rec["sync_untrusted"] = True; rec["sync_untrusted_reason"] = f"{dropped[key]}/{n} clips dropped in baseline scan"
            path.write_text(json.dumps(rec, ensure_ascii=False)); flagged.append(key)
    print(f"flagged {len(flagged)} site-timed days as untrusted: {flagged}")
    print("next: python pipeline/align_day.py --year <y> --days <stems> --force   then rebuild")
    return 0


if __name__ == "__main__":
    sys.exit(main())
