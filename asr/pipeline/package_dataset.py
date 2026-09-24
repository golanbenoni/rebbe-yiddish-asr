#!/usr/bin/env python3
"""Bundle manifests plus every referenced FLAC into one tar for a GPU machine.

Paths inside the tar stay relative to the Sichos root (asr/data/...), so on the
other machine `tar xf` next to a copy of asr/pipeline/ reproduces the layout the
scripts expect.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", default=str(ROOT / "asr" / "data" / "manifests"))
    ap.add_argument("--splits", nargs="*", default=["train", "dev", "test"])
    ap.add_argument("--out", default=str(ROOT / "asr" / "data" / f"dataset-{time.strftime('%Y%m%d')}.tar"))
    args = ap.parse_args()
    audio = []
    with tarfile.open(args.out, "w") as tar:
        for split in args.splits:
            m = Path(args.manifest_dir) / f"{split}.jsonl"
            tar.add(m, arcname=str(m.relative_to(ROOT)))
            for line in m.read_text().splitlines():
                if line.strip():
                    audio.append(json.loads(line)["audio"])
        for rel in sorted(set(audio)):
            tar.add(ROOT / rel, arcname=rel)
        for extra in ("asr/pipeline", "asr/README.md", "asr/requirements.txt"):
            tar.add(ROOT / extra, arcname=extra)
    size = Path(args.out).stat().st_size / 1e9
    print(f"{args.out}: {len(set(audio))} clips, {size:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
