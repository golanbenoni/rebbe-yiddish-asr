#!/usr/bin/env python3
"""Average the weights of several Hugging Face Whisper checkpoints (same architecture).

Used by cluster/train_dp_rounds.sh: each machine trains on its data shard for a
round, the checkpoints are averaged here, and the average seeds the next round
(periodic model averaging, a simple data-parallel scheme that needs only shared
storage, no RDMA). Also useful for a final "soup" of several good runs.

  python average_checkpoints.py --out runs/dp/round_3/avg runs/dp/round_3/node_*
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def load_state(ckpt: Path) -> dict:
    index = ckpt / "model.safetensors.index.json"
    files = sorted({ckpt / v for v in json.loads(index.read_text())["weight_map"].values()}) if index.exists() else [ckpt / "model.safetensors"]
    state = {}
    for f in files:
        state.update(load_file(str(f)))
    return state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None, help="optional per-checkpoint weights")
    args = ap.parse_args()
    ckpts = [Path(c) for c in args.checkpoints]
    w = args.weights or [1.0] * len(ckpts)
    assert len(w) == len(ckpts)
    total = sum(w)
    avg = None
    for ck, wi in zip(ckpts, w):
        st = load_state(ck)
        if avg is None:
            avg = {k: v.to(torch.float32) * (wi / total) for k, v in st.items()}
        else:
            for k, v in st.items():
                avg[k] += v.to(torch.float32) * (wi / total)
        print(f"  + {ck} (weight {wi/total:.3f}, {len(st)} tensors)")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ref = load_state(ckpts[0])
    save_file({k: v.to(ref[k].dtype).contiguous() for k, v in avg.items()}, str(out / "model.safetensors"), metadata={"format": "pt"})
    for f in ckpts[0].iterdir():
        if f.is_file() and f.suffix in (".json", ".txt") and not f.name.startswith("model") and f.name != "training_args.bin":
            shutil.copy(f, out / f.name)
    print(f"averaged {len(ckpts)} checkpoints -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
