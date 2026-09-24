#!/usr/bin/env python3
"""Cut local Daily Sicha audio into training clips using the site's segment timings.

Reads the cached day JSON (from fetch_days.py), decodes the local mp3 to 16 kHz
mono, merges consecutive timed segments into chunks of at most --max-seconds,
writes each chunk as FLAC under asr/data/segments/<year>/<day>/, and emits
JSONL manifests (all/train/dev/test, split by day, deterministic).

Training text = normalize.training_text(yiddish) (diacritics dropped by default);
the raw text plus the Hebrew and English renderings are kept alongside.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio import SR, decode_16k_mono  # noqa: E402
from normalize import training_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "asr" / "data"




GEMATRIA = {"א":1,"ב":2,"ג":3,"ד":4,"ה":5,"ו":6,"ז":7,"ח":8,"ט":9,"י":10,"כ":20,"ל":30,"מ":40,"נ":50,"ס":60,"ע":70,"פ":80,"צ":90}


def source_year(delivered_year: str | None) -> int | None:
    """'תשלח' / 'תשמ"א' -> 5738 / 5741 (the farbrengen the excerpt comes from)."""
    if not delivered_year:
        return None
    s = delivered_year.replace('"', "").replace("'", "").replace("״", "").replace("׳", "")
    s = s[2:] if s.startswith("תש") else s
    v = sum(GEMATRIA.get(c, 0) for c in s)
    return 5700 + v if 0 < v < 100 else None


def split_for(day_stem: str) -> str:
    h = int(hashlib.sha1(day_stem.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "dev" if h == 1 else "train"


def chunk_segments(segments: list[dict], max_seconds: float, min_seconds: float) -> list[dict]:
    chunks, cur = [], None
    for s in segments:
        text = (s.get("yiddish") or "").strip()
        start, end = float(s["start"]), float(s["end"])
        if not text or end <= start:
            continue
        if cur and end - cur["start"] <= max_seconds and start - cur["end"] < 2.0:
            cur["end"] = end; cur["segments"].append(s)
        else:
            if cur: chunks.append(cur)
            cur = {"start": start, "end": end, "segments": [s]}
    if cur: chunks.append(cur)
    return [c for c in chunks if c["end"] - c["start"] >= min_seconds]


def build(day_json: Path, out_root: Path, max_seconds: float, min_seconds: float, keep_diacritics: bool,
          dev_days: set[str] | None = None, test_days: set[str] | None = None, test_b_days: set[str] | None = None,
          min_anchor_rate: float = 0.3, require_anchored: bool = True, prefer_local: bool = False) -> list[dict]:
    rec = json.loads(day_json.read_text())
    if not rec.get("match"):
        return []
    site_ok = (bool((rec.get("sync") or {}).get("segments")) and rec.get("local_offset_seconds") is not None
               and not rec.get("sync_untrusted"))  # set by flag_untrusted_site_timings.py when the site file mismatches the audio
    if (rec.get("sync") or {}).get("segments") and not site_ok and not (rec.get("sync_local") or {}).get("segments"):
        print(f"  ! {day_json.stem}: site timings but offset unknown/refused and no local alignment; run compute_offsets.py or align_day.py")
        return []
    if site_ok and not prefer_local:
        segs, timing_source = rec["sync"]["segments"], "site"
        # calibrated per-day shift (calibrate_site_timings.py) beats the size/xcorr prefix offset: the site's
        # timing files are not all in the same reference frame
        offset = rec["site_shift_seconds"] if rec.get("site_shift_seconds") is not None else rec["local_offset_seconds"]
    elif (rec.get("sync_local") or {}).get("segments"):
        segs, timing_source, offset = rec["sync_local"]["segments"], "local", 0.0
        segs = [g for g in segs if g.get("anchor_rate", 1) >= min_anchor_rate
                and (not require_anchored or (g.get("start_anchored") and g.get("end_anchored")))]
    else:
        return []
    if offset:
        segs = [dict(g, start=float(g["start"]) + offset, end=float(g["end"]) + offset) for g in segs]
    # a calibrated shift can push the first segment slightly below zero: clamp, and drop anything left empty
    segs = [dict(g, start=max(0.0, float(g["start"])), end=max(0.0, float(g["end"]))) for g in segs]
    segs = [g for g in segs if g["end"] - g["start"] >= 0.2]
    local = ROOT / rec["local"]["path"]
    audio = decode_16k_mono(local)
    total = len(audio) / SR
    stem = day_json.stem
    year = rec["local"]["year"]
    out_dir = out_root / str(year) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, c in enumerate(chunk_segments(segs, max_seconds, min_seconds)):
        a, b = int(c["start"] * SR), int(min(c["end"], total) * SR)
        if b - a < int(min_seconds * SR):
            continue
        clip = out_dir / f"{i:03d}_{int(c['start']*100):07d}.flac"
        if not clip.exists():
            sf.write(clip, audio[a:b], SR, format="FLAC")
        raw = " ".join((s.get("yiddish") or "").strip() for s in c["segments"])
        rows.append({
            "id": f"{year}/{stem}/{i:03d}",
            "audio": str(clip.relative_to(ROOT)),
            "start": round(c["start"], 3), "end": round(min(c["end"], total), 3),
            "duration": round((b - a) / SR, 3),
            "text": training_text(raw, keep_diacritics=keep_diacritics),
            "text_raw": raw,
            "hebrew": " ".join((s.get("hebrew") or "").strip() for s in c["segments"]),
            "english": " ".join((s.get("english") or "").strip() for s in c["segments"]),
            "n_segments": len(c["segments"]),
            "timing_source": timing_source, "offset": offset,
            "anchor_rate": (round(sum(g.get("anchor_rate", 1) * g.get("n_words", 1) for g in c["segments"])
                                  / max(1, sum(g.get("n_words", 1) for g in c["segments"])), 3) if timing_source == "local" else None),
            "year": year, "day": stem, "source_mp3": rec["local"]["path"],
            "source_year": source_year((rec["api"].get("sicha") or {}).get("deliveredYear")),
            "source_date": (rec["api"].get("sicha") or {}).get("deliveredDate"),
            "split": ("test" if stem in (test_days or ()) else "test_b" if stem in (test_b_days or ()) else "dev" if stem in (dev_days or ())
                      else "train" if (dev_days or test_days) else split_for(stem)),
        })
    return rows


def _build_one(job):
    day_json, args_tuple = job
    try:
        return build(day_json, *args_tuple)
    except Exception as e:  # noqa: BLE001
        print(f"  ! {day_json.stem}: FAILED {e}", flush=True); return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1, help="parallel day workers (decode + slice); 8 is fine on a Mac Studio")
    ap.add_argument("--year", nargs="*", type=int, default=[])
    ap.add_argument("--max-seconds", type=float, default=28.0)
    ap.add_argument("--min-seconds", type=float, default=1.0)
    ap.add_argument("--keep-diacritics", action="store_true")
    ap.add_argument("--manifest-dir", default=str(DATA / "manifests"))
    ap.add_argument("--dev-days", nargs="*", default=[], help="day stems; if given with --test-days, all other days are train")
    ap.add_argument("--test-days", nargs="*", default=[])
    ap.add_argument("--splits-file", default=None, help="JSON {\"dev\": [...], \"test\": [...]} of day stems; overrides --dev-days/--test-days")
    ap.add_argument("--splits-b-file", default=None, help="JSON {\"test_b\": [...]} - a second held-out set (data/splits_b.json), written as test_b.jsonl and never trained on")
    ap.add_argument("--min-anchor-rate", type=float, default=0.3, help="local alignment only")
    ap.add_argument("--allow-unanchored-boundaries", action="store_true", help="local alignment only")
    ap.add_argument("--prefer-local", action="store_true", help="use sync_local even when site timings exist")
    args = ap.parse_args()

    days = sorted((DATA / "days").rglob("*.json"))
    if args.year:
        days = [d for d in days if int(d.parent.name) in args.year]
    dev_days, test_days, test_b_days = set(args.dev_days), set(args.test_days), set()
    if args.splits_file:
        sp = json.loads(Path(args.splits_file).read_text()); dev_days, test_days = set(sp["dev"]), set(sp["test"])
        print(f"splits from {args.splits_file}: {len(dev_days)} dev days, {len(test_days)} test days")
    if args.splits_b_file:
        test_b_days = set(json.loads(Path(args.splits_b_file).read_text())["test_b"])
        print(f"second held-out set from {args.splits_b_file}: {len(test_b_days)} test_b days")
    rows = []
    args_tuple = (DATA / "segments", args.max_seconds, args.min_seconds, args.keep_diacritics,
                  dev_days, test_days, test_b_days, args.min_anchor_rate, not args.allow_unanchored_boundaries, args.prefer_local)
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_build_one, [(d, args_tuple) for d in days]))
    else:
        results = [_build_one((d, args_tuple)) for d in days]
    for d, r in zip(days, results):
        rows.extend(r)
        src = r[0]["timing_source"] if r else "-"
        print(f"  {d.stem:<28} clips={len(r):3d}  audio={sum(x['duration'] for x in r)/60:5.1f} min  timing={src}")
    mdir = Path(args.manifest_dir); mdir.mkdir(parents=True, exist_ok=True)
    for split in ("all", "train", "dev", "test", "test_b"):
        sel = rows if split == "all" else [r for r in rows if r["split"] == split]
        (mdir / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sel))
        hrs = sum(r["duration"] for r in sel) / 3600
        words = sum(len(r["text"].split()) for r in sel)
        days_n = len({r["day"] for r in sel})
        print(f"{split:>5}: {len(sel):5d} clips  {hrs:6.2f} h  {words:7d} words  {days_n:3d} days")
    if rows:
        durs = np.array([r["duration"] for r in rows])
        print(f"clip duration: mean {durs.mean():.1f}s  median {np.median(durs):.1f}s  max {durs.max():.1f}s  min {durs.min():.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
