#!/usr/bin/env python3
"""Measure where the site's raw excerpt starts inside each local daily mp3.

The site's sync timings are relative to sicha.recUrl (the raw excerpt). The
daily file you have is that excerpt with, on some days, a dedication recording
prepended (about 10 s). For every cached day this script stores
`local_offset_seconds` in the day JSON. The API's dedicationRec flag is NOT
reliable (older days carry the prefix without the flag), so every day is checked:
the excerpt's byte size (HEAD) is converted to seconds at the local file's bitrate
and compared with the local duration; when they agree within 1 s the offset is 0,
otherwise the excerpt is downloaded and cross-correlated against the local file,
and the lag must agree with the duration difference within 2 s or the day is
refused (offset None) rather than misaligned.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio import SR, decode_16k_mono as decode, duration_seconds as local_duration  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "asr" / "data"
CACHE = DATA / "records"




def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    req = urllib.request.Request(url.replace(" ", "%20"), headers={"User-Agent": "Mozilla/5.0", "Referer": "https://thedailysicha.com/"})
    with urllib.request.urlopen(req, timeout=180) as r:
        dest.write_bytes(r.read())


def offset_by_xcorr(record: np.ndarray, local: np.ndarray, probe_s: float = 20.0, search_s: float = 60.0, step: int = 4) -> tuple[float, float, float]:
    probe = record[: int(probe_s * SR)][::step]; hay = local[: int(search_s * SR)][::step]
    probe = (probe - probe.mean()) / (probe.std() + 1e-9); hay = (hay - hay.mean()) / (hay.std() + 1e-9)
    corr = np.correlate(hay, probe, mode="valid") / len(probe)
    lag = int(np.argmax(corr)); top = float(corr[lag]); runner = float(np.sort(corr)[-2]) if len(corr) > 1 else 0.0
    return lag * step / SR, top, runner


def head_bytes(url: str) -> int:
    req = urllib.request.Request(url.replace(" ", "%20"), headers={"User-Agent": "Mozilla/5.0", "Referer": "https://thedailysicha.com/"}, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length") or 0)



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute even verified offsets")
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--tolerance", type=float, default=1.0, help="seconds of size/duration disagreement that triggers a download")
    ap.add_argument("--verify-sample", type=int, default=0, help="cross-correlate N random size-match days to check the size heuristic")
    args = ap.parse_args()
    if args.verify_sample:
        import random
        cands = [d for d in sorted((DATA / "days").rglob("*.json")) if json.loads(d.read_text()).get("offset_method") == "size-match"]
        random.seed(7); bad = 0
        for d in random.sample(cands, min(args.verify_sample, len(cands))):
            rec = json.loads(d.read_text()); target = CACHE / Path(rec["api"]["sicha"]["recUrl"]).name
            download(rec["api"]["sicha"]["recUrl"], target); time.sleep(args.delay)
            record = decode(target); local = decode(ROOT / rec["local"]["path"])
            lag, top, runner = offset_by_xcorr(record, local)
            flag = "OK" if abs(lag) < 0.5 else "PREFIX MISSED"
            bad += flag != "OK"
            print(f"  verify {d.stem:<28} lag {lag:6.3f}s peak {top:.2f}/{runner:.2f} record {len(record)/SR:.1f}s local {len(local)/SR:.1f}s  {flag}", flush=True)
        print(f"verification: {bad} of {min(args.verify_sample, len(cands))} size-match days had a hidden prefix")
        return 0
    CACHE.mkdir(parents=True, exist_ok=True)
    days = sorted((DATA / "days").rglob("*.json"))
    counts = {"size-match": 0, "xcorr": 0, "refused": 0, "skipped": 0}
    for d in days:
        rec = json.loads(d.read_text())
        if rec.get("offset_method") in ("size-match", "xcorr", "xcorr-dedication", "xcorr-verify") and not args.force:
            counts["skipped"] += 1; continue
        sicha = rec["api"].get("sicha") or {}; rec_url = sicha.get("recUrl")
        local_path = ROOT / rec["local"]["path"]
        if not rec_url:
            rec.update(local_offset_seconds=None, offset_method="no-record-url"); d.write_text(json.dumps(rec, ensure_ascii=False)); counts["refused"] += 1; continue
        try:
            rbytes = head_bytes(rec_url)
        except Exception as e:  # noqa: BLE001
            print(f"  {d.stem}: HEAD failed {str(e)[:80]}", flush=True); counts["refused"] += 1; continue
        time.sleep(args.delay)
        lbytes = local_path.stat().st_size; lsec = local_duration(local_path)
        est_record_sec = rbytes * lsec / lbytes
        diff = lsec - est_record_sec
        info = {"record_bytes": rbytes, "local_bytes": lbytes, "local_seconds": round(lsec, 2), "est_record_seconds": round(est_record_sec, 2),
                "duration_diff": round(diff, 2), "dedication_flag": bool((rec["api"].get("day") or {}).get("dedicationRec"))}
        if abs(diff) < args.tolerance:
            rec.update(local_offset_seconds=0.0, offset_method="size-match", **info); counts["size-match"] += 1
        else:
            target = CACHE / Path(rec_url).name
            download(rec_url, target); time.sleep(args.delay)
            record = decode(target); local = decode(local_path)
            lag, top, runner = offset_by_xcorr(record, local)
            true_diff = len(local) / SR - len(record) / SR
            ok = abs(lag - true_diff) < 2.0
            rec.update(local_offset_seconds=round(lag, 3) if ok else None, offset_method="xcorr" if ok else "xcorr-inconsistent",
                       offset_peak=round(top, 3), offset_runner_up=round(runner, 3), record_seconds=round(len(record) / SR, 2), **info)
            counts["xcorr" if ok else "refused"] += 1
            print(f"  {d.stem:<28} prefix {lag:6.3f}s (duration diff {true_diff:5.2f}s) peak {top:.2f}/{runner:.2f} "
                  f"dedication_flag={info['dedication_flag']} {'OK' if ok else 'INCONSISTENT'}", flush=True)
        d.write_text(json.dumps(rec, ensure_ascii=False))
    print("offsets:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
