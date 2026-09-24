#!/usr/bin/env python3
"""Calibrate each site timing file's reference frame against our own alignment.

The site's generated-sync files are not all in one time frame: some are relative to the raw
excerpt (need + prefix offset), some already include the daily file's dedication prefix, and some
were made against a different version of the daily file. For every site-timed day that also has
`sync_local`, this matches site segments to local chunks by their first four normalized words,
takes the median of (local_start - site_start) as the day's shift, measures agreement after the
shift, and stores `site_shift_seconds`, `site_shift_agreement`, `site_shift_pairs` in the day JSON.
`build_dataset.py` uses the shift instead of `local_offset_seconds`; days with agreement < 0.5 or
fewer than 5 pairs are marked `sync_untrusted` so the builder falls back to `sync_local`.

  python calibrate_site_timings.py          # all site-timed days
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import scoring_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "asr" / "data"


def main() -> int:
    calibrated = untrusted = nolocal = 0; kinds = {"record": 0, "daily": 0, "other": 0}
    for p in sorted((DATA / "days").rglob("*.json")):
        rec = json.loads(p.read_text())
        site = (rec.get("sync") or {}).get("segments")
        if not site:
            continue
        loc = (rec.get("sync_local") or {}).get("segments")
        rec.pop("sync_untrusted", None); rec.pop("sync_untrusted_reason", None)
        if not loc:
            rec["site_shift_source"] = "offset"; nolocal += 1
            p.write_text(json.dumps(rec, ensure_ascii=False)); continue
        firsts = {}
        for c in loc:
            w = scoring_text(c["yiddish"]).split()
            if len(w) >= 4:
                firsts.setdefault(" ".join(w[:4]), float(c["start"]))
        pairs = []
        for s in site:
            w = scoring_text(s.get("yiddish") or "").split()
            if len(w) >= 4 and " ".join(w[:4]) in firsts:
                pairs.append((float(s["start"]), firsts[" ".join(w[:4])]))
        if len(pairs) < 5:
            rec.update(site_shift_source="insufficient", site_shift_pairs=len(pairs), sync_untrusted=True,
                       sync_untrusted_reason=f"only {len(pairs)} matched segments")
            untrusted += 1; p.write_text(json.dumps(rec, ensure_ascii=False)); continue
        shift = st.median(l - s for s, l in pairs)
        agree = sum(1 for s, l in pairs if abs(l - (s + shift)) <= 1.0) / len(pairs)
        off = float(rec.get("local_offset_seconds") or 0.0)
        kind = "record" if abs(shift - off) < 1.5 else "daily" if abs(shift) < 1.5 else "other"
        kinds[kind] += 1
        rec.update(site_shift_seconds=round(shift, 3), site_shift_agreement=round(agree, 3), site_shift_pairs=len(pairs),
                   site_shift_source="calibrated", site_frame=kind)
        if agree < 0.5:
            rec.update(sync_untrusted=True, sync_untrusted_reason=f"agreement {agree:.2f} after shift {shift:.2f}s"); untrusted += 1
        else:
            calibrated += 1
        p.write_text(json.dumps(rec, ensure_ascii=False))
    print(f"calibrated (trusted): {calibrated}; untrusted -> local alignment: {untrusted}; no cross-check (offset kept): {nolocal}")
    print(f"frames: {kinds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
