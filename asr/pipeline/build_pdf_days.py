#!/usr/bin/env python3
"""Build day records for years whose Yiddish text comes from the yearly hanacha PDFs (no site API record):
5775-5781 (audio in the archive folders, text in data/pdf_text/<year>/NNN.json from pdf_hanachos.py or
pdf_fontmap.py). Writes data/days/<year>/<NNN D-Month YEAR>.json in the shape align_day.py and build_dataset.py
expect: local (from data/inventory.jsonl), a synthetic api.sicha.contentHtml (one <p> per PDF line, trailing
dedication removed), deliveredYear parsed from the source line, match, local_offset_seconds = 0 (no site excerpt),
no sync (timings come from align_day.py --all-missing). Existing records with sync_local are left alone unless
--force. Days listed in --exclude are written with match=null so build_dataset.py skips them.

  python -u pipeline/build_pdf_days.py --year 5778 5779 5780 5781 5775 5776 5777 --exclude "053 17-Kisleiv 5781" "068 9-Teiveis 5781"
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_drafts_vs_pdf import ref_text  # noqa: E402  (drops the trailing dedication after the last lone '*')

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "asr" / "data"
YEAR_RE = re.compile(r"ה'?(תש[א-ת\"'׳״]{1,4})")      # ה'תשל"ג inside the source line


def delivered_year(source: str | None, header: str | None) -> str | None:
    for s in (source, header):
        if s:
            m = YEAR_RE.search(s)
            if m:
                return m.group(1)
    return None


def to_html(body: str) -> str:
    lines = [l.strip() for l in body.split("\n") if l.strip() and l.strip() != "*"]
    return "".join(f"<p>{html.escape(l)}</p>" for l in lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, nargs="+", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="day stems (with year suffix) to write with match=null")
    ap.add_argument("--force", action="store_true", help="overwrite records that already have sync_local")
    args = ap.parse_args()
    inv = {}
    for line in open(DATA / "inventory.jsonl", encoding="utf-8"):
        r = json.loads(line)
        inv.setdefault(r["year"], {})[Path(r["path"]).stem[:3]] = r
    excluded = set(args.exclude)
    for year in args.year:
        recs = sorted((DATA / "pdf_text" / str(year)).glob("[0-9][0-9][0-9].json"))
        out_dir = DATA / "days" / str(year); out_dir.mkdir(parents=True, exist_ok=True)
        n_written = n_skipped = n_nomp3 = n_excl = 0
        for f in recs:
            rec = json.loads(f.read_text(encoding="utf-8"))
            if not rec.get("mp3"):
                n_nomp3 += 1; continue
            mp3 = rec["mp3"][0]
            stem = Path(mp3).stem                      # "011 18-Tishrei"
            day_stem = f"{stem} {year}"                 # unique across years, like the 5787 convention
            out = out_dir / f"{day_stem}.json"
            if out.exists() and not args.force:
                old = json.loads(out.read_text(encoding="utf-8"))
                if (old.get("sync_local") or {}).get("segments"):
                    n_skipped += 1; continue
            local = dict(inv.get(year, {}).get(rec["num"]) or {"path": mp3, "basename": Path(mp3).name, "year": year,
                                                                "month_dir": Path(mp3).parent.name, "status": "ok"})
            body = ref_text(rec)
            dy = delivered_year(rec.get("source"), rec.get("header"))
            excluded_day = day_stem in excluded
            record = {
                "local": local, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "api": {"sicha": {"contentHtml": to_html(body), "content": body, "deliveredYear": dy, "deliveredDate": None,
                                  "recUrl": None, "source_line": rec.get("source"), "header": rec.get("header")},
                        "day": {"date": local.get("api_date"), "title": rec.get("header"), "id": None},
                        "synthetic": True, "text_source": (rec.get("decode") or {}).get("method", "pdf") if isinstance(rec.get("decode"), dict) else ("pdf-fontmap" if (DATA / "pdf_text" / str(year) / "_fontmap.json").exists() else "pdf-text")},
                "match": None if excluded_day else {"kind": "pdf-day-number", "num": rec["num"], "num_source": rec.get("num_source", "printed"), "pdf_pages": rec.get("pages")},
                "excluded": "audio/text mismatch (Golan 2026-09-23)" if excluded_day else None,
                "site_basename": None, "sync": None, "local_offset_seconds": 0.0, "offset_method": "none-no-site-excerpt",
                "pdf_words": rec.get("n_words"),
            }
            out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
            n_written += 1; n_excl += excluded_day
        print(f"{year}: wrote {n_written} day records ({n_excl} excluded, match=null), kept {n_skipped} already aligned, {n_nomp3} PDF days without audio", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
