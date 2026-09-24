#!/usr/bin/env python3
"""Inventory the local Daily Sicha mp3 archive and fetch each day's metadata.

For every local file (named like '001 3-Tishrei 5787.mp3' inside a year folder)
the Hebrew date is resolved to the civil date the site is keyed by, then
GET /daily-sicha/get-daily-sicha?date=DD-MM-YYYY is fetched together with the
linked sync JSON (segment timings + Yiddish/Hebrew/English text). Results are
cached as asr/data/days/<year>/<basename>.json and never re-fetched unless
--force is given. Requests are spaced out; the site is someone else's server.

Usage:
  python fetch_days.py --inventory-only            # resolve dates for all files
  python fetch_days.py --year 5787 --month-dir 01-Tishrei
  python fetch_days.py --year 5786 5787            # everything the site has online
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from pyluach.dates import HebrewDate

ROOT = Path(__file__).resolve().parents[2]  # .../Documents/Sichos
DATA = ROOT / "asr" / "data"
BASE = "https://thedailysicha.com/backend"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Referer": "https://thedailysicha.com/"}
MONTHS = {
    "nisan": 1, "nissan": 1, "iyar": 2, "sivan": 3, "tamuz": 4, "tammuz": 4, "av": 5,
    "elul": 6, "tishrei": 7, "cheshvan": 8, "kisleiv": 9, "kislev": 9, "teiveis": 10,
    "teves": 10, "tevet": 10, "shvat": 11, "shevat": 11, "adar": 12,
}
FILE_RE = re.compile(
    r"^(?P<seq>\d{3})\s+(?P<day>\d{1,2})-(?P<month>[A-Za-z]+)(?:\s+(?P<adar>[12]))?(?:\s+(?P<year>5\d{3}))?\.mp3$"
)


def is_leap(year: int) -> bool:
    try:
        HebrewDate(year, 13, 1)
        return True
    except ValueError:
        return False


def inventory() -> list[dict]:
    rows = []
    for year_dir in sorted(p for p in ROOT.iterdir() if p.is_dir() and re.search(r"5\d{3}", p.name)):
        year = int(re.search(r"5\d{3}", year_dir.name).group())
        for mp3 in sorted(year_dir.rglob("*.mp3")):
            row = {"path": str(mp3.relative_to(ROOT)), "basename": mp3.name, "year": year,
                   "month_dir": mp3.parent.name, "status": "ok"}
            m = FILE_RE.match(mp3.name)
            if not m:
                row["status"] = "unparsed"; rows.append(row); continue
            month = MONTHS.get(m["month"].lower())
            if month is None:
                row["status"] = f"unknown month {m['month']}"; rows.append(row); continue
            if month == 12:
                if m["adar"] == "2":
                    month = 13
                elif m["adar"] is None and is_leap(year):
                    row["status"] = "ambiguous Adar in leap year"
            try:
                hd = HebrewDate(year, month, int(m["day"]))
            except ValueError as e:
                row["status"] = f"bad date: {e}"; rows.append(row); continue
            row.update(hebrew_month=month, hebrew_day=int(m["day"]), civil=hd.to_pydate().isoformat(),
                       api_date=hd.to_pydate().strftime("%d-%m-%Y"))
            rows.append(row)
    return rows


def get_json(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def fetch_day(row: dict, force: bool, delay: float) -> dict:
    out = DATA / "days" / str(row["year"]) / (Path(row["basename"]).stem + ".json")
    if out.exists() and not force:
        return json.loads(out.read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    api = get_json(f"{BASE}/daily-sicha/get-daily-sicha?date={row['api_date']}")
    time.sleep(delay)
    day = api.get("day") or {}
    site_basename = urllib.parse.unquote(Path(day.get("recUrl", "")).name)
    record = {"local": row, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "api": api,
              "match": site_basename == row["basename"], "site_basename": site_basename, "sync": None}
    sync_url = (api.get("sicha") or {}).get("syncedTranslationUrl")
    if sync_url and record["match"]:
        try:
            record["sync"] = get_json(sync_url)
        except Exception as e:  # noqa: BLE001
            record["sync_error"] = str(e)[:200]
        time.sleep(delay)
    out.write_text(json.dumps(record, ensure_ascii=False))
    return record


HEBREW_MONTH_DIR = {7: "01-Tishrei", 8: "02-Cheshvan", 9: "03-Kislev", 10: "04-Teiveis", 11: "05-Shvat", 12: "06-Adar",
                    13: "06b-Adar 2", 1: "07-Nisan", 2: "08-Iyar", 3: "09-Sivan", 4: "10-Tamuz", 5: "11-Av", 6: "12-Elul"}


def download(url: str, dest: Path, delay: float) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url.replace(" ", "%20"), headers={"User-Agent": "Mozilla/5.0", "Referer": "https://thedailysicha.com/"})
    with urllib.request.urlopen(req, timeout=300) as r:
        dest.write_bytes(r.read())
    time.sleep(delay)


def fetch_site_year(year: int, force: bool, delay: float, limit: int) -> None:
    """Years with no local audio: walk every day of the Hebrew year, fetch metadata + sync, and
    download the daily mp3 into asr/data/audio/<year>/ so the rest of the pipeline sees it as local."""
    d = HebrewDate(year, 7, 1); end = HebrewDate(year + 1, 7, 1)
    seen = matched = synced = 0
    while d < end and (not limit or matched < limit):
        api_date = d.to_pydate().strftime("%d-%m-%Y")
        month = d.month if not (d.month == 12 and is_leap(year)) else 12
        month_dir = HEBREW_MONTH_DIR.get(13 if (d.month == 13) else month, f"{month:02d}")
        if d.month == 12 and is_leap(year):
            month_dir = "06a-Adar 1"
        d = d + 1
        try:
            api = get_json(f"{BASE}/daily-sicha/get-daily-sicha?date={api_date}")
        except Exception as e:  # noqa: BLE001
            print(f"  {api_date}: ERROR {str(e)[:80]}"); time.sleep(delay); continue
        time.sleep(delay)
        day = api.get("day") or {}
        if not day.get("recUrl") or not (api.get("sicha") or {}).get("contentHtml"):
            continue
        if day.get("date") and day["date"] != api_date:   # site returns the nearest sicha for dates without one
            continue
        seen += 1
        basename = urllib.parse.unquote(Path(day["recUrl"]).name)
        out = DATA / "days" / str(year) / (Path(basename).stem + ".json")
        if out.exists() and not force:
            matched += 1; continue
        audio_path = DATA / "audio" / str(year) / basename
        try:
            download(day["recUrl"], audio_path, delay)
        except Exception as e:  # noqa: BLE001
            print(f"  {basename}: audio download failed {str(e)[:80]}"); continue
        row = {"path": str(audio_path.relative_to(ROOT)), "basename": basename, "year": year, "month_dir": month_dir,
               "status": "ok", "civil": api_date[6:] + "-" + api_date[3:5] + "-" + api_date[:2], "api_date": api_date, "source": "site-download"}
        record = {"local": row, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "api": api, "match": True,
                  "site_basename": basename, "sync": None}
        sync_url = api["sicha"].get("syncedTranslationUrl")
        if sync_url:
            try:
                record["sync"] = get_json(sync_url); synced += 1
            except Exception as e:  # noqa: BLE001
                record["sync_error"] = str(e)[:200]
            time.sleep(delay)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, ensure_ascii=False))
        matched += 1
        print(f"  {basename:<12} {api_date}  {audio_path.stat().st_size/1e6:4.1f} MB  sync={bool(record['sync'])!s:<5} {day.get('title','')[:40]}", flush=True)
    print(f"{year}: {seen} days with a sicha online, {matched} cached, {synced} with sync timings (this run)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-years", nargs="*", type=int, default=[], help="years with no local audio: fetch metadata and download the daily mp3s")
    ap.add_argument("--inventory-only", action="store_true")
    ap.add_argument("--year", nargs="*", type=int, default=[])
    ap.add_argument("--month-dir", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--delay", type=float, default=0.8)
    args = ap.parse_args()

    if args.site_years:
        DATA.mkdir(parents=True, exist_ok=True)
        for y in args.site_years:
            fetch_site_year(y, args.force, args.delay, args.limit)
        return 0
    rows = inventory()
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "inventory.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    by_year = {}
    for r in rows:
        by_year.setdefault(r["year"], [0, 0]); by_year[r["year"]][0] += 1
        if r["status"] != "ok": by_year[r["year"]][1] += 1
    print(f"inventory: {len(rows)} files -> {DATA/'inventory.jsonl'}")
    for y, (n, bad) in sorted(by_year.items()):
        print(f"  {y}: {n} files, {bad} unresolved")
    for r in rows:
        if r["status"] != "ok":
            print("   !", r["path"], "->", r["status"])
    if args.inventory_only:
        return 0

    online = set(map(int, get_json(f"{BASE}/daily-sicha/get-years")["years"]))
    print("years with text online:", sorted(online))
    todo = [r for r in rows if r["status"] == "ok" and r["year"] in online
            and (not args.year or r["year"] in args.year)
            and (not args.month_dir or r["month_dir"] in args.month_dir)]
    if args.limit:
        todo = todo[: args.limit]
    print(f"fetching {len(todo)} days")
    matched = synced = 0
    for i, r in enumerate(todo, 1):
        try:
            rec = fetch_day(r, args.force, args.delay)
        except Exception as e:  # noqa: BLE001
            print(f"  {i:4d} {r['basename']}: ERROR {str(e)[:120]}"); continue
        seg = len((rec.get("sync") or {}).get("segments") or [])
        matched += rec["match"]; synced += bool(seg)
        flag = "" if rec["match"] else f"  (site has {rec['site_basename'] or 'nothing'})"
        print(f"  {i:4d} {r['basename']:<28} {r['api_date']}  match={rec['match']!s:<5} segments={seg:<4}{flag}")
    print(f"done: {matched}/{len(todo)} matched the local filename, {synced} with sync timings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
