#!/usr/bin/env python3
"""Split farbrengen transcript PDFs (and .doc) into Yiddish, Hebrew and English text streams.

Input: the two "Filemail.com ..." folders of full/near-full farbrengen hanachos (5732-5747).
Each PDF mixes cover sheets, English synopses, Yiddish hanacha, Hebrew translation or maamar.
For every file this writes asr/data/hanachos/<slug>.{yi,he,en}.txt (page-classified,
running headers and page numbers removed, flipped leading punctuation repaired) and an
index.json with per-file stats, the Hebrew year, English synopsis entries with durations
("Sicha 1 (17 minutes)") and a needs_ocr flag for scanned or garbled-font files.
Text streams are for vocabulary/LM use and, once audio is obtained, for alignment.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "asr" / "data" / "hanachos"
SOURCES = [ROOT / "Filemail.com - Full or almost full farbrengens - transcripts",
           ROOT / "Filemail.com - 3 more such farbrengens - left off the previous email"]
HEB = re.compile(r"[א-ת]"); LAT = re.compile(r"[A-Za-z]"); NIK = re.compile(r"[֑-ׇ]")
YID = {"און","וואס","דער","די","דאס","איז","פון","אויף","מיט","זיך","ניט","נישט","צו","דעם","מען","האט","אז","ווי","אבער","זיין","דאך","נאך","געווען","וועט","ביי","פאר","אויך","וואו","דערפון","זאל","דארף","אלע","אויס","ווערט","זיינען"}
HEBW = {"של","את","על","הוא","זה","לא","כל","גם","אם","כי","אבל","אשר","היה","יש","שהוא","כמו","אל","או","עם","ענין","בנוגע","שזה","שיש","אינו","הם","זהו","ולכן","אלא","לפי","כאשר","ישראל"}
GEMATRIA = {"א":1,"ב":2,"ג":3,"ד":4,"ה":5,"ו":6,"ז":7,"ח":8,"ט":9,"י":10,"כ":20,"ל":30,"מ":40,"נ":50,"ס":60,"ע":70,"פ":80,"צ":90,"ק":100,"ר":200,"ש":300,"ת":400}
SYNOPSIS = re.compile(r"(Sicha[h]?\s*\d*|Ma[’'`]?amar|Maamar)[^()\n]{0,80}\((\d+(?:[:.]\d+)?)\s*[Mm]in", re.I)


def classify(text: str) -> str:
    words = text.split()
    if len(words) < 5:
        return "blank"
    heb = sum(1 for w in words if HEB.search(w)); lat = sum(1 for w in words if LAT.search(w))
    if lat > heb * 2:
        return "english"
    letters = len(HEB.findall(text))
    other = len(re.findall(r"[^\s\w֑-ׇ.,;:!?()\[\]\"'\-–־׳״0-9]", text))
    if letters and other > letters * 0.15:
        return "garbled"
    ws = [NIK.sub("", re.sub(r"[^א-ת֑-ׇ]", "", w)) for w in words]
    y = sum(w in YID for w in ws); h = sum(w in HEBW for w in ws)
    if y >= 3 and y >= 1.5 * h:
        return "yiddish"
    if h >= 3 and h > y:
        return "hebrew"
    return "hebscript"


def hebrew_year(name: str) -> int | None:
    m = re.search(r"(57\d\d)", name)
    if m:
        return int(m.group(1))
    n = name.replace("_", '"').replace("״", '"')
    m = re.search(r"תש([א-ת])\"([א-ת])", n)          # תשל"ב, תשמ"א (never the month תשרי)
    if m:
        return 5700 + GEMATRIA.get(m.group(1), 0) + GEMATRIA.get(m.group(2), 0)
    m = re.search(r"(?<![א-ת])([א-ת])\"([א-ת])(?![א-ת])", n)  # מ"ב -> 5742
    if m:
        return 5700 + GEMATRIA.get(m.group(1), 0) + GEMATRIA.get(m.group(2), 0)
    return None


def clean_line(line: str) -> str:
    line = re.sub(r"\b0{3,}\b", " ", line).replace("\xad", "").replace("‏", "").replace("‎", "")
    toks = line.split(); out = []
    for t in toks:
        if len(t) > 1 and t[0] in ".,;:!?" and out and HEB.search(t):   # flipped leading punctuation
            out[-1] += t[0]; t = t[1:]
        out.append(t)
    return " ".join(out)


FRONT_WORDS = re.compile(r"יוצא לאור|ועד תלמידי|ועד הנחות|מפתח|תוכן הענינים|Table of Contents|Published by|Copyright|©|לזכות|לעילוי נשמת|ת\.נ\.צ\.ב\.ה")
DOT_LEADER = re.compile(r"(\.\s?){6,}")


def is_front_matter(lines: list[str]) -> bool:
    """Cover, publisher, dedication and table-of-contents pages."""
    if not lines:
        return False
    toc = sum(1 for l in lines if DOT_LEADER.search(l))
    front = sum(1 for l in lines if FRONT_WORDS.search(l))
    words = sum(len(l.split()) for l in lines)
    return toc >= max(3, 0.3 * len(lines)) or (front >= 1 and words < 120) or words < 25


def is_page_number(line: str) -> bool:
    s = line.strip()
    return bool(re.fullmatch(r"[\d\-–\s|]{1,6}", s) or re.fullmatch(r"[א-ת]{1,3}[\"'׳״]?", s) or re.fullmatch(r"(page|עמוד)?\s*\d+", s, re.I))


def ocr_page(page, langs: str, dpi: int = 300, tessdata_dir: str | None = None) -> str:
    """Render a page and OCR it with tesseract (needs the heb/yid packs: brew install tesseract-lang).
    tessdata_dir lets you use custom models, e.g. asr/models/tessdata (book118 + copies of heb/yid)."""
    if not shutil.which("tesseract"):
        raise RuntimeError("tesseract not installed")
    env = dict(os.environ, TESSDATA_PREFIX=tessdata_dir) if tessdata_dir else None
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        png = Path(tmp.name)
    try:
        pix.save(str(png))
        res = subprocess.run(["tesseract", str(png), "stdout", "-l", langs, "--psm", "6"], capture_output=True, text=True, env=env)
        return res.stdout
    finally:
        png.unlink(missing_ok=True)


def extract_pdf(path: Path, ocr: bool = False, ocr_langs: str = "yid+heb", tessdata_dir: str | None = None) -> tuple[dict, dict]:
    doc = pymupdf.open(str(path))
    if doc.needs_pass:
        doc.authenticate("")
    pages = []; ocr_pages = 0
    for i in range(doc.page_count):
        text = doc[i].get_text("text") or ""
        if ocr and classify(text) in ("blank", "garbled") and (doc[i].get_images() or classify(text) == "garbled"):
            text = ocr_page(doc[i], ocr_langs, tessdata_dir=tessdata_dir); ocr_pages += 1
        lines = [clean_line(l) for l in text.splitlines()]
        lines = [l for l in lines if l.strip()]
        pages.append(lines)
    # running headers/footers: first/last lines repeated on >= 25% of pages
    edge = Counter()
    for lines in pages:
        for l in set(lines[:2] + lines[-2:]):
            edge[re.sub(r"\d+", "#", l)] += 1
    running = {k for k, v in edge.items() if v >= max(3, 0.25 * len(pages))}
    streams = {"yi": [], "he": [], "en": []}; classes = []; synopsis = []
    for pno, lines in enumerate(pages, 1):
        body = [l for l in lines if re.sub(r"\d+", "#", l) not in running and not is_page_number(l)
                and not re.search(r"פארברענגען עם הרבי|פאַרברענגען עם הרבי", l)]
        text = "\n".join(body)
        c = "frontmatter" if is_front_matter(body) else classify(text); classes.append(c)
        for m in SYNOPSIS.finditer(text):
            synopsis.append({"page": pno, "label": re.sub(r"\s+", " ", m.group(1)).strip(), "minutes": m.group(2)})
        if c in ("yiddish", "hebscript"):
            streams["yi"].append(f"\n[[page {pno}]]\n" + text)
        elif c == "hebrew":
            streams["he"].append(f"\n[[page {pno}]]\n" + text)
        elif c == "english":
            streams["en"].append(f"\n[[page {pno}]]\n" + text)
    cnt = Counter(classes)
    needs_ocr = cnt.get("garbled", 0) + cnt.get("blank", 0) >= 0.5 * len(pages)
    return streams, {"pages": len(pages), "classes": dict(cnt), "running_headers": sorted(running)[:6],
                     "needs_ocr": needs_ocr and not ocr, "ocr_pages": ocr_pages, "synopsis": synopsis}


def extract_doc(path: Path) -> tuple[dict, dict]:
    text = subprocess.run(["textutil", "-convert", "txt", "-stdout", str(path)], capture_output=True, text=True).stdout
    paras = [clean_line(p) for p in text.splitlines() if p.strip()]
    streams = {"yi": [], "he": [], "en": []}; classes = Counter()
    for p in paras:
        c = classify(p) if len(p.split()) >= 5 else "yiddish"; classes[c] += 1
        streams["yi" if c in ("yiddish", "hebscript", "blank") else "he" if c == "hebrew" else "en"].append(p)
    return streams, {"pages": 1, "classes": dict(classes), "running_headers": [], "needs_ocr": False, "synopsis": []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", action="store_true", help="OCR scanned/garbled pages with tesseract")
    ap.add_argument("--ocr-langs", default="yid+heb")
    ap.add_argument("--tessdata-dir", default=None, help="custom tessdata dir, e.g. models/tessdata for book118")
    ap.add_argument("--only", nargs="*", default=[], help="substrings of file names to process")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    index = json.loads((OUT / "index.json").read_text()) if (OUT / "index.json").exists() and args.only else []
    for src in SOURCES:
        for f in sorted(p for p in src.iterdir() if p.suffix.lower() in (".pdf", ".doc", ".docx")):
            if args.only and not any(k in f.name for k in args.only):
                continue
            try:
                streams, meta = extract_pdf(f, args.ocr, args.ocr_langs, args.tessdata_dir) if f.suffix.lower() == ".pdf" else extract_doc(f)
            except Exception as e:  # noqa: BLE001
                print(f"  {f.name}: FAILED {e}"); continue
            slug = re.sub(r"[^\wא-ת]+", "_", f.stem).strip("_")[:70]
            words = {}
            yi_text = re.sub(r"\[\[page \d+\]\]", "", "\n".join(streams["yi"]))
            toks = re.findall(r"[\u05d0-\u05ea\u0591-\u05C7'\"]+", yi_text)
            meta["nikkud_ratio"] = round(sum(1 for t in toks if NIK.search(t)) / max(1, len(toks)), 3)
            for lang, parts in streams.items():
                text = "\n".join(parts).strip()
                words[lang] = len(re.findall(r"\S+", re.sub(r"\[\[page \d+\]\]", "", text)))
                if text:
                    (OUT / f"{slug}.{lang}.txt").write_text(f"# source: {f.relative_to(ROOT)}\n# language stream: {lang}\n" + text + "\n")
            entry = {"file": str(f.relative_to(ROOT)), "slug": slug, "year": hebrew_year(f.name), **meta, "words": words}
            index = [e for e in index if e["slug"] != slug] + [entry]
            flag = "NEEDS OCR" if meta["needs_ocr"] else ""
            print(f"  {f.name[:52]:<52} yr={entry['year']} p={meta['pages']:4d} yi={words['yi']:6d} he={words['he']:6d} en={words['en']:5d} synopsis={len(meta['synopsis']):2d} {flag}")
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    # word frequency list of the 5739+ Yiddish streams (vocabulary / biasing / convention study)
    freq = Counter()
    for e in index:
        if (e["year"] or 0) >= 5739:
            f = OUT / f"{e['slug']}.yi.txt"
            if f.exists():
                for t in re.findall(r"[\u05d0-\u05ea]+", NIK.sub("", f.read_text())):
                    freq[t] += 1
    (OUT / "vocab_yi_5739plus.tsv").write_text("".join(f"{w}\t{c}\n" for w, c in freq.most_common()))
    print(f"vocabulary (5739+): {len(freq):,} distinct forms, {sum(freq.values()):,} tokens -> vocab_yi_5739plus.tsv")
    tot = sum(e["words"]["yi"] for e in index); tot39 = sum(e["words"]["yi"] for e in index if (e["year"] or 0) >= 5739)
    print(f"\n{len(index)} files -> {OUT}; Yiddish words: {tot:,} total, {tot39:,} from 5739+; needs OCR: {[e['slug'] for e in index if e['needs_ocr']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
