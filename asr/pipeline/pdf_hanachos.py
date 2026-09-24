#!/usr/bin/env python3
"""Extract per-day Yiddish hanachos from The Daily Sicha yearly PDFs (Filemail 2026-09-23).

Usage (from asr/):  python -u pipeline/pdf_hanachos.py --year 5781 [--year 5782 ...] [--pdf-dir DIR] [--out data/pdf_text]
Requires: pip install pymupdf   (pdftotext fragments these PDFs; PyMuPDF rawdict + geometric line
rebuild gives correct RTL reading order).

Works only on years whose PDF has a real Hebrew text layer (checked 2026-09-23):
5771 5772 5773 5781 5782 (single-column layout; 5771-5773 print no day number, so days are numbered
sequentially).  5784-5787 have a two-column body layout that this extractor does not handle; their text
already comes from the site API (data/days/), so they are refused here.  The other years (5768-5770,
5774-5780) use custom font encodings and must go through OCR (extract_hanachos.py --ocr style) instead.

Layout of every day (2 pages): header line 'בס"ד. "השיחה היומית" ליום ...', 'התוכן' + Hebrew summary
(small font), a source line '... משיחת ... [NNN]' where NNN is the day number = the mp3 stem prefix,
a marker line 'הנחה פרטית בלתי מוגה' (or similar), then the Yiddish body in the main font size.
Page footers are in small type (< 8 pt) and are dropped by size; body text can run to the page bottom.  Brackets/parentheses are stored mirrored and digit runs
reversed (visual order); both are corrected here.  Nikud is dropped (v1 targets strip it anyway).

Output: <out>/<year>/<NNN>.json  {num, year, header, source, body, pages, n_words, mp3}  and
<out>/<year>/_summary.json with match statistics against the year's audio folder.
"""
import argparse, json, re, sys
from collections import Counter
from pathlib import Path
try:
    import pymupdf
except ImportError:
    sys.exit("pip install pymupdf")

ROOT = Path(__file__).resolve().parents[2]          # folder that contains asr/
PDF_DIR = ROOT / "Filemail.com - text sicha yomis"
BIDI = re.compile('[‎‏‪-‮⁦-⁩]')
NIKUD = re.compile('[֑-ׇ]')
MIRROR = str.maketrans({'(': ')', ')': '(', '[': ']', ']': '[', '{': '}', '}': '{', '<': '>', '>': '<'})
YEAR_PDF = {  # year -> file name
    5768: "DS 5768 vayikra.pdf", 5769: "DS 5769.pdf", 5770: "DS 5770.pdf", 5771: "DS 5771.pdf",
    5772: "DS 5772.pdf", 5773: "DS 5773.pdf", 5774: "DS 5774.pdf", 5775: "DS 5775.pdf",
    5776: "DS 5776.pdf", 5777: "DS 5777.pdf", 5778: "DS 5778.pdf",
    5779: "Daily sicha yomis 5779 DS_5779_Yiddish.pdf", 5780: "DS_5780_Yiddish.pdf",
    5781: "DS-5781_Y.pdf", 5782: "DS_5782_y.pdf", 5784: "DS_5784_y.pdf", 5785: "DS_5785_y.pdf",
    5786: "DS_5786_y.pdf", 5787: "DS_5787_y.pdf",
}
AUDIO_DIR = {5775: "DS 5775 CHUL", 5776: "DS 5776", 5777: "DS 5777", 5778: "DS 5778",
             5779: "Sicha Yomis 5779 audio", 5780: "Sicha Yomis 5780 audio", 5781: "Sicha Yomis 5781 audio",
             5786: "DS 5786", 5787: "DS_5787"}
HEADER_RE = re.compile(r'השיחה היומית')
NUM_RE = re.compile(r'\[(\d{3})\]')
MARKER_RE = re.compile(r'^(הנחה|רשימה|תמליל)\b.*(מוגה|מילולית)')


def fix_visual(s):
    s = s.translate(MIRROR)
    s = re.sub(r'[0-9]+', lambda m: m.group()[::-1], s)
    return s


def page_lines(page, ytol=2.5, gap=1.2):
    """Rebuild lines geometrically: cluster chars by baseline, order right-to-left."""
    chars = []
    for b in page.get_text('rawdict')['blocks']:
        if b.get('type', 0) != 0:
            continue
        for l in b['lines']:
            for s in l['spans']:
                for c in s['chars']:
                    ch = BIDI.sub('', c['c'])
                    if not ch or ch.isspace() or NIKUD.match(ch):
                        continue
                    x0, y0, x1, y1 = c['bbox']
                    chars.append(((y0 + y1) / 2, x0, x1, ch, s['size']))
    if not chars:
        return []
    chars.sort(key=lambda t: t[0])
    rows, cur = [], [chars[0]]
    for c in chars[1:]:
        (cur if abs(c[0] - cur[-1][0]) <= ytol else rows).append(c if abs(c[0] - cur[-1][0]) <= ytol else cur)
        if abs(c[0] - cur[-1][0]) > ytol:
            cur = [c]
    rows.append(cur)
    out = []
    for r in rows:
        r.sort(key=lambda t: -t[2])
        s, prev = '', None
        for y, x0, x1, ch, sz in r:
            if prev is not None and prev - x1 > gap:
                s += ' '
            s += ch
            prev = x0
        out.append({'y': r[0][0], 'size': round(max(t[4] for t in r), 1), 'text': fix_visual(s.strip())})
    return out


UNSUPPORTED = {5784, 5785, 5786, 5787}   # two-column layout; text comes from the site API instead
OCR_YEARS = {5768, 5769, 5770, 5774, 5775, 5776, 5777, 5778, 5779, 5780}


def extract_year(year, pdf_dir, out_dir, footer_y=10000.0, min_size=8.0):
    if year in UNSUPPORTED:
        return {'year': year, 'skipped': 'two-column layout; use data/days/<year> from the site API'}
    if year in OCR_YEARS:
        return {'year': year, 'skipped': 'garbled text layer; OCR required'}
    pdf = pdf_dir / YEAR_PDF[year]
    doc = pymupdf.open(pdf)
    days, cur = [], None
    sizes = Counter()
    for pno in range(len(doc)):
        for ln in page_lines(doc[pno]):
            if ln['y'] > footer_y or ln['size'] < min_size or not ln['text']:
                continue
            t = ln['text']
            if HEADER_RE.search(t) and ln['y'] < 60 and ln['size'] > 11 and re.search(r'בס.?ד', t):   # tolerant of a mangled gershayim (font-map decoded years)
                cur = {'year': year, 'header': t, 'source': None, 'num': None, 'lines': [], 'pages': [pno + 1], 'phase': 'summary'}
                days.append(cur)
                continue
            if cur is None:
                continue
            if pno + 1 not in cur['pages']:
                cur['pages'].append(pno + 1)
            if cur['phase'] == 'summary':
                m = NUM_RE.search(t)
                if m:
                    cur['num'] = m.group(1)
                    cur['source'] = t
                    cur['phase'] = 'marker'
                    continue
                if MARKER_RE.search(t):          # older layout: no numbered source line
                    cur['marker'] = t
                    cur['phase'] = 'body'
                    continue
                if ln['size'] >= 11.5 and ln['y'] > 120:   # body started without any marker
                    cur['phase'] = 'body'
                else:
                    continue
            if cur['phase'] == 'marker':
                cur['phase'] = 'body'
                if MARKER_RE.search(t):
                    cur['marker'] = t
                    continue
            if cur['phase'] == 'body':
                cur['lines'].append(t)
                sizes[ln['size']] += len(t)
    body_size = sizes.most_common(1)[0][0] if sizes else None
    if days and not any(d['num'] for d in days):        # number days sequentially (5771-5773 layout)
        for i, d in enumerate(days, 1):
            d['num'] = f'{i:03d}'
            d['num_source'] = 'sequential'
    od = out_dir / str(year)
    od.mkdir(parents=True, exist_ok=True)
    audio = {}
    adir = ROOT / AUDIO_DIR[year] if year in AUDIO_DIR else None
    if adir is not None and adir.is_dir():
        for p in adir.rglob('*.mp3'):
            audio.setdefault(p.stem[:3], []).append(str(p.relative_to(ROOT)))
    written, missing_num, dup = 0, 0, Counter()
    for d in days:
        if not d['num']:
            missing_num += 1
            continue
        dup[d['num']] += 1
        body = '\n'.join(d['lines']).strip()
        rec = {'num': d['num'], 'num_source': d.get('num_source', 'printed'), 'year': year, 'header': d['header'], 'source': d['source'],
               'marker': d.get('marker'), 'body': body, 'pages': d['pages'],
               'n_words': len(re.findall(r'[א-ת]+', body)),
               'mp3': audio.get(d['num'], [])}
        name = d['num'] if dup[d['num']] == 1 else f"{d['num']}_dup{dup[d['num']]}"
        (od / f'{name}.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
        written += 1
    nums = {d['num'] for d in days if d['num']}
    summary = {'year': year, 'pdf': pdf.name, 'pages': len(doc), 'days_found': len(days), 'days_written': written,
               'days_without_number': missing_num, 'duplicate_numbers': [k for k, v in dup.items() if v > 1],
               'body_font_size': body_size, 'audio_files': sum(len(v) for v in audio.values()),
               'matched_to_audio': len([n for n in nums if n in audio]),
               'audio_without_text': sorted(set(audio) - nums), 'text_without_audio': sorted(nums - set(audio)) if audio else [],
               'total_words': sum(len(re.findall(r'[א-ת]+', '\n'.join(d['lines']))) for d in days)}
    (od / '_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, nargs='+', required=True)
    ap.add_argument('--pdf-dir', default=str(PDF_DIR))
    ap.add_argument('--out', default=str(ROOT / 'asr' / 'data' / 'pdf_text'))
    a = ap.parse_args()
    for y in a.year:
        s = extract_year(y, Path(a.pdf_dir), Path(a.out))
        print(json.dumps(s, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
