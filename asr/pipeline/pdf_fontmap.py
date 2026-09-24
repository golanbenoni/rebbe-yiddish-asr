#!/usr/bin/env python3
"""Decode the Daily Sicha yearly PDFs whose text layer is garbled by custom font encodings (5768-5770, 5774-5780).

The glyph codes PyMuPDF returns for these fonts are a fixed permutation of the Hebrew alphabet per font, so the
text can be recovered EXACTLY (no OCR): for each font we learn the code -> letter map by hill-climbing the share of
decoded words that exist in the training vocabulary (data/manifests/train.clean.jsonl), starting from a
letter-frequency ranking. A random map scores ~0% in-vocabulary, the right one > 90%. Tesseract on this typeface
scored 0.39 WER (2026-09-23), so this is both cleaner and faster.

  python -u pipeline/pdf_fontmap.py --year 5778 [--year ...]        # learn maps, decode, split days like pdf_hanachos.py
  -> data/pdf_text/<year>/NNN.json (+ _summary.json, _fontmap.json)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pdf_hanachos as ph  # noqa: E402
from normalize import scoring_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LETTERS = [chr(c) for c in range(0x5D0, 0x5EB)]          # 27 Hebrew letters incl. finals
SYMBOLS = ["'", '"', "-", ""]                              # codes that are not letters; "" = nikkud/combining glyph (dropped)
KEEP_ASCII = set(".,;:!?()[]{}0123456789 *\"'-\u2013\u2014")   # ASCII punctuation, quotes and dashes are literal in these PDFs


def vocab() -> tuple[set[str], list[str]]:
    """Known words (count >= 2) of the clean training text and the token-weighted Hebrew letter frequency order."""
    cnt: Counter = Counter(); letters: Counter = Counter()
    for line in open(ROOT / "asr" / "data" / "manifests" / "train.clean.jsonl", encoding="utf-8"):
        toks = scoring_text(json.loads(line)["text"]).split()
        cnt.update(toks)
        for t in toks:
            letters.update(t)
    order = [c for c, _ in letters.most_common() if c in LETTERS] + [c for c in LETTERS if c not in letters]
    tot = sum(v for k, v in letters.items() if k in LETTERS)
    global LETTER_DIST
    LETTER_DIST = {L: letters.get(L, 0) / max(1, tot) for L in LETTERS}
    return {w for w, c in cnt.items() if c >= 2}, order


def header_vocab() -> set[str]:
    """Calendar/header words (months, weekdays, parshiyot, year names, 'השיחה היומית' ...) from the clean-text years:
    the spoken corpus barely contains them, yet every day's header and source line is made of them."""
    words: set[str] = set()
    for f in (ROOT / "asr" / "data" / "pdf_text").glob("57[78][0-9]/[0-9]*.json"):
        rec = json.loads(f.read_text(encoding="utf-8"))
        for k in ("header", "source", "marker"):
            if rec.get(k):
                words.update(w for w in scoring_text(rec[k]).split() if len(w) >= 2)
    return words


def fix_quotes(assign: dict, samples: list[str]) -> dict:
    """Decide which symbol code is gershayim: it sits second-to-last in abbreviations (רש"י, ע"י, ה'תשע"ח), while
    the geresh ends words (גדלי') or follows a one-letter prefix (ס'איז, מ'האט)."""
    sym_codes = [c for c, v in assign.items() if v in QUOTES]
    if len(sym_codes) < 2:
        return assign
    penult: Counter = Counter(); total: Counter = Counter()
    for w in samples:
        for i, c in enumerate(w):
            if c in sym_codes:
                total[c] += 1
                if len(w) >= 3 and i == len(w) - 2:
                    penult[c] += 1
    ranked = sorted(sym_codes, key=lambda c: -(penult[c] / max(1, total[c])))
    assign = dict(assign)
    assign[ranked[0]] = '"'
    for c in ranked[1:]:
        assign[c] = "'"
    return assign


MAX_GLYPH_SIZE = 40.0   # 5776-5778 carry a diagonal watermark (size ~80-85) whose letters land on body rows and glue words


def page_chars(page, max_size: float = MAX_GLYPH_SIZE):
    """(y_center, x0, x1, code, size, font) for every glyph on the page (rawdict). Only U+0020 is a space; U+00A0 and
    other "whitespace" code points are letter glyphs in these fonts. Glyphs larger than max_size (watermark) are dropped."""
    out = []
    for b in page.get_text("rawdict")["blocks"]:
        if b.get("type", 0) != 0:
            continue
        for l in b["lines"]:
            for s in l["spans"]:
                if s["size"] > max_size:
                    continue
                for c in s["chars"]:
                    ch = ph.BIDI.sub("", c["c"])
                    if not ch:
                        continue                     # control-range codes are LETTERS in the 5774-5776 encodings
                    x0, y0, x1, y1 = c["bbox"]
                    out.append(((y0 + y1) / 2, x0, x1, ch, s["size"], s["font"]))
    return out


def rows_of(chars, ytol=2.5):
    """Cluster glyphs into baseline rows, each row ordered right-to-left."""
    if not chars:
        return []
    chars = sorted(chars, key=lambda t: t[0])
    rows, cur = [], [chars[0]]
    for c in chars[1:]:
        if abs(c[0] - cur[-1][0]) <= ytol:
            cur.append(c)
        else:
            rows.append(cur); cur = [c]
    rows.append(cur)
    for r in rows:
        r.sort(key=lambda t: -t[2])
    return rows


def row_tokens(row, gap=2.5):
    """Split one RTL row into word tokens: lists of (code, font). A U+0020 glyph is a word space only when it does
    not overlap the previous glyph (overlapping spaces just position a vowel mark); a large gap also separates words.
    KEEP_ASCII punctuation becomes its own token."""
    tokens, cur, prev_x0 = [], [], None
    for y, x0, x1, ch, sz, font in row:
        if ch == " ":
            if prev_x0 is None or x1 <= prev_x0 + 0.5:
                if cur:
                    tokens.append(cur); cur = []
            continue
        if prev_x0 is not None and prev_x0 - x1 > gap and cur:
            tokens.append(cur); cur = []
        cur.append((ch, font))            # punctuation stays attached to its word (בס"ד, ה'תשע"ח, (אף, בפועל.)
        if x1 > x0:                      # zero-width marks do not move the pen
            prev_x0 = x0 if prev_x0 is None else min(prev_x0, x0)
    if cur:
        tokens.append(cur)
    return tokens


def decode_token(tok, maps):
    out = ""
    for ch, font in tok:
        if ch in KEEP_ASCII:
            out += ch
        else:
            out += maps.get(font, {}).get(ch, ch) if maps is not None else ch
    return out


def rebuild_lines(chars, maps=None, ytol=2.5, gap=2.5):
    """Rows -> lines of decoded text (spaces between tokens; punctuation tokens glued to the preceding word)."""
    out = []
    for r in rows_of(chars, ytol):
        toks = row_tokens(r, gap)
        text = " ".join(d for d in (decode_token(tok, maps) for tok in toks) if d)
        text = ph.fix_visual(re.sub(r" +", " ", text).strip())
        fonts = Counter(f for c, f in (x for tok in toks for x in tok))
        out.append({"y": r[0][0], "size": round(max(t[4] for t in r), 1), "text": text,
                    "font": fonts.most_common(1)[0][0] if fonts else ""})
    return out


ZERO_WIDTH: dict = defaultdict(Counter)   # font -> Counter(code) of zero-width occurrences (vowel marks)


def collect_words(doc, max_pages=60):
    """Per font: code counts and word samples (code strings) from up to max_pages evenly spaced pages.
    Also records which codes are zero-width glyphs (vowel marks) so they are dropped, not mapped."""
    counts: dict[str, Counter] = defaultdict(Counter); words: dict[str, list] = defaultdict(list)
    step = max(1, len(doc) // max_pages)
    for pno in range(0, len(doc), step):
        for r in rows_of(page_chars(doc[pno])):
            for y, x0, x1, ch, sz, font in r:
                if x1 - x0 < 0.05 and ch not in KEEP_ASCII and ch != " ":
                    ZERO_WIDTH[font][ch] += 1
            for tok in row_tokens(r):
                if len(tok) == 1 and tok[0][0] in KEEP_ASCII:
                    continue
                font = Counter(f for _, f in tok).most_common(1)[0][0]
                codes = [c for c, f in tok if c not in KEEP_ASCII]
                if not codes:
                    continue
                words[font].append("".join(codes))
                counts[font].update(codes)
    return counts, words



# --- header crib -------------------------------------------------------------------------------------------------
# Every day's first page carries the same bold header line  בס"ד. "השיחה היומית" ליום <day>', <date> <month>, ה'תשע"ה
# and the label התוכן under it. Its fixed words pin 15+ of the 27 letters of the bold font before any search, which
# is what the hill-climb needs: the bold font's samples are mostly headers, and on 5775/5776 the unguided search
# settled in a wrong permutation (in-vocab 0.03 / 0.51, no day found).
YEAR_HEB = {5768: 'תשס"ח', 5769: 'תשס"ט', 5770: 'תש"ע', 5771: 'תשע"א', 5772: 'תשע"ב', 5773: 'תשע"ג', 5774: 'תשע"ד',
            5775: 'תשע"ה', 5776: 'תשע"ו', 5777: 'תשע"ז', 5778: 'תשע"ח', 5779: 'תשע"ט', 5780: 'תש"פ', 5781: 'תשפ"א', 5782: 'תשפ"ב'}
HEADER_PREFIX = 'בס"ד. "השיחה היומית" ליום'
CONTENTS_LABEL = "התוכן"
MONTHS = ["תשרי", "חשון", "מרחשון", "כסלו", "טבת", "שבט", "אדר", "ניסן", "אייר", "סיון", "תמוז", "אב", "אלול"]


def _is_letter(ch: str) -> bool:
    return "א" <= ch <= "ת"


def _shape_text(t: str) -> str:
    return "".join("L" if _is_letter(ch) else ch for ch in t)


def _row_seq(row, font):
    """Codes of one row in reading order, ' ' between tokens, zero-width marks dropped; None unless the row is set
    mostly in `font`."""
    row = [t for t in row if t[3] == " " or t[3] in KEEP_ASCII or t[2] - t[1] >= 0.05]
    toks = row_tokens(row)
    fonts = Counter(f for tok in toks for _, f in tok)
    if not fonts or fonts.most_common(1)[0][0] != font:
        return None
    seq: list[str] = []
    for tok in toks:
        if seq:
            seq.append(" ")
        seq.extend(ch for ch, _ in tok)
    return seq


def _confident(votes: dict, min_votes: int = 3, min_share: float = 0.8) -> dict:
    out = {}
    for c, cnt in votes.items():
        L, n = cnt.most_common(1)[0]
        if sum(cnt.values()) >= min_votes and n >= min_share * sum(cnt.values()):
            out[c] = (L, n)
    # injective: a letter claimed by two codes goes to the better-supported one
    best: dict = {}
    for c, (L, n) in out.items():
        if L not in best or n > out[best[L]][1]:
            best[L] = c
    return {c: L for L, c in best.items()}


def crib_fixed(doc, font: str, year: int, y_max: float = 60.0) -> tuple[dict, int]:
    """code -> letter assignments read off the fixed header words (+ the month name once the header letters pin it
    down). Returns (fixed map, number of header rows matched)."""
    if year not in YEAR_HEB:
        return {}, 0
    votes: dict = defaultdict(Counter)
    prefix_shape = _shape_text(HEADER_PREFIX)
    year_text = "ה'" + YEAR_HEB[year]; year_shape = _shape_text(year_text)
    heads = []   # (seq, page rows) for the month pass
    for page in doc:
        rows = rows_of(page_chars(page))
        for i, r in enumerate(rows):
            if r[0][0] > y_max or max(t[4] for t in r) < 11:
                continue
            seq = _row_seq(r, font)
            if seq is None:
                continue
            shape = "".join("L" if (c != " " and c not in KEEP_ASCII) else c for c in seq)
            if not shape.startswith(prefix_shape):
                continue
            for k, ch in enumerate(HEADER_PREFIX):
                if _is_letter(ch):
                    votes[seq[k]][ch] += 1
            if shape.endswith(year_shape) and shape[-len(year_shape) - 1] == " ":
                for k, ch in enumerate(year_text):
                    if _is_letter(ch):
                        votes[seq[len(seq) - len(year_text) + k]][ch] += 1
            heads.append(seq)
            for r2 in rows[i + 1:i + 4]:            # the התוכן label sits a few rows under the header
                if r2[0][0] > 90:
                    break
                s2 = _row_seq(r2, font)
                if s2 is not None and len(s2) == len(CONTENTS_LABEL) and all(c != " " and c not in KEEP_ASCII for c in s2):
                    for k, ch in enumerate(CONTENTS_LABEL):
                        votes[s2[k]][ch] += 1
                    break
    # month names: the third token after the prefix, disambiguated by the letters already pinned
    for _ in range(3):
        conf = _confident(votes)
        taken = set(conf.values())
        for seq in heads:
            rest = "".join(seq[len(HEADER_PREFIX):]).strip().split(" ")
            if len(rest) < 4:
                continue
            mon = [c for c in rest[-2].rstrip(",") if c not in KEEP_ASCII]
            cands = []
            for m in MONTHS:
                if len(m) != len(mon):
                    continue
                ok = True; local: dict = {}
                for c, L in zip(mon, m):
                    if c in conf:
                        ok &= conf[c] == L
                    else:
                        ok &= (L not in taken or L in local.values()) and local.setdefault(c, L) == L
                if ok:
                    cands.append(m)
            if len(cands) == 1:
                for c, L in zip(mon, cands[0]):
                    votes[c][L] += 1
    return _confident(votes), len(heads)


def apply_fixed(m: dict, fixed: dict) -> dict:
    """Force the crib letters into a map, keeping it injective (the displaced letter goes to the displaced code)."""
    for c, L in fixed.items():
        if m.get(c) == L:
            continue
        other = next((k for k, v in m.items() if v == L and k != c), None)
        old = m.get(c, "")
        m[c] = L
        if other is not None:
            m[other] = old
    return m


QUOTES = {"'", '"'}
LETTER_DIST: dict = {}
DIST_WEIGHT = 1.0   # penalty weight on the L1 distance between decoded and corpus letter distributions


def _word_ok(d: str, voc: set[str]) -> bool:
    """Decoded word counts if its letters (2+) form a known word and any symbols inside are only geresh/gershayim
    (רש"י, גדלי') - at most two of them; hyphen-mapped codes inside a word make it wrong."""
    letters = "".join(ch for ch in d if "\u05d0" <= ch <= "\u05ea")
    syms = [ch for ch in d if not ("\u05d0" <= ch <= "\u05ea")]
    if len(syms) > 2 or any(ch not in QUOTES for ch in syms):
        return False
    if len(letters) == 1:
        return len(syms) == 1 and d.endswith("'")      # abbreviations like ג' / פ' in the headers
    return len(letters) >= 2 and letters in voc


def _dist_penalty(assign: dict, code_counts: Counter) -> float:
    """L1 distance between the letter distribution implied by the map and the corpus distribution: stops the search
    from parking a rare letter on a frequent glyph or dropping a letter altogether."""
    if not LETTER_DIST:
        return 0.0
    dec: Counter = Counter()
    for c, n in code_counts.items():
        t = assign.get(c, "")
        if t in LETTER_DIST:
            dec[t] += n
    tot = sum(dec.values()) or 1
    return sum(abs(dec.get(L, 0) / tot - LETTER_DIST[L]) for L in LETTERS)


def score_map(assign: dict, samples: list[str], voc: set[str], code_counts: Counter | None = None) -> float:
    sample = samples[:3000]
    ok = sum(1 for w in sample if _word_ok("".join(assign.get(c, "?") for c in w), voc)) / max(1, len(sample))
    return ok - (DIST_WEIGHT * _dist_penalty(assign, code_counts) if code_counts else 0.0)


def learn_map(codes: Counter, samples: list[str], voc: set[str], letter_freq: list[str], iters: int = 12, seed: int = 0, font_name: str | None = None, dist_weight: float = 1.0, fixed: dict | None = None) -> tuple[dict, float]:
    """Learn a bijection code -> letter for 27 codes (the rest -> symbols) maximizing the share of known decoded words.
    Start from frequency-rank matching, then alternate (a) guided moves voted by near-miss words (a decoded word one
    letter away from a known word votes for that code -> letter correction) and (b) a full pairwise-swap sweep."""
    fixed = {c: L for c, L in (fixed or {}).items() if c in codes}
    marks = {c for c in codes if ZERO_WIDTH.get(font_name, {}).get(c, 0) >= 0.9 * codes[c]} if font_name else set()
    marks -= set(fixed)
    codes_sorted = [c for c, _ in codes.most_common() if c not in marks and c not in fixed]   # the free codes
    free_letters = [L for L in letter_freq if L not in set(fixed.values())][:len(LETTERS) - len(fixed)]
    n_sym = max(0, len(codes_sorted) - len(free_letters))
    targets = free_letters + (["'", '"', "-"] + [""] * n_sym)[:n_sym]
    if len(codes_sorted) < len(free_letters):
        targets = free_letters[:len(codes_sorted)]
    assign = dict(zip(codes_sorted, targets))
    assign.update(fixed)
    for c in marks:
        assign[c] = ""
    rng = random.Random(seed)
    sample = rng.sample(samples, min(3000, len(samples)))
    sample_codes = [list(w) for w in sample]
    by_letter_len: dict[tuple, set] = defaultdict(set)   # (len, i, letter-with-hole) -> known words, for near-miss lookup
    for w in voc:
        if 3 <= len(w) <= 12:
            for i in range(len(w)):
                by_letter_len[(len(w), i, w[:i] + "_" + w[i + 1:])].add(w[i])

    sample_counts: Counter = Counter()
    for w in sample_codes:
        sample_counts.update(w)

    def score(a):
        ok = sum(1 for w in sample_codes if _word_ok("".join(a.get(c, "?") for c in w), voc)) / max(1, len(sample_codes))
        return ok - dist_weight * _dist_penalty(a, sample_counts)

    def swap_to(c, target):
        """make code c map to target, giving c's old target to whoever had target (keeps the bijection)"""
        old = assign[c]
        other = next((k for k, v in assign.items() if v == target and k != c), None)
        assign[c] = target
        if other is not None:
            assign[other] = old
        return old, other

    def undo(c, target, old, other):
        assign[c] = old
        if other is not None:
            assign[other] = target

    best = score(assign)
    for it in range(iters):
        # (a) guided votes
        votes: Counter = Counter()
        for w in sample_codes:
            d = "".join(assign.get(c, "?") for c in w)
            if len(d) < 3 or len(d) > 12 or _word_ok(d, voc) or not all("\u05d0" <= ch <= "\u05ea" for ch in d):
                continue
            for i in range(len(d)):
                cands = by_letter_len.get((len(d), i, d[:i] + "_" + d[i + 1:]), ())
                for L in cands:
                    if L != d[i]:
                        votes[(w[i], L)] += 1.0 / len(cands)
        moved = False
        for (c, L), v in votes.most_common(40):
            if assign.get(c) == L or c in fixed or any(k in fixed and v2 == L for k, v2 in assign.items()):
                continue
            old, other = swap_to(c, L)
            s_new = score(assign)
            if s_new > best + 1e-9:
                best = s_new; moved = True
            else:
                undo(c, L, old, other)
        # (b) pairwise sweep
        improved = False
        for i in range(len(codes_sorted)):
            for j in range(i + 1, len(codes_sorted)):
                c1, c2 = codes_sorted[i], codes_sorted[j]
                if assign[c1] == assign[c2]:
                    continue
                assign[c1], assign[c2] = assign[c2], assign[c1]
                s_new = score(assign)
                if s_new > best + 1e-9:
                    best = s_new; improved = True
                else:
                    assign[c1], assign[c2] = assign[c2], assign[c1]
        if not moved and not improved:
            break
    # (c) many-to-one refinement: Yiddish typography has several glyphs per letter (פּ/פ, בּ/ב, כּ/כ, תּ/ת, שׂ/ש,
    # וּ/ו, יִ/י) and ligatures (וו, וי, יי); the bijection above parks the extra glyphs on symbol slots. Let every
    # code try every letter and ligature, non-bijectively, keeping any move that raises the word score.
    LIGATURES = ["\u05d5\u05d5", "\u05d5\u05d9", "\u05d9\u05d9"]
    def word_score(a):
        return sum(1 for w in sample_codes if _word_ok("".join(a.get(c, "?") for c in w), voc)) / max(1, len(sample_codes))
    cur = word_score(assign)
    for _ in range(2):
        changed = False
        for c in codes_sorted:
            old = assign[c]; best_t, best_s = old, cur
            for t in LETTERS + LIGATURES + ["'", '"', "-", ""]:
                if t == old:
                    continue
                assign[c] = t
                s_new = word_score(assign)
                if s_new > best_s + 1e-9:
                    best_t, best_s = t, s_new
            assign[c] = best_t
            if best_t != old:
                cur = best_s; changed = True
        if not changed:
            break
    return assign, cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, nargs="+", required=True)
    ap.add_argument("--pdf-dir", default=str(ph.PDF_DIR))
    ap.add_argument("--out", default=str(ROOT / "asr" / "data" / "pdf_text"))
    ap.add_argument("--min-samples", type=int, default=300, help="fonts with fewer sampled words are left undecoded")
    ap.add_argument("--reuse-maps", action="store_true", help="load data/pdf_text/<year>/_fontmap.json instead of re-learning (extraction only)")
    ap.add_argument("--footer-y", type=float, default=10000.0, help="rows below this y are dropped (disabled: the body runs to the page bottom on most pages)")
    ap.add_argument("--min-size", type=float, default=8.0, help="rows set smaller than this are dropped (some days are typeset at 10 pt, so the footer is removed by font, not size)")
    ap.add_argument("--body-fonts", nargs="+", default=["David"], help="keep only rows whose dominant font name contains one of these; the footer small print (Arial/Tahoma), page numbers and the watermark use other fonts")
    args = ap.parse_args()
    voc, letter_freq = vocab()
    voc |= header_vocab()
    for year in args.year:
        pdf = Path(args.pdf_dir) / ph.YEAR_PDF[year]
        doc = pymupdf.open(pdf)
        counts, words = collect_words(doc, max_pages=120)
        maps, report = {}, {}
        fonts = sorted(counts.items(), key=lambda kv: -sum(kv[1].values()))
        body_map = None
        od = Path(args.out) / str(year)
        if args.reuse_maps and (od / "_fontmap.json").exists():
            maps = json.loads((od / "_fontmap.json").read_text(encoding="utf-8"))
            try:
                report = json.loads((od / "_summary.json").read_text(encoding="utf-8")).get("decode", {}).get("fonts", {})
            except Exception:
                report = {}
            print(f"{year}: reusing {len(maps)} font maps from {od / '_fontmap.json'}", flush=True)
            fonts = []
        for font, cnt in fonts:
            if len(words[font]) < args.min_samples:
                report[font] = {"codes": len(cnt), "words": len(words[font]), "skipped": "too few samples"}; continue
            # the hill-climb is seed-sensitive (one start in four lands in a poor basin): keep the best of several seeds
            fx, n_heads = crib_fixed(doc, font, year)
            m, s = max((learn_map(cnt, words[font], voc, letter_freq, font_name=font, dist_weight=1.0 if body_map is None else 0.2, seed=sd, fixed=fx) for sd in range(4)), key=lambda r: r[1])
            m = apply_fixed(fix_quotes(m, words[font]), fx); s = score_map(m, words[font], voc, cnt)
            if body_map is not None:   # same family (David / David,Bold): the body map often fits the other font better
                t = score_map(apply_fixed({c: body_map.get(c, "?") for c in cnt}, fx), words[font], voc, cnt)
                if t > s:
                    m, s = apply_fixed({c: body_map.get(c, "?") for c in cnt}, fx), t
            if body_map is None:
                body_map = m
            maps[font] = m; report[font] = {"codes": len(cnt), "words": len(words[font]), "in_vocab": round(s, 3), "crib_rows": n_heads, "crib_fixed": len(fx)}
            print(f"{year} font {font}: {len(cnt)} codes, {len(words[font])} sample words, crib {n_heads} header rows -> {len(fx)} letters pinned, in-vocab share {score_map(m, words[font], voc):.3f}", flush=True)
        # decode + split days with pdf_hanachos' logic
        ph.OCR_YEARS = set()
        body_ok = lambda font: any(tag in font for tag in args.body_fonts)
        ph.page_lines = lambda page, ytol=2.5, gap=1.2: [ln for ln in rebuild_lines(page_chars(page), maps, ytol, gap) if body_ok(ln["font"])]
        summary = ph.extract_year(year, Path(args.pdf_dir), Path(args.out), footer_y=args.footer_y, min_size=args.min_size)
        summary["decode"] = {"method": "font-map", "fonts": report, "footer_y": args.footer_y, "min_size": args.min_size, "body_fonts": args.body_fonts, "max_glyph_size": MAX_GLYPH_SIZE}
        (od / "_fontmap.json").write_text(json.dumps({f: {k: v for k, v in m.items()} for f, m in maps.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
        (od / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "decode"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
