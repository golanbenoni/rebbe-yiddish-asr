#!/usr/bin/env python3
"""PRELIMINARY scorer: whole-file WER/CER of the Phase 7 drafts (asr/output/<year folder>/*.txt) against the
Yiddish hanachos extracted from the yearly PDFs (data/pdf_text/<year>/NNN.json by pipeline/pdf_hanachos.py).
Reference = PDF body with a trailing dedication (short tail after the last lone '*' line) removed; both sides go
through normalize.scoring_text. WER is corpus-level (sum of edits / sum of ref words), same convention as
evaluate.py, but computed with rapidfuzz Levenshtein here rather than jiwer: treat as preliminary until
evaluate_longform.py is re-run with these references on the fleet.
Usage (from asr/): python -u pipeline/score_drafts_vs_pdf.py --year 5781 [--shard i/n] --out reports/drafts-vs-pdf
"""
import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import scoring_text  # noqa: E402
ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = {5778: "DS 5778", 5779: "Sicha Yomis 5779 audio", 5780: "Sicha Yomis 5780 audio", 5781: "Sicha Yomis 5781 audio"}


def edits(ref, hyp):
    """Levenshtein distance between two token lists (rapidfuzz; pip install rapidfuzz)."""
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(ref, hyp)


DEDICATION = re.compile(r'^(לזכות|לע"נ|לעילוי|מוקדש|נדבת|לרפואה|לזכרון|ע"י|נתרם)')


def ref_text(rec):
    """Body without the trailing dedication: '*' also separates parts within a day, so only cut after the
    LAST '*' and only when the tail is short and reads like a dedication."""
    lines = rec['body'].split('\n')
    if '*' in lines:
        k = len(lines) - 1 - lines[::-1].index('*')
        tail = [l for l in lines[k + 1:] if l.strip()]
        if tail and sum(len(l.split()) for l in tail) < 80 and any(DEDICATION.match(l) for l in tail):
            lines = lines[:k]
    return '\n'.join(l for l in lines if l.strip() != '*')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--out', default='reports/drafts-vs-pdf')
    ap.add_argument('--hyp-root', default=str(ROOT / 'asr' / 'output'), help='root holding <year folder>/<month>/<stem>.txt')
    ap.add_argument('--only', default=None, help='file with mp3 stems to score, one per line')
    a = ap.parse_args()
    only = set(Path(a.only).read_text().split('\n')) - {''} if a.only else None
    i, n = map(int, a.shard.split('/'))
    recs = sorted((ROOT / 'asr' / 'data' / 'pdf_text' / str(a.year)).glob('[0-9]*.json'))[i::n]
    out = ROOT / 'asr' / a.out; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in recs:
        rec = json.loads(p.read_text(encoding='utf-8'))
        if not rec['mp3']: continue
        stem = Path(rec['mp3'][0]).stem
        month = Path(rec['mp3'][0]).parent.name
        if only is not None and stem not in only: continue
        hyp_p = Path(a.hyp_root) / AUDIO_DIR[a.year] / month / f'{stem}.txt'
        if not hyp_p.exists():
            rows.append({'num': rec['num'], 'stem': stem, 'error': 'no draft'}); continue
        r = scoring_text(ref_text(rec)); h = scoring_text(hyp_p.read_text(encoding='utf-8'))
        rw, hw = r.split(), h.split()
        we = edits(rw, hw); ce = edits(list(r.replace(' ', '')), list(h.replace(' ', '')))
        rows.append({'num': rec['num'], 'stem': stem, 'ref_words': len(rw), 'hyp_words': len(hw), 'word_edits': we,
                     'wer': round(we / max(1, len(rw)), 4), 'ref_chars': len(r.replace(' ', '')), 'char_edits': ce,
                     'cer': round(ce / max(1, len(r.replace(' ', ''))), 4)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    (out / f'{a.year}-shard{i}of{n}.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
