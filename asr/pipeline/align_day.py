#!/usr/bin/env python3
"""Align a day's Yiddish hanacha to its audio when the site has no timing file.

Method: transcribe with the best available Yiddish model (word timestamps on),
align the noisy hypothesis to the reference text word by word (exact anchors,
then fuzzy matches inside the gaps), interpolate times for unmatched reference
words, and cut <= --max-seconds chunks at paragraph, sentence and pause
boundaries. The result is stored in the cached day JSON as `sync_local`, in the
same shape as the site's sync (segments with start/end/yiddish) plus
diagnostics, so build_dataset.py can use either. When the site's own sync exists
the script also measures how far its segment times are from ours, which is the
built-in validation of the method.

  python align_day.py --days "001 3-Tishrei 5787"                 # validate against site sync
  python align_day.py --all-missing --year 5786                   # every 5786 day without site sync
  python align_day.py --all-missing --device cuda --compute-type float16 --model ivrit-ai/yi-whisper-large-v3-ct2
  for i in 0 1 2; do python align_day.py --all-missing --shard $i/3 & done; wait   # three processes in parallel
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

import jiwer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio import decode_16k_mono  # noqa: E402
from normalize import hanacha_paragraphs, scoring_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "asr" / "data"
SR = 16000
BRACKETS = re.compile(r"\[[^\]]*\]")
HYPHEN_SPLIT = re.compile(r"([־–—-])")
WORDS_PER_SEC = 2.8  # used only to extrapolate before the first / after the last anchor


def ref_tokens(paras: list[str], keep_brackets: bool) -> list[dict]:
    toks = []
    for pi, para in enumerate(paras):
        if not keep_brackets:
            para = BRACKETS.sub(" ", para)
        for raw in para.split():
            parts = HYPHEN_SPLIT.split(raw)
            sep = " "
            for part in parts:
                if not part:
                    continue
                if HYPHEN_SPLIT.fullmatch(part):
                    sep = part; continue
                norm = scoring_text(part).replace(" ", "")
                if not norm:  # bare punctuation: glue to previous token
                    if toks: toks[-1]["surface"] += part
                    continue
                toks.append({"surface": part, "sep": sep, "norm": norm, "para": pi,
                             "sent_end": part[-1] in ".?!:", "comma": part[-1] in ",;", "para_end": False})
                sep = " "
        if toks and toks[-1]["para"] == pi:
            toks[-1]["para_end"] = True
    return toks


def transcribe_words(audio: np.ndarray, model_name: str, device: str, compute_type: str, language: str) -> list[dict]:
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segs, _ = model.transcribe(audio, language=language, task="transcribe", beam_size=5, word_timestamps=True,
                               vad_filter=True, vad_parameters={"min_silence_duration_ms": 300},
                               condition_on_previous_text=False)
    words = []
    for s in segs:
        for w in s.words or []:
            norms = scoring_text(w.word).split()
            if not norms:
                continue
            span = (w.end - w.start) / len(norms)
            for k, n in enumerate(norms):
                words.append({"norm": n, "start": w.start + k * span, "end": w.start + (k + 1) * span, "p": w.probability})
    return words


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def align(ref: list[dict], hyp: list[dict], fuzzy_min: float) -> dict[int, int]:
    rn = [t["norm"] for t in ref]; hn = [w["norm"] for w in hyp]
    anchors: dict[int, int] = {}
    for a, b, n in difflib.SequenceMatcher(None, rn, hn, autojunk=False).get_matching_blocks():
        for k in range(n):
            anchors[a + k] = b + k
    # fuzzy pass inside gaps between exact anchors (monotone DP on similarity)
    keys = sorted(anchors)
    bounds = [(-1, -1)] + [(k, anchors[k]) for k in keys] + [(len(ref), len(hyp))]
    for (ra, ha), (rb, hb) in zip(bounds, bounds[1:]):
        R = list(range(ra + 1, rb)); H = list(range(ha + 1, hb))
        if not R or not H or len(R) * len(H) > 40000:
            continue
        score = np.zeros((len(R) + 1, len(H) + 1)); choice = np.zeros((len(R) + 1, len(H) + 1), dtype=np.int8)
        for i in range(1, len(R) + 1):
            for j in range(1, len(H) + 1):
                s = similarity(rn[R[i - 1]], hn[H[j - 1]])
                best = score[i - 1, j]; c = 1
                if score[i, j - 1] > best: best, c = score[i, j - 1], 2
                if s >= fuzzy_min and score[i - 1, j - 1] + s > best: best, c = score[i - 1, j - 1] + s, 3
                score[i, j], choice[i, j] = best, c
        i, j = len(R), len(H)
        while i > 0 and j > 0:
            c = choice[i, j]
            if c == 3: anchors[R[i - 1]] = H[j - 1]; i -= 1; j -= 1
            elif c == 1: i -= 1
            else: j -= 1
    return anchors


def assign_times(ref: list[dict], hyp: list[dict], anchors: dict[int, int], duration: float, intro_end: float) -> None:
    for i, t in enumerate(ref):
        t["anchored"] = i in anchors
        if t["anchored"]:
            t["start"], t["end"] = hyp[anchors[i]]["start"], hyp[anchors[i]]["end"]
    idx = [i for i, t in enumerate(ref) if t["anchored"]]
    if not idx:
        raise RuntimeError("no anchors at all")
    # before first anchor
    first = idx[0]
    if first > 0:
        n = first; end = ref[first]["start"]; start = max(intro_end, end - n / WORDS_PER_SEC)
        for k, i in enumerate(range(first)):
            ref[i]["start"] = start + (end - start) * k / n; ref[i]["end"] = start + (end - start) * (k + 1) / n
    # between anchors: distribute by character length
    for a, b in zip(idx, idx[1:]):
        if b - a > 1:
            gap_start, gap_end = ref[a]["end"], ref[b]["start"]
            if gap_end < gap_start: gap_end = gap_start
            lens = [len(ref[i]["norm"]) + 1 for i in range(a + 1, b)]; total = sum(lens); pos = gap_start
            for i, L in zip(range(a + 1, b), lens):
                ref[i]["start"] = pos; pos += (gap_end - gap_start) * L / total; ref[i]["end"] = pos
    # after last anchor
    last = idx[-1]
    if last < len(ref) - 1:
        n = len(ref) - 1 - last; start = ref[last]["end"]; end = min(duration, start + n / WORDS_PER_SEC)
        for k, i in enumerate(range(last + 1, len(ref))):
            ref[i]["start"] = start + (end - start) * k / n; ref[i]["end"] = start + (end - start) * (k + 1) / n


def chunk(ref: list[dict], max_seconds: float, min_seconds: float, pad: float, duration: float) -> list[dict]:
    def cut_score(i: int) -> float:  # how good is a cut AFTER token i
        t = ref[i]; s = 0.0
        if t["para_end"]: s += 3
        if t["sent_end"]: s += 2
        if t["comma"]: s += 1
        if t["anchored"] and i + 1 < len(ref) and ref[i + 1]["anchored"]:
            gap = ref[i + 1]["start"] - t["end"]
            if gap >= 0.3: s += 1 + min(gap, 1.5)
        elif t["anchored"]: s += 0.5
        return s

    chunks, start_i, prev_end = [], 0, 0.0
    while start_i < len(ref):
        best_i, best_s, i = None, -1.0, start_i
        while i < len(ref) and ref[i]["end"] - ref[start_i]["start"] <= max_seconds:
            s = cut_score(i)
            if s >= best_s: best_i, best_s = i, s  # ties -> later cut (longer chunk)
            i += 1
        end_i = best_i if best_i is not None else start_i  # at least one token
        if i >= len(ref) and (best_i is None or ref[-1]["end"] - ref[start_i]["start"] <= max_seconds):
            end_i = len(ref) - 1
        toks = ref[start_i:end_i + 1]
        start = max(prev_end, toks[0]["start"] - pad); end = min(duration, toks[-1]["end"] + pad)
        text = "".join(t["sep"] + t["surface"] for t in toks).strip()
        anchored = sum(t["anchored"] for t in toks)
        chunks.append({"start": round(start, 3), "end": round(end, 3), "yiddish": text, "n_words": len(toks),
                       "anchor_rate": round(anchored / len(toks), 3),
                       "start_anchored": bool(toks[0]["anchored"] or (start_i > 0 and ref[start_i - 1]["anchored"])),
                       "end_anchored": bool(toks[-1]["anchored"] or (end_i + 1 < len(ref) and ref[end_i + 1]["anchored"]))})
        prev_end = end; start_i = end_i + 1
    return [c for c in chunks if c["end"] - c["start"] >= min_seconds]


def compare_with_site(ref: list[dict], site_segments: list[dict], offset: float = 0.0) -> dict:
    rn = [t["norm"] for t in ref]; pointer = 0; d_start, d_end = [], []
    for seg in site_segments:
        words = scoring_text(seg.get("yiddish") or "").split()
        if not words: continue
        found = None
        for i in range(pointer, min(len(rn) - len(words) + 1, pointer + 400)):
            if rn[i:i + len(words)] == words: found = i; break
        if found is None: continue
        d_start.append(ref[found]["start"] - float(seg["start"]) - offset); d_end.append(ref[found + len(words) - 1]["end"] - float(seg["end"]) - offset)
        pointer = found + len(words)
    if not d_start:
        return {"compared": 0}
    ds, de = np.abs(d_start), np.abs(d_end)
    return {"compared": len(d_start), "of": len(site_segments), "median_abs_start_diff": round(float(np.median(ds)), 2),
            "p90_abs_start_diff": round(float(np.percentile(ds, 90)), 2), "median_abs_end_diff": round(float(np.median(de)), 2),
            "within_1s": round(float(np.mean((ds <= 1.0) & (de <= 1.0))), 3)}


def process(day_json: Path, args) -> dict:
    rec = json.loads(day_json.read_text())
    paras = hanacha_paragraphs(rec["api"]["sicha"].get("contentHtml") or "")
    ref = ref_tokens(paras, args.keep_brackets)
    audio = decode_16k_mono(ROOT / rec["local"]["path"]); duration = len(audio) / SR
    t0 = time.time()
    hyp = transcribe_words(audio, args.model, args.device, args.compute_type, args.language)
    anchors = align(ref, hyp, args.fuzzy_min)
    intro_end = float(((rec.get("sync") or {}).get("diagnostics") or {}).get("hiddenAlignmentIntroEnd") or args.intro_seconds) \
        + float(rec.get("local_offset_seconds") or 0.0)
    assign_times(ref, hyp, anchors, duration, intro_end)
    chunks = chunk(ref, args.max_seconds, args.min_seconds, args.pad, duration)
    hyp_wer = jiwer.wer(" ".join(t["norm"] for t in ref), " ".join(w["norm"] for w in hyp))
    diag = {"model": args.model, "n_ref_words": len(ref), "n_hyp_words": len(hyp), "anchor_rate": round(len(anchors) / max(1, len(ref)), 3),
            "hyp_wer": round(hyp_wer, 3), "audio_seconds": round(duration, 1), "chunks": len(chunks),
            "mean_chunk_anchor_rate": round(float(np.mean([c["anchor_rate"] for c in chunks])), 3) if chunks else 0,
            "seconds": round(time.time() - t0, 1)}
    if (rec.get("sync") or {}).get("segments"):
        diag["vs_site"] = compare_with_site(ref, rec["sync"]["segments"], float(rec.get("local_offset_seconds") or 0.0))
        diag["vs_site"]["offset_applied"] = rec.get("local_offset_seconds")
    rec["sync_local"] = {"kind": "local-anchor-alignment", "segments": chunks, "diagnostics": diag}
    day_json.write_text(json.dumps(rec, ensure_ascii=False))
    return diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="*", default=[])
    ap.add_argument("--year", nargs="*", type=int, default=[])
    ap.add_argument("--month-dir", nargs="*", default=[])
    ap.add_argument("--all-missing", action="store_true", help="every cached day without usable site timings (no sync, or offset refused)")
    ap.add_argument("--site-timed", action="store_true", help="every cached day WITH site timings (to cross-check them; use with --force)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--model", default="ivrit-ai/yi-whisper-large-v3-turbo-ct2")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--language", default="yi")
    ap.add_argument("--fuzzy-min", type=float, default=0.75)
    ap.add_argument("--max-seconds", type=float, default=28.0); ap.add_argument("--min-seconds", type=float, default=1.0)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--intro-seconds", type=float, default=7.0)
    ap.add_argument("--keep-brackets", action="store_true", default=True)
    ap.add_argument("--drop-brackets", dest="keep_brackets", action="store_false", help="treat [..] as unspoken insertions")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th selected day starting at i (run n processes in parallel)")
    args = ap.parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    days = sorted((DATA / "days").rglob("*.json"))
    sel = []
    for d in days:
        rec = json.loads(d.read_text())
        if args.days and d.stem not in args.days: continue
        if args.year and rec["local"]["year"] not in args.year: continue
        if args.month_dir and rec["local"]["month_dir"] not in args.month_dir: continue
        if args.all_missing and (rec.get("sync") or {}).get("segments") and rec.get("local_offset_seconds") is not None: continue  # keep refused-offset days
        if args.site_timed and not (rec.get("sync") or {}).get("segments"): continue
        if not args.force and rec.get("sync_local") and not args.days: continue
        sel.append(d)
    sel = sel[shard_i::shard_n]
    print(f"aligning {len(sel)} days with {args.model} (shard {args.shard})")
    for d in sel:
        try:
            diag = process(d, args)
            print(f"  {d.stem:<28} anchors {diag['anchor_rate']:.2f}  hypWER {diag['hyp_wer']:.2f}  chunks {diag['chunks']:3d}  "
                  f"chunk-anchor {diag['mean_chunk_anchor_rate']:.2f}  {diag['seconds']:.0f}s" + (f"  vs site: {diag['vs_site']}" if 'vs_site' in diag else ""), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {d.stem}: FAILED {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
