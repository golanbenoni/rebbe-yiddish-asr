#!/usr/bin/env python3
"""Where does the model still err? Merge per-clip evaluation reports (shards included) and break the errors down.

  python pipeline/error_analysis.py --reports 'reports/v1/v1-turbo-full-lr1e5-3ep-shard*/report.json' --name turbo-full-lr1e5-3ep

Writes reports/error-analysis-<name>.md: WER by split, by word-frequency class (how often the word occurs in the
training transcripts; 0 = out of vocabulary), by spelling class (consonantal spelling = loshon-kodesh-like words vs
Yiddish words, abbreviations written with gershayim in the reference), the most frequent substitution pairs,
deletions and insertions, per-day WER and the worst clips. Everything is computed on the shared scoring
normalization (Hebrew letters only), the same one the WER numbers use.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import scoring_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VOWELS = set("אעויײױיִ")


def spelling_class(tok: str) -> str:
    if len(tok) < 3:
        return "short (<3 letters)"
    v = sum(c in VOWELS for c in tok) / len(tok)
    return "consonantal (loshon-kodesh-like)" if v < 0.25 else "yiddish-like"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True, help="glob of report.json files to merge")
    ap.add_argument("--name", required=True)
    ap.add_argument("--train-manifest", default=str(ROOT / "asr" / "data" / "manifests" / "train.clean.jsonl"))
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    clips = []
    for f in sorted(glob.glob(args.reports)):
        d = json.load(open(f))
        clips += d.get("clips", [])
    if not clips:
        print("no clips found"); return 1
    train_counts = Counter()
    for line in open(args.train_manifest):
        if line.strip():
            train_counts.update(scoring_text(json.loads(line)["text"]).split())

    def freq_class(tok: str) -> str:
        c = train_counts.get(tok, 0)
        return "0 (OOV)" if c == 0 else "1-9" if c < 10 else "10-99" if c < 100 else "100+"

    subs, dels, ins = Counter(), Counter(), Counter()
    by_split = defaultdict(lambda: [0, 0]); by_freq = defaultdict(lambda: [0, 0]); by_spell = defaultdict(lambda: [0, 0])
    abbrev = [0, 0]; by_day = defaultdict(lambda: [0, 0, 0])
    oov_tokens = Counter()
    for c in clips:
        ref_raw_toks = c["ref"].split()
        quoted = {scoring_text(t) for t in ref_raw_toks if '"' in t or "״" in t}
        ref, hyp = scoring_text(c["ref"]), scoring_text(c["hyp"])
        if not ref.split():
            continue
        out = jiwer.process_words(ref, hyp)
        R, H = out.references[0], out.hypotheses[0]
        err_flags = [False] * len(R)
        for ch in out.alignments[0]:
            if ch.type == "substitute":
                for i, j in zip(range(ch.ref_start_idx, ch.ref_end_idx), range(ch.hyp_start_idx, ch.hyp_end_idx)):
                    subs[(R[i], H[j])] += 1; err_flags[i] = True
            elif ch.type == "delete":
                for i in range(ch.ref_start_idx, ch.ref_end_idx):
                    dels[R[i]] += 1; err_flags[i] = True
            elif ch.type == "insert":
                for j in range(ch.hyp_start_idx, ch.hyp_end_idx):
                    ins[H[j]] += 1
        n_ins = out.insertions
        for i, tok in enumerate(R):
            e = int(err_flags[i])
            by_split[c["split"]][0] += 1; by_split[c["split"]][1] += e
            by_freq[freq_class(tok)][0] += 1; by_freq[freq_class(tok)][1] += e
            by_spell[spelling_class(tok)][0] += 1; by_spell[spelling_class(tok)][1] += e
            if tok in quoted:
                abbrev[0] += 1; abbrev[1] += e
            if train_counts.get(tok, 0) == 0:
                oov_tokens[tok] += 1
        by_split[c["split"]][1] += n_ins  # insertions count against the split's WER like jiwer does
        d = by_day[(c["split"], c["day"])]; d[0] += len(R); d[1] += out.substitutions + out.deletions + n_ins; d[2] += 1

    def rate(x): return f"{x[1] / max(1, x[0]):.3f}"
    total = [sum(v[0] for v in by_split.values()), sum(v[1] for v in by_split.values())]
    md = [f"# Error analysis: {args.name}", "", f"reports: `{args.reports}`  clips: {len(clips)}  ref words: {total[0]}  WER {rate(total)}", "",
          "## By split", "", "| split | ref words | WER |", "|---|---:|---:|"]
    md += [f"| {k} | {v[0]} | {rate(v)} |" for k, v in sorted(by_split.items())]
    md += ["", "## By word frequency in the training transcripts (substitution+deletion rate of reference words)", "",
           "| train count | ref words | share | error rate |", "|---|---:|---:|---:|"]
    for k in ["100+", "10-99", "1-9", "0 (OOV)"]:
        v = by_freq[k]; md.append(f"| {k} | {v[0]} | {v[0] / max(1, total[0]):.1%} | {rate(v)} |")
    md += ["", "## By spelling class", "", "| class | ref words | error rate |", "|---|---:|---:|"]
    md += [f"| {k} | {v[0]} | {rate(v)} |" for k, v in sorted(by_spell.items())]
    md += [f"| abbreviations (gershayim in the reference) | {abbrev[0]} | {rate(abbrev)} |", ""]
    md += [f"## Top {args.top} substitutions (reference -> hypothesis)", "", "| count | reference | hypothesis | train count of ref |", "|---:|---|---|---:|"]
    md += [f"| {n} | {r} | {h} | {train_counts.get(r, 0)} |" for (r, h), n in subs.most_common(args.top)]
    md += ["", f"## Top {args.top // 2} deletions", "", "| count | reference word |", "|---:|---|"]
    md += [f"| {n} | {w} |" for w, n in dels.most_common(args.top // 2)]
    md += ["", f"## Top {args.top // 2} insertions", "", "| count | hypothesis word |", "|---:|---|"]
    md += [f"| {n} | {w} |" for w, n in ins.most_common(args.top // 2)]
    md += ["", f"## Most frequent out-of-vocabulary reference words", "", "| count | word |", "|---:|---|"]
    md += [f"| {n} | {w} |" for w, n in oov_tokens.most_common(25)]
    md += ["", "## Per day", "", "| split | day | clips | ref words | WER |", "|---|---|---:|---:|---:|"]
    md += [f"| {s} | {d} | {v[2]} | {v[0]} | {v[1] / max(1, v[0]):.3f} |" for (s, d), v in sorted(by_day.items())]
    worst = sorted(clips, key=lambda c: -c["wer"])[:8]
    md += ["", "## Worst clips", ""]
    for c in worst:
        md += [f"### {c['id']}  WER {c['wer']:.2f}", f"- ref: {c['ref']}", f"- hyp: {c['hyp']}", ""]
    out = ROOT / "asr" / "reports" / f"error-analysis-{args.name}.md"
    out.write_text("\n".join(md))
    print(f"wrote {out}"); print("\n".join(md[:30]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
