#!/usr/bin/env python3
"""Populate / refresh the public GitHub repository tree (asr/public-repo) from this project, allowlist only.

Published: pipeline code, fleet scripts and their docs, requirements, the corpus data sheet (as the GitHub Pages site,
with the one verbatim hanacha excerpt withheld), its charts and aggregate tables, and the aggregate reports.
Never published: audio, clips, hanacha text, day records, manifests, transcripts, evaluation reports with
reference/hypothesis text, the internal handoff, hosts/IP lists, keys.

  python publish_repo.py                 # refresh the tree, refresh the README status block, show git status
  python publish_repo.py --push          # also commit and push to origin/main
  python publish_repo.py --status "..."  # set the one-line status shown in the README
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ASR = Path(__file__).resolve().parent
REPO = ASR / "public-repo"
REPO_NAME = "rebbe-yiddish-asr"
PAGES_URL = f"https://golanbenoni.github.io/{REPO_NAME}/"

# names of people and internal wording that stay out of the public copy
PUBLIC_REPLACEMENTS = [
    ("The Daily Sicha (Asher Schochet)", "The Daily Sicha"), ("Asher Schochet", "The Daily Sicha team"), ("Asher", "The Daily Sicha team"),
    ("Rabbi Notik", "the hanachos' author"), ("R. Notik", "the hanachos' author"),
    ("Golan's go-ahead", "the project owner's go-ahead"), ("Golan", "the project owner"),
]


def public_text(t: str) -> str:
    for a, b in PUBLIC_REPLACEMENTS:
        t = t.replace(a, b)
    return t


def copy_tree() -> None:
    (REPO / "asr" / "pipeline").mkdir(parents=True, exist_ok=True)
    (REPO / "asr" / "cluster").mkdir(parents=True, exist_ok=True)
    for p in sorted((ASR / "pipeline").glob("*.py")):
        shutil.copy2(p, REPO / "asr" / "pipeline" / p.name)
    for p in sorted(ASR.glob("requirements*.txt")):
        shutil.copy2(p, REPO / "asr" / p.name)
    shutil.copy2(ASR / "publish_repo.py", REPO / "asr" / "publish_repo.py")
    for p in sorted((ASR / "cluster").iterdir()):
        if p.name in ("tailscale_ips.txt",) or (p.name.startswith("hosts") and p.suffix == ".txt"):
            continue                                   # internal host lists and tailnet addresses stay private
        if p.suffix in (".sh", ".example", ".md") or p.name.startswith("train_configs"):
            shutil.copy2(p, REPO / "asr" / "cluster" / p.name)
    # data sheet: the export bundle is the stable source (reports/corpus-datasheet-export.zip)
    docs = REPO / "docs"
    for sub in ("charts-svg", "charts-png", "data"):
        shutil.rmtree(docs / sub, ignore_errors=True)
    docs.mkdir(exist_ok=True)
    with zipfile.ZipFile(ASR / "reports" / "corpus-datasheet-export.zip") as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            target = docs / ("index.html" if info.filename == "corpus-datasheet.html" else ("export-README.txt" if info.filename == "README.txt" else info.filename))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))
    html = (docs / "index.html").read_text(encoding="utf-8")
    html, n = re.subn(r'<p class="yi">.*?</p>', '<p class="yi muted">[the example text is withheld on the public page: the hanachos belong to The Daily Sicha and are not republished here]</p>', html, count=1, flags=re.S)
    assert n == 1, "sample paragraph not found"
    html = public_text(html)
    (docs / "index.html").write_text(html, encoding="utf-8")
    stats_p = docs / "data" / "dataset-stats.json"
    if stats_p.exists():                                 # aggregate statistics only: drop any per-clip samples
        stats = json.loads(stats_p.read_text(encoding="utf-8"))
        for k in ("dropped_sample", "dev_detail_sample", "test_detail_sample"):
            stats.pop(k, None)
        stats_p.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy2(ASR / "reports" / "corpus-datasheet.pdf", docs / "corpus-datasheet.pdf")
    (docs / ".nojekyll").write_text("")
    rep = docs / "reports"; rep.mkdir(exist_ok=True)
    for src, dst in (("reports/final-comparison.md", "model-comparison.md"), ("reports/drafts-vs-pdf/by-year.md", "drafts-vs-hanachos-by-year.md"), ("reports/windowing/summary.txt", "windowing-experiment.txt")):
        if (ASR / src).exists():
            (rep / dst).write_text(public_text((ASR / src).read_text(encoding="utf-8")), encoding="utf-8")


def refresh_readme(status: str | None) -> None:
    readme = REPO / "README.md"
    t = readme.read_text(encoding="utf-8")
    comp = json.loads((ASR / "reports" / "final-comparison.json").read_text())
    rows = [r for r in comp if r["model"].startswith("hf-") and "lf_test_wer_reported" in r]
    rows.sort(key=lambda r: r["lf_test_wer_reported"])
    f = lambda r, k: f"{r[k]:.3f}" if k in r else "–"
    table = ["| model (GPU decoding path) | clip test WER | clip test CER | whole-file test WER | whole-file test CER |", "|---|---:|---:|---:|---:|"]
    for r in rows[:10]:
        table.append(f"| {r['model'][3:]} | {f(r,'clip_test_wer')} | {f(r,'clip_test_cer')} | {f(r,'lf_test_wer_reported')} | {f(r,'lf_test_cer_reported')} |")
    by_year = ""
    p = ASR / "reports" / "drafts-vs-pdf" / "by-year.md"
    if p.exists():
        by_year = "\n".join(l for l in p.read_text(encoding="utf-8").splitlines() if l.startswith("|"))
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M %Z").strip()
    if status is None:
        m = re.search(r"\*\*Status:\*\* (.*)", t)
        status = m.group(1) if m else "see the data sheet"
    block = f"""<!-- status:start -->
_Last refreshed {stamp} by `asr/publish_repo.py`._

**Status:** {status}

**Best models on the fixed 20-day test set** (488 clips / 20 whole recordings, scored against the published hanachos; full table in [docs/reports/model-comparison.md](docs/reports/model-comparison.md)):

{chr(10).join(table)}

**Delivered transcripts of the product years against their hanachos** (whole-file WER over all days of the year, including days whose printed text is a different or heavily edited sicha):

{by_year}
<!-- status:end -->"""
    t2, n = re.subn(r"<!-- status:start -->.*?<!-- status:end -->", block, t, count=1, flags=re.S)
    assert n == 1
    readme.write_text(t2, encoding="utf-8")


def git(*args: str, check: bool = True) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=check).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--status", default=None)
    ap.add_argument("--message", default=None)
    args = ap.parse_args()
    copy_tree()
    refresh_readme(args.status)
    if not (REPO / ".git").exists():
        git("init", "-b", "main")
        git("config", "user.name", "Golan Ben-Oni"); git("config", "user.email", "golanbenoni@users.noreply.github.com")
    print(git("status", "--short") or "  (tree unchanged)")
    if args.push:
        git("add", "-A")
        if git("status", "--short").strip():
            msg = args.message or f"Refresh code, data sheet and reports ({dt.date.today()})"
            git("commit", "-q", "-m", msg + "\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>")
        out = subprocess.run(["git", "push", "-u", "origin", "main"], cwd=REPO, capture_output=True, text=True)
        print(out.stdout + out.stderr)
        return out.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
