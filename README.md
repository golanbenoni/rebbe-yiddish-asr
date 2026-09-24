# Rebbe Yiddish ASR

Speech recognition for the Rebbe's recorded farbrengens: fine-tuning Whisper on The Daily Sicha archive (the Rebbe's own
Yiddish audio paired with the published Yiddish *hanachos*), using the fine-tuned model to produce draft transcripts for
recordings that have no text, and measuring every step against human text.

**Data sheet with every chart, table and number: <https://golanbenoni.github.io/rebbe-yiddish-asr/>** (also as
[PDF](docs/corpus-datasheet.pdf)). This repository holds the code and the aggregate documentation only. The
recordings, the hanachos, the clips and the machine transcripts belong to The Daily Sicha and are not published here;
nothing produced by this project is made public or searchable before its owners have reviewed it.

<!-- status:start -->
_Last refreshed 2026-09-24 05:56 by `asr/publish_repo.py`._

**Status:** v4 complete; v5 data (5775-5781) being aligned on the fleet, dataset build and three v5 training runs follow automatically.

**Best models on the fixed 20-day test set** (488 clips / 20 whole recordings, scored against the published hanachos; full table in [docs/reports/model-comparison.md](docs/reports/model-comparison.md)):

| model (GPU decoding path) | clip test WER | clip test CER | whole-file test WER | whole-file test CER |
|---|---:|---:|---:|---:|
| turbo-full-lr3e5-4ep-seed7 | 0.087 | 0.050 | 0.080 | 0.045 |
| turbo-full-v4-lr3e5self-lr1e5-2ep | 0.087 | 0.050 | 0.081 | 0.045 |
| turbo-full-lr3e5-4ep | 0.087 | 0.050 | 0.081 | 0.045 |
| turbo-full-lr3e5-4ep-winauto | – | – | 0.081 | 0.045 |
| turbo-full-lr5e5-4ep | 0.089 | 0.053 | 0.082 | 0.046 |
| large-full-lr2e5-3ep | 0.083 | 0.048 | 0.082 | 0.047 |
| turbo-full-pseudo-self-lr1e5-2ep | 0.090 | 0.051 | 0.082 | 0.046 |
| turbo-full-pseudo-cont-lr1e5-2ep | 0.090 | 0.052 | 0.084 | 0.046 |
| large-full-v4-cont-lr1e5-2ep | 0.089 | 0.050 | 0.085 | 0.047 |
| turbo-full-pseudo-lr2e5-2ep | 0.092 | 0.053 | 0.086 | 0.047 |

**Delivered transcripts of the product years against their hanachos** (whole-file WER over all days of the year, including days whose printed text is a different or heavily edited sicha):

| year | days scored | ref words | WER (all words) | CER (all chars) | median day WER | days > 0.20 | no draft |
|---|---|---|---|---|---|---|---|
| 5778 | 247 | 319,404 | 0.0716 | 0.0407 | 0.059 | 10 | 0 |
| 5779 | 266 | 341,973 | 0.0698 | 0.0404 | 0.057 | 5 | 0 |
| 5780 | 245 | 306,533 | 0.0728 | 0.0419 | 0.057 | 10 | 0 |
| 5781 | 247 | 312,740 | 0.0898 | 0.0497 | 0.071 | 9 | 0 |
<!-- status:end -->

## The problem

The Rebbe's farbrengens (public talks, 1950-1992) exist as tens of thousands of hours of audio in Yiddish with heavy
loshon-kodesh (Hebrew/Aramaic quotation and terminology). Written hanachos exist for a part of them. Off-the-shelf
speech recognition, including Whisper models adapted to Yiddish, transcribes this speaker and this register badly:
the untouched Yiddish Whisper large-v3 scores a word error rate of about 0.57 on whole recordings. The goal is a model
that turns the remaining recordings into usable draft text, and honest numbers for how good that text is.

## What the data is

The Daily Sicha publishes one roughly ten-minute excerpt of a farbrengen per day with a Yiddish hanacha, and for
recent years a sync file with segment timings. Two deliveries of source material are in use:

| material | days | audio | text |
|---|---:|---:|---|
| site years 5783-5787 (2022-2027): daily mp3s + the site's hanachos and, for one year, its timings | 1,468 | 253 h | 1.86 M words |
| hanachos PDFs for 5775-5781 (2014-2021) + the matching recordings (received 2026-09-23) | 1,758 | 306 h | 2.31 M words |

From the first delivery the pipeline cut **30,212 clips (168 h)**; after filtering, **27,833 clips (153 h)** train every
model reported below. The second delivery is being aligned and will roughly double the training set (v5).

One training example is a clip of at most 28 s of the Rebbe's speech paired with the words the hanacha gives for that
stretch: Yiddish in unpointed Hebrew script, loshon-kodesh embedded verbatim, sources abbreviated with gershayim
(רש"י, הקב"ה), punctuation kept, vowel points removed. Splits are by day: 20 fixed dev days, 20 fixed test days (site-timed,
farbrengens from 5739 on), never trained on and never changed; from v5 on a second held-out set of 20 days of 5781
(clean printed text) is reported next to it.

## How it works

Everything runs from `asr/` with `python -u`; long jobs run under `nohup` with logs in `asr/reports/`.

| step | script | what it does |
|---|---|---|
| inventory + fetch | `pipeline/fetch_days.py` | resolves each mp3 to its Hebrew date, fetches the day's hanacha and sync JSON (one request at a time) |
| offsets | `pipeline/compute_offsets.py`, `calibrate_site_timings.py` | many daily files carry a dedication before the excerpt; cross-correlation finds where the excerpt starts, so site timings land on the right frame |
| hanachos from PDFs | `pipeline/pdf_fontmap.py`, `pdf_hanachos.py`, `build_pdf_days.py` | the older PDFs use custom-encoded fonts (each glyph code is a fixed permutation of the alphabet, per font); the maps are learned by hill-climbing the share of decoded words found in the training vocabulary, the bold header font is pinned by the words every header shares, days are split by header and matched to the recordings by their printed number |
| alignment | `pipeline/align_day.py` (+ `cluster/align_fleet.sh`) | where no timings exist, a Yiddish Whisper model transcribes with word times and the hypothesis is anchored to the reference (exact, then fuzzy); anchor rates are 0.94-0.98 |
| dataset | `pipeline/build_dataset.py`, `filter_manifest.py` | merges timed segments into clips of at most 28 s, writes FLAC clips and JSONL manifests split by day; a scan with a strong model drops clips whose transcription disagrees wildly with the text (misaligned timings) |
| training | `pipeline/finetune.py` (+ `cluster/train_fleet.sh`) | Hugging Face `Seq2SeqTrainer`, fp32 on Apple-silicon GPUs, one run per Mac Studio; full fine-tuning, LoRA or frozen-encoder recipes; `--resume`, `--init-from`, `--seed` |
| evaluation | `pipeline/evaluate.py`, `evaluate_longform.py`, `compare_runs.py`, `error_analysis.py` | clip WER/CER on the fixed dev/test days and whole-file WER on the 20 test recordings; every table is recomputed from stored reference/hypothesis pairs |
| production decoding | `pipeline/transcribe_file.py`, `hf_longform.py`, `decoding.py` | the HF fp32 model on the GPU with speech windows of at most 28 s, no timestamp tokens, a repetition guard, and confidence-gated recovery of speech the voice-activity detector drops; writes `.txt`, `.srt` and `.json` per recording |
| audit | `pipeline/audit_transcripts.py`, `score_drafts_vs_pdf.py` | words per second, window coverage and word counts against the printed text; whole-file WER per year of delivered transcripts |
| fleet | `cluster/*.sh` | push, align, build + scan, train, harvest, transcribe and monitor on 5 Mac Studios with 256 GB and 3 with 512 GB of unified memory, linked by Thunderbolt/RDMA |

Two decoder details matter more than any recipe: Whisper's default token-suppression list blocks the ASCII double quote,
which in Hebrew text is the gershayim of every abbreviation (`decoding.py --suppress allow-quotes` frees it), and
faster-whisper's long-form path loses about three WER points against the HF fp32 path, so every number here comes from
the HF path.

## What was learned

- Full fine-tuning of the turbo model at learning rate 3e-5 for 4 epochs is the best recipe found on the first dataset:
  whole-file test WER **0.081** (character error rate 0.045) from 0.57 for the untouched Yiddish Whisper. large-v3 at
  2e-5 reaches 0.082 whole-file and is the best clip model (0.083), at about 2.5x the training and decoding cost.
- Higher learning rates (4e-5, 5e-5) are worse; freezing the encoder plateaus near 0.20; LoRA lands near 0.084.
- Self-training on the model's own transcripts of the text-less years lifts weaker starting points by 0.5-1 point and
  never beats the best supervised recipe (production 0.081 -> 0.081; large 0.082 -> 0.085).
- A seed repeat of the production recipe scores 0.080: run-to-run noise is about 0.001, so differences under 0.003
  in any table mean nothing. The 200-clip dev subset evaluated during training does not rank models; only the full
  fixed test set through the GPU path does.
- Half of the remaining word errors are words never seen in training (names, sources, loshon-kodesh phrases);
  frequent words are right about 95% of the time.
- The voice-activity detector silently drops parts of loud, clipped recordings (13 of 1,005 files lost 11-52% of their
  audio, one lost 99%). Word-count audits against the printed text caught it; the decoder now recovers such gaps when
  their decode is confident, with no change on normal recordings.
- The delivered transcripts of the product years score 0.070-0.073 WER against their hanachos (5778-5780) and 0.090
  for 5781 (0.080 without two days whose printed text is a different sicha).

## Running it

```bash
cd asr
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # requirements-mac.txt for the MPS training nodes, requirements-align.txt for alignment-only nodes
python -u pipeline/compute_offsets.py      # before every dataset build
python -u pipeline/build_dataset.py --year 5783 5784 5785 5786 5787 --splits-file data/splits.json
python -u pipeline/evaluate.py --model <ct2-model> --name train-scan --splits train
python -u pipeline/filter_manifest.py --manifest data/manifests/train.jsonl --report reports/train-scan/report.json --out data/manifests/train.clean.jsonl
python -u pipeline/finetune.py --base-model ivrit-ai/yi-whisper-large-v3-turbo --epochs 4 --lr 3e-5 --batch 8 --grad-accum 4 --gradient-checkpointing --train-manifest train.clean.jsonl --output runs/turbo-full-lr3e5-4ep
python -u pipeline/evaluate_longform.py --backend hf --model runs/turbo-full-lr3e5-4ep --name hf-turbo-full-lr3e5-4ep --splits-file data/splits.json
python -u pipeline/transcribe_file.py --backend hf --model runs/turbo-full-lr3e5-4ep --out output --skip-seconds 7 <mp3 or folder>
```

The data files these commands expect (`data/days/`, `data/manifests/`, `data/segments/`, the recordings) are not in this
repository; `asr/cluster/README.md` describes the fleet setup and the launch scripts.

## Repository layout

```
README.md                    this file (the status block is refreshed by asr/publish_repo.py)
docs/                        GitHub Pages: the data sheet (index.html, PDF), charts as SVG/PNG, aggregate CSV/JSON tables, reports
asr/pipeline/                the pipeline (Python 3.11; PyTorch, transformers, faster-whisper/CTranslate2, PyMuPDF, jiwer)
asr/cluster/                 fleet scripts for the Mac Studios (push, align, build+scan, train, harvest, transcribe, status)
asr/requirements*.txt        dependency sets
asr/publish_repo.py          the allowlist script that populates this repository from the working project
```

## Acknowledgements and terms

Audio and text: The Daily Sicha, used under the project's agreement with its producers; base models:
[ivrit.ai](https://huggingface.co/ivrit-ai) Yiddish-adapted Whisper large-v3 and large-v3-turbo. No license has been
granted for the code yet (all rights reserved until one is chosen). Contact through the repository owner.
