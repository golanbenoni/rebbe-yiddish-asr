#!/usr/bin/env python3
"""Produce text for recordings that have none: long-form transcription of mp3/wav files.

Uses faster-whisper with voice-activity chunking (Silero VAD) so 10-minute or
multi-hour farbrengen recordings decode in ~30 s windows without drifting, and
writes three outputs per input: plain text, SRT captions, and a JSON with
segment timestamps and per-segment confidence (avg logprob, no-speech prob).
Point --model at a baseline CT2 checkpoint or at a fine-tuned run converted with
ct2-transformers-converter.

  python transcribe_file.py --model ivrit-ai/yi-whisper-large-v3-ct2 --out reports/new-text \
      "../DS 5778/01 Tishrei/001 4-Tishrei.mp3"
  python transcribe_file.py --model runs/yi-large-v3-ft-ct2 --device cuda --compute-type float16 --out out ../DS\ 5778
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio import SR, decode_16k_mono as decode  # noqa: E402
from normalize import training_text  # noqa: E402
from decoding import add_suppress_arg, suppress_tokens  # noqa: E402





def srt_time(t: float) -> str:
    ms = int(round(t * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    add_suppress_arg(ap)
    ap.add_argument("inputs", nargs="*", help="mp3/wav/flac files or directories")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="yi")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--no-vad", action="store_true")
    ap.add_argument("--skip-seconds", type=float, default=0.0, help="skip the announcer intro, e.g. 7")
    ap.add_argument("--limit-seconds", type=float, default=0.0, help="transcribe only the first N seconds (testing)")
    ap.add_argument("--initial-prompt", default=None)
    ap.add_argument("--list", default=None, help="text file with one input path per line (in addition to inputs)")
    ap.add_argument("--mirror-root", default=None, help="write outputs under --out mirroring each input's path relative to this root")
    ap.add_argument("--backend", default="ct2", choices=["ct2", "hf"], help="ct2 = faster-whisper int8 on CPU; hf = HF fp32 model on MPS/CUDA (pipeline/hf_longform.py)")
    ap.add_argument("--batch", type=int, default=8, help="hf backend: windows per batch")
    ap.add_argument("--max-window", type=float, default=28.0, help="hf backend: max seconds of speech per decoded window")
    ap.add_argument("--windows", default="auto", choices=["vad", "energy", "auto", "fixed"], help="hf backend: window strategy (hf_longform.speech_windows_flagged); auto = VAD windows + confidence-gated loud gaps, production default since 2026-09-23")
    args = ap.parse_args()

    files = []
    for inp in args.inputs:
        p = Path(inp)
        files += sorted(p.rglob("*.mp3")) + sorted(p.rglob("*.wav")) if p.is_dir() else [p]
    if args.list:
        files += [Path(l.strip()) for l in Path(args.list).read_text().splitlines() if l.strip()]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.backend == "hf":
        from hf_longform import HFTranscriber
        model = HFTranscriber(args.model, language=args.language, beam=args.beam, batch=args.batch)
        sup = None; print(f"hf backend on {model.device} ({model.dtype})", flush=True)
    else:
        from faster_whisper import WhisperModel
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
        sup = suppress_tokens(model, args.suppress, args.language)
    for f in files:
        t0 = time.time()
        audio = decode(f)
        if args.skip_seconds:
            audio = audio[int(args.skip_seconds * SR):]
        if args.limit_seconds:
            audio = audio[: int(args.limit_seconds * SR)]
        if args.backend == "hf":
            segs = model.transcribe(audio, max_seconds=args.max_window, min_silence_ms=400, mode=args.windows)
        else:
            segs, info = model.transcribe(
                audio, language=args.language, task="transcribe", beam_size=args.beam,
                vad_filter=not args.no_vad, vad_parameters={"min_silence_duration_ms": 400},
                condition_on_previous_text=False, suppress_tokens=sup, initial_prompt=args.initial_prompt)
        rows = []
        for s in segs:
            rows.append({"start": round(s.start + args.skip_seconds, 2), "end": round(s.end + args.skip_seconds, 2),
                         "text": training_text(s.text), "avg_logprob": round(s.avg_logprob, 3),
                         "no_speech_prob": round(s.no_speech_prob, 3)})
        stem = f.stem
        if args.mirror_root:
            out = Path(args.out) / f.resolve().parent.relative_to(Path(args.mirror_root).resolve()); out.mkdir(parents=True, exist_ok=True)
        (out / f"{stem}.txt").write_text("\n".join(r["text"] for r in rows) + "\n")
        (out / f"{stem}.srt").write_text("".join(
            f"{i}\n{srt_time(r['start'])} --> {srt_time(r['end'])}\n{r['text']}\n\n" for i, r in enumerate(rows, 1)))
        (out / f"{stem}.json").write_text(json.dumps({
            "source": str(f), "model": args.model, "backend": args.backend, "windows": args.windows if args.backend == "hf" else None, "language": args.language,
            "audio_seconds": round(len(audio) / SR, 1), "segments": rows}, ensure_ascii=False, indent=1))
        low = sum(1 for r in rows if r["avg_logprob"] < -1.0)
        print(f"{f.name}: {len(rows)} segments, {sum(len(r['text'].split()) for r in rows)} words, "
              f"{low} low-confidence, {len(audio)/SR/(time.time()-t0):.2f}x realtime -> {out/stem}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
