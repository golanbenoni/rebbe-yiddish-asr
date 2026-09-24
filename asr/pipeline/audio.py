"""Portable audio decode for the pipeline: afconvert on macOS, ffmpeg elsewhere."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def decode_16k_mono(path: Path | str) -> np.ndarray:
    """Return float32 mono audio at 16 kHz for mp3/wav/flac input."""
    path = Path(path)
    if path.suffix.lower() in (".wav", ".flac"):
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            raise ValueError(f"{path}: {sr} Hz, expected {SR}")
        return audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav = Path(tmp.name)
    try:
        if shutil.which("afconvert"):
            subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", "-c", "1", str(path), str(wav)],
                           check=True, capture_output=True)
        elif shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "wav", str(wav)],
                           check=True, capture_output=True)
        else:
            raise RuntimeError("need afconvert (macOS) or ffmpeg (Linux) to decode mp3")
        audio, _ = sf.read(wav, dtype="float32")
    finally:
        wav.unlink(missing_ok=True)
    return audio if audio.ndim == 1 else audio.mean(axis=1)


def duration_seconds(path: Path | str) -> float:
    """Fast duration via Spotlight metadata on macOS, decode fallback elsewhere."""
    if shutil.which("mdls"):
        out = subprocess.run(["mdls", "-raw", "-name", "kMDItemDurationSeconds", str(path)],
                             capture_output=True, text=True).stdout.strip()
        try:
            return float(out)
        except ValueError:
            pass
    return len(decode_16k_mono(path)) / SR
