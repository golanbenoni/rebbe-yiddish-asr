#!/usr/bin/env python3
"""Long-form transcription through the Hugging Face Whisper model on the GPU (MPS or CUDA).

faster-whisper (CTranslate2 int8 on the CPU) is the default engine of evaluate_longform.py and transcribe_file.py.
This module gives them a `--backend hf` alternative: the silero VAD (faster-whisper's copy) splits speech into
windows of at most `max_seconds`, the windows are decoded in batches with beam search by the fp32 HF model, and
each window becomes one segment whose bounds are the window's and whose confidence is the beam's mean token
log-prob. Measured 2026-09-21 on the 451 dev clips of turbo-full-lr1e5-3ep: HF fp32 on MPS WER 0.0879 at 23x
realtime vs CT2 int8 on the CPU 0.0933 at 1.5x. No token suppression is applied (the saved generation config has
an empty list), so the gershayim of abbreviations comes out right without --suppress.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

SR = 16000


def _frame_rms(audio: np.ndarray, hop: float = 0.1) -> np.ndarray:
    n = int(SR * hop); m = max(1, len(audio) // n)
    x = audio[: m * n].reshape(m, n)
    return np.sqrt((x ** 2).mean(axis=1))


def _split_long(runs, frms: np.ndarray, max_seconds: float, hop: float = 0.1, look: float = 10.0):
    """Cut every run longer than max_seconds at the quietest 100 ms frame within the last `look` seconds before the limit."""
    out = []
    for a, b in runs:
        while b - a > max_seconds:
            lo, hi = int((a + max_seconds - look) / hop), min(len(frms), int((a + max_seconds) / hop))
            k = lo + int(np.argmin(frms[lo:hi])) if hi > lo + 1 else hi
            cut = min(k * hop, b)
            if cut <= a + 1.0:
                cut = a + max_seconds
            out.append((a, cut)); a = cut
        out.append((a, b))
    return out


GATE_LOGPROB = -0.35   # confidence gate for windows added outside the VAD (mode "auto"); measured 2026-09-23 on 13 files:
GATE_WPS = 1.0         # real missed speech decodes at avg_logprob -0.02..-0.08 and 1.4-2.6 words/s, non-speech at -0.44..-0.84 and 0.3-0.9 words/s


def _loud_gap_windows(frms: np.ndarray, windows, total: float, max_seconds: float, hop: float = 0.1, ratio: float = 0.3, min_gap: float = 1.0):
    """Regions outside the VAD windows whose RMS exceeds ratio x the RMS inside them, cut to <= max_seconds."""
    mask = np.zeros(len(frms), dtype=bool)
    for a, b in windows:
        mask[int(a / hop): min(len(frms), int(np.ceil(b / hop)))] = True
    if not mask.any():
        return []
    speech_rms = float(np.sqrt(np.mean(frms[mask] ** 2)))
    out = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            i += 1; continue
        j = i
        while j < n and not mask[j]:
            j += 1
        if (j - i) * hop >= min_gap and float(np.sqrt(np.mean(frms[i:j] ** 2))) > ratio * speech_rms:
            out.extend(_split_long([(i * hop, min(total, j * hop))], frms, max_seconds, hop))
        i = j
    return out


def speech_windows_flagged(audio: np.ndarray, max_seconds: float = 28.0, min_silence_ms: int = 400, min_window: float = 0.4, mode: str = "vad") -> list[tuple[float, float, bool]]:
    """Decoding windows (start, end, added) of <= max_seconds.
    mode "vad":    silero VAD speech spans merged into windows (production since 2026-09-21); added is always False.
    mode "energy": the VAD windows unchanged PLUS every gap between them that is as loud as the speech (>= 0.3x RMS),
                   as separate windows flagged added=True (the VAD drops parts of loud recordings: 13 of 1,005 Phase 7
                   files had 48-89% coverage).
    mode "auto":   same windows as "energy"; HFTranscriber.transcribe then keeps an added window only when its decode
                   is confident (avg_logprob > GATE_LOGPROB and >= GATE_WPS words/s), so files without missed speech
                   come out exactly as with "vad".
    mode "fixed":  no VAD: ~25 s windows cut at the quietest frame.
    Fallback (2026-09-23, file 011 18-Tishrei 5781): on a loud, clipped recording the VAD kept 3 windows out of 638 s
    although the speech transcribes perfectly without VAD. If the VAD covers less than 30% of a recording longer than
    two minutes, retry at threshold 0.25, then use the fixed windows."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    total = len(audio) / SR
    hop = 0.1; frms = _frame_rms(audio, hop)
    if mode == "fixed":
        spans = _split_long([(0.0, total)], frms, max_seconds - 3.0, hop)
        return [(a, b, False) for a, b in spans if b - a >= min_window]
    spans: list[tuple[float, float]] = []
    for threshold in (0.5, 0.25):
        opts = VadOptions(threshold=threshold, min_silence_duration_ms=min_silence_ms, max_speech_duration_s=max_seconds)
        spans = [(s["start"] / SR, s["end"] / SR) for s in get_speech_timestamps(audio, opts)]
        if total < 120 or sum(b - a for a, b in spans) >= 0.3 * total:
            break
    if total >= 120 and sum(b - a for a, b in spans) < 0.3 * total:
        return speech_windows_flagged(audio, max_seconds, min_silence_ms, min_window, mode="fixed")
    windows: list[list[float]] = []
    for st, en in spans:
        if windows and en - windows[-1][0] <= max_seconds:
            windows[-1][1] = en
        else:
            windows.append([st, en])
    out = [(a, b, False) for a, b in windows if b - a >= min_window]
    if mode in ("energy", "auto") and out:
        out += [(a, b, True) for a, b in _loud_gap_windows(frms, [(a, b) for a, b, _ in out], total, max_seconds, hop) if b - a >= min_window]
        out.sort(key=lambda w: w[0])
    return out


def speech_windows(audio: np.ndarray, max_seconds: float = 28.0, min_silence_ms: int = 400, min_window: float = 0.4, mode: str = "vad") -> list[tuple[float, float]]:
    return [(a, b) for a, b, _ in speech_windows_flagged(audio, max_seconds, min_silence_ms, min_window, mode)]


def _periodic(words: list[str], maxp: int = 12) -> tuple[int, int, int]:
    """Longest stretch (in words) where words[i] == words[i+p] for some period p <= maxp: (length, p, start)."""
    best = (0, 0, 0); n = len(words)
    for p in range(1, maxp + 1):
        i = 0
        while i + p < n:
            j = i
            while j + p < n and words[j] == words[j + p]:
                j += 1
            if j - i >= p and j - i > best[0]:
                best = (j - i, p, i)
            i = j + 1
    return best


def is_repetitive(text: str, seconds: float) -> bool:
    """Decoder loop test, calibrated 2026-09-21 on 26,818 windows (flags ~0.1%): a periodic stretch of >= 16 words
    with period <= 12 ("בין כל שנה, בין כל שנה, ..."), a run of >= 4 identical words, a zlib compression ratio > 3.5
    (normal Yiddish windows: p50 2.09, p99 2.54, p99.9 3.95), or more than 5 words per second (p99 = 2.95)."""
    import zlib
    words = text.split()
    if len(words) < 4:
        return False
    if _periodic(words)[0] >= 16:
        return True
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        if run >= 4:
            return True
    raw = text.encode("utf-8")
    return len(raw) / max(1, len(zlib.compress(raw))) > 3.5 or len(words) / max(0.5, seconds) > 5.0


def cut_repeats(text: str, seconds: float | None = None) -> str:
    """Keep the text up to and including the first period of the loop; drop the rest."""
    words = text.split()
    length, p, start = _periodic(words)
    if length >= 16 or (p == 1 and length >= 3):
        return " ".join(words[:start + p]).strip()
    if seconds:
        return " ".join(words[:int(5 * seconds) + 5]).strip()
    return text


class HFTranscriber:
    def __init__(self, model_dir: str, language: str = "yi", beam: int = 5, batch: int = 8, device: str | None = None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.language, self.beam, self.batch = language, beam, batch
        self.proc = WhisperProcessor.from_pretrained(model_dir, language=language, task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir, torch_dtype=self.dtype).to(self.device).eval()

    def _generate(self, clips: list[np.ndarray], max_new_tokens: int, **extra) -> list[tuple[str, float]]:
        feats = self.proc.feature_extractor(clips, sampling_rate=SR, return_tensors="pt").input_features.to(self.device, self.dtype)
        with self.torch.no_grad():
            gen = self.model.generate(feats, language=self.language, task="transcribe", num_beams=self.beam, max_new_tokens=max_new_tokens,
                                      return_dict_in_generate=True, output_scores=True, **extra)
        texts = self.proc.batch_decode(gen.sequences, skip_special_tokens=True)
        scores = gen.sequences_scores.tolist() if getattr(gen, "sequences_scores", None) is not None else [0.0] * len(texts)
        return [(t.strip(), float(s)) for t, s in zip(texts, scores)]

    def transcribe_clips(self, clips: list[np.ndarray]) -> list[tuple[str, float]]:
        """Decode <=30 s clips in batches; returns (text, mean token log-prob of the winning beam) per clip.

        Repetition guard (2026-09-21): plain beam search occasionally loops ("בית בית בית ..."); ~1% of windows in the
        first GPU pass (31 windows). Windows whose text looks repetitive (is_repetitive) are re-decoded with a repetition penalty
        and a no-repeat-ngram constraint; if still repetitive the looped tail is cut. The token budget also scales
        with the clip length so a runaway loop cannot fill 225 tokens for a 7 s window.
        """
        out: list[tuple[str, float]] = []
        for b in range(0, len(clips), self.batch):
            chunk = clips[b:b + self.batch]
            budget = int(min(225, 40 + 9 * max(len(c) for c in chunk) / SR))
            res = self._generate(chunk, budget)
            redo = [i for i, (t, _) in enumerate(res) if is_repetitive(t, len(chunk[i]) / SR)]
            if redo:
                fixed = self._generate([chunk[i] for i in redo], budget, repetition_penalty=1.3, no_repeat_ngram_size=3)
                for i, (t, lp) in zip(redo, fixed):
                    if is_repetitive(t, len(chunk[i]) / SR):
                        t = cut_repeats(t, len(chunk[i]) / SR)
                    res[i] = (t, lp)
            out += res
        return out

    def transcribe(self, audio: np.ndarray, max_seconds: float = 28.0, min_silence_ms: int = 400, mode: str = "auto") -> list[SimpleNamespace]:
        """Whole recording -> segments (start, end, text, avg_logprob, no_speech_prob) like faster-whisper's."""
        windows = speech_windows_flagged(audio, max_seconds, min_silence_ms, mode=mode)
        clips = [audio[int(a * SR):int(b * SR)] for a, b, _ in windows]
        res = self.transcribe_clips(clips) if clips else []
        segs = []
        for (a, b, added), (t, lp) in zip(windows, res):
            if mode == "auto" and added and (lp < GATE_LOGPROB or len(t.split()) / max(0.1, b - a) < GATE_WPS):
                continue      # a loud gap that decodes without confidence is noise or singing, not missed speech
            segs.append(SimpleNamespace(start=a, end=b, text=t, avg_logprob=lp, no_speech_prob=0.0, added=added))
        return segs
