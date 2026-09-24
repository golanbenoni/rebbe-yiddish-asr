"""Decoding options shared by the faster-whisper callers (evaluate, evaluate_longform, transcribe_file).

Whisper suppresses a fixed list of "non-speech" symbol tokens at every decoding step, and the ASCII double quote
is on that list. In Hebrew/Yiddish text the double quote is the gershayim of abbreviations (רש"י, הקב"ה, עאכו"כ):
the fine-tuned models learned to write it but could never emit it, so they produced the nearest allowed spelling
(רשע, רש'א, הקב') - 30% of abbreviations were scored wrong (reports/error-analysis-turbo-full-lr1e5-3ep.md).
--suppress allow-quotes keeps the rest of Whisper's list and frees the quote, colon, semicolon and dashes.
"""
from __future__ import annotations

ALLOWED = {'"', ":", ";", "--", "---"}


def add_suppress_arg(ap) -> None:
    ap.add_argument("--suppress", default="default", choices=["default", "none", "allow-quotes"],
                    help='default = Whisper\'s non-speech symbol list (blocks the gershayim "); allow-quotes = that list minus " : ; -- ---; none = no symbol suppression')


def suppress_tokens(model, mode: str, language: str) -> list[int]:
    """Return the suppress_tokens argument for WhisperModel.transcribe for the given mode."""
    if mode == "default":
        return [-1]
    if mode == "none":
        return []
    from faster_whisper.tokenizer import Tokenizer
    tok = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task="transcribe", language=language)
    return [t for t in tok.non_speech_tokens if model.hf_tokenizer.decode([t]).strip() not in ALLOWED]
