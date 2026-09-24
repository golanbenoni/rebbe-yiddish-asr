"""Text normalization for Chabad Yiddish/Hebrew transcripts.

Shared by dataset building (training targets) and scoring (WER/CER), so the
same rules apply everywhere. Hanachos use YIVO-style pasekh/komets (אַ, אָ);
the pretrained Yiddish Whisper checkpoints emit text without them, so v1
training targets drop diacritics. Everything is reversible: originals are kept
in the cached day JSON.
"""
from __future__ import annotations

import html
import re

NIKKUD = re.compile(r"[֑-ׇ]")  # nikkud + te'amim: אָ -> א, אַ -> א
HTML_TAG = re.compile(r"<[^>]+>")
PARA = re.compile(r"<p[^>]*>(.*?)</p>", re.S)
HYPHENS = re.compile(r"[־–—\-]")
QUOTES = re.compile(r"[\"״“”'׳‘’`]")
NON_HEBREW = re.compile(r"[^א-ת\s]")
SPACES = re.compile(r"\s+")
INVISIBLE = re.compile(r"[‎‏‪-‮﻿]")
BRACKETED = re.compile(r"\[[^\]]*\]")  # kept: measured as mostly spoken in these hanachos (WER rises 0.04 if dropped)
STRAY_BRACKETS = re.compile(r"[\[\]]")
JUNK_TOKEN = re.compile(r"\S*[A-Za-z\u0250-\u02af\u0370-\u03ff\u0400-\u04ff]\S*")  # tokens with Latin/IPA/Greek/Cyrillic letters: PDF footer small print decoded through the wrong font (2026-09-23), URLs, e-mail
HEADER_WORDS = ("בס\"ד", "בס״ד", "הנחה", "תרגום", "UNEDITED", "BS")


def strip_html(text: str) -> str:
    return html.unescape(HTML_TAG.sub(" ", text))


def hanacha_paragraphs(content_html: str, drop_headers: bool = True) -> list[str]:
    """Paragraph texts of a Daily Sicha contentHtml; drops the unspoken header lines
    (source citation, 'הנחה פרטית בלתי מוגה', translation notices)."""
    paras = [strip_html(p).strip() for p in PARA.findall(content_html)]
    paras = [p for p in paras if p]
    if drop_headers:
        while paras and paras[0].startswith(HEADER_WORDS):
            paras.pop(0)
    return paras


def training_text(text: str, keep_diacritics: bool = False) -> str:
    """Target text for fine-tuning: keeps punctuation and abbreviation marks
    (ר"ה, גדלי'), unifies quote characters, removes invisible marks."""
    text = strip_html(text)
    text = JUNK_TOKEN.sub(" ", text)     # nothing spoken in these hanachos is written in Latin letters
    text = STRAY_BRACKETS.sub("", text)  # keep bracketed words, drop the bracket characters
    text = INVISIBLE.sub("", text).replace("\xa0", " ")
    if not keep_diacritics:
        text = NIKKUD.sub("", text)
    text = text.replace("״", '"').replace("׳", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return SPACES.sub(" ", text).strip()


def scoring_text(text: str) -> str:
    """Aggressive normalization for WER/CER: Hebrew letters and single spaces only.
    Bracketed passages are kept: in these hanachos they are spoken asides, not insertions."""
    text = STRAY_BRACKETS.sub("", strip_html(text))
    text = NIKKUD.sub("", text)
    text = HYPHENS.sub(" ", text)
    text = QUOTES.sub("", text)
    text = NON_HEBREW.sub(" ", text)
    return SPACES.sub(" ", text).strip()
