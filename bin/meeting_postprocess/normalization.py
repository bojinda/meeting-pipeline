"""Deterministic transcript repairs; never infer identities or business facts."""
from __future__ import annotations

import re


SPEAKER_LABEL = re.compile(r"\bSPEAKER_\d+\b")
MAC_YARD_VARIANT = re.compile(r"\b(?:mack\s*yard|mac\s*yard)\b", re.IGNORECASE)
MOJIBAKE = re.compile(r"Ã[\S\u0080-\u009f]|Â[\S\u0080-\u009f]|â(?:€|[\u0080-\u009f])|ðŸ|ï¿½|\ufffd")


def _encoding_repairs() -> dict[str, str]:
    repairs = {}
    # Repair only known UTF-8 sequences misread as Western encodings. Converting
    # whole lines would damage correctly encoded accents elsewhere on the line.
    characters = "‘’“”–—…€™\u00a0" + "".join(chr(i) for i in range(0xA1, 0x100))
    for character in characters:
        for encoding in ("cp1252", "latin1"):
            try:
                broken = character.encode("utf-8").decode(encoding)
            except UnicodeDecodeError:
                continue
            repairs[broken] = character
    return repairs


_REPAIRS = _encoding_repairs()
_BROKEN_SEQUENCE = re.compile("|".join(re.escape(s) for s in sorted(_REPAIRS, key=len, reverse=True)))


def normalize_text(text: str) -> str:
    # Two passes also handle common double-encoded punctuation.
    for _ in range(2):
        text = _BROKEN_SEQUENCE.sub(lambda match: _REPAIRS[match.group()], text)
    text = MAC_YARD_VARIANT.sub("Mac Yard", text)
    return re.sub(r"\bhypodermical\b", "hypodermic", text, flags=re.IGNORECASE)


def apply_aliases(text: str, aliases: dict[str, str]) -> str:
    return SPEAKER_LABEL.sub(lambda match: aliases.get(match.group(), match.group()), text)


def prepare_text(text: str, aliases: dict[str, str]) -> str:
    return apply_aliases(normalize_text(text), aliases)
