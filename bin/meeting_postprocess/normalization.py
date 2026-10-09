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


def clean_speaker_annotations(text: str, source: str, aliases: dict[str, str], *, approved_passages=()) -> str:
    """Remove model-invented identity pairings; never infer a speaker alias."""
    from .motions import role_names
    evidence = {"mover": set(), "seconder": set()}
    source_turns = re.findall(r"^\[([^\]]+)\][ \t]*(.*)$", source, re.MULTILINE)
    for line in source.splitlines():
        if re.search(r"\b(?:if|hypothetically|maybe|joking|joke|would be)\b", line, re.IGNORECASE):
            continue
        for role, name in role_names(line):
            evidence[role].add(name.casefold())
    person = r"[A-Z][\w'’.-]*(?:[ \t]+[A-Z][\w'’.-]*){0,3}"
    pair = re.compile(r"(?P<label>SPEAKER_\d+)\s*\((?P<name>" + person + r")\)|(?P<reverse>" + person + r")\s*\((?:\*\*)?(?P<reverse_label>SPEAKER_\d+)(?:\*\*)?\)")
    def replace(match):
        label, name = match["label"] or match["reverse_label"], match["name"] or match["reverse"]
        # A name elsewhere cannot establish this pairing. A turn override needs
        # both approved original-label provenance and the exact local passage.
        body = re.sub(r"^(?:\*\*|__)", "", text[match.end():].split("\n", 1)[0]).strip(" .")
        approved = {approved_name for original, approved_name, passage in approved_passages
                    if original == label and body == passage.strip(" .")
                    and {speaker for speaker, statement in source_turns if statement.strip(" .") == body} == {approved_name}}
        if len(approved) == 1:
            return next(iter(approved))
        lead = text[max(0, match.start() - 40):match.start()]
        role = re.search(r"\b(?P<role>motion by|moved by|made by|seconded by|mover|seconder)\s*:?\s*$", lead, re.IGNORECASE)
        if role:
            kind = "seconder" if role["role"].casefold().startswith("second") else "mover"
            if name.casefold() in evidence[kind]:
                return name  # Preserve the explicitly announced role, not a global pairing.
        if label in aliases:
            return aliases[label]
        if re.search(r"\b(?:chair|chairperson|secretary|treasurer|representative|rep|officer)\b", name, re.IGNORECASE):
            return match[0]  # A role annotation is not a proposed person identity.
        return label
    return pair.sub(replace, text)
