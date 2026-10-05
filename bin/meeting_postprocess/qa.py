"""Flag final-minutes defects for review; do not invent repairs or attributions."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .normalization import MAC_YARD_VARIANT, MOJIBAKE, SPEAKER_LABEL


@dataclass
class Finding:
    code: str
    line: int
    message: str
    excerpt: str


_ROLE = re.compile(r"\b(moved\s+by|seconded\s+by|mover|seconder)\b\s*:?\s*", re.IGNORECASE)
_UNKNOWN = re.compile(r"^(?:unknown|unclear|unidentified|not (?:noted|known|recorded|specified)|none(?: noted)?|n/?a|tbd|\?+)$", re.IGNORECASE)


def _role_names(line: str) -> list[tuple[str, str]]:
    # Strip emphasis first so '**Moved by:** Alice' is handled like plain text.
    plain = line.replace("**", "").replace("__", "")
    matches = list(_ROLE.finditer(plain))
    names = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(plain)
        name = re.split(r"[;|,()]|\.(?:\s|$)", plain[match.end():end], maxsplit=1)[0]
        name = re.sub(r"\s+and\s*$", "", name, flags=re.IGNORECASE).strip(" .:-*[]")
        role = "mover" if match.group(1).lower().startswith(("moved", "mover")) else "seconder"
        names.append((role, name))
    return names


def _motion_evidence(source: str) -> dict[str, set[str]]:
    evidence: dict[str, set[str]] = {"mover": set(), "seconder": set()}
    speakers = set(re.findall(r"^\[([^\]\n]+)\]", source, re.MULTILINE))
    for line in source.splitlines():
        turn = re.match(r"^\[([^\]\n]+)\]\s*(.*)", line)
        utterance = turn.group(2) if turn else line
        if turn:
            name = turn.group(1).casefold()
            if re.search(r"\bI (?:move|moved|make a motion)\b", utterance, re.IGNORECASE):
                evidence["mover"].add(name)
            if re.search(r"\bI (?:second|seconded)(?:\s+(?:it|that|this|the motion)|[.!]|$)", utterance, re.IGNORECASE):
                evidence["seconder"].add(name)
        for role, name in _role_names(utterance):
            if name:
                evidence[role].add(name.casefold())
        for name in speakers:
            for role, verb in (("mover", "moved"), ("seconder", "seconded")):
                if re.search(r"(?<!\w)" + re.escape(name) + r"\s+" + verb + r"\b", utterance, re.IGNORECASE):
                    evidence[role].add(name.casefold())
    return evidence


def _malformed_bullet(line: str) -> bool:
    stripped = line.lstrip()
    if re.fullmatch(r"(?:[-*_]\s*){3,}", stripped):
        return False  # A valid thematic break.
    if re.match(r"^(?:[-+]\S|\d+\.(?=[A-Za-z])|[•–—]\s)", stripped):
        return True
    if re.match(r"^\*[^*\s]", stripped) and not stripped.rstrip().endswith("*"):
        return True
    return bool(re.match(r"^[-+*](?:\s*$|\s+[-+*](?:\s|$))", stripped))


def check_minutes(content: str, source: str | None = None) -> list[Finding]:
    findings = []
    evidence = _motion_evidence(source) if source is not None else None
    motion_block: dict[str, str] = {}
    fence = None
    for number, line in enumerate(content.splitlines(), start=1):
        def flag(code: str, message: str) -> None:
            findings.append(Finding(code, number, message, line))

        if SPEAKER_LABEL.search(line):
            flag("unresolved_speaker", "Unresolved speaker label; supply a verified speaker alias.")
        if MOJIBAKE.search(line):
            flag("mojibake", "Possible encoding artifact remains.")
        if any(match.group() != "Mac Yard" for match in MAC_YARD_VARIANT.finditer(line)):
            flag("mac_yard_spelling", "Use the canonical spelling Mac Yard.")

        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker.group(1)[0]
            elif marker.group(1)[0] == fence:
                fence = None
            continue
        if fence is not None:
            continue
        if _malformed_bullet(line):
            flag("malformed_bullet", "Possible malformed Markdown bullet or list marker.")
        roles = _role_names(line)
        if not line.strip() or line.lstrip().startswith("#") or (re.match(r"^\s*[-+*]\s", line) and not roles):
            motion_block = {}
        for role, name in roles:
            if not name or _UNKNOWN.fullmatch(name) or SPEAKER_LABEL.search(name):
                flag("questionable_motion", f"Motion {role} is missing or unresolved; verify against the transcript.")
            elif evidence is not None and name.casefold() not in evidence[role]:
                flag("questionable_motion", f"Motion {role} '{name}' has no explicit role evidence in the source; verify attribution.")
            motion_block[role] = name.casefold()
        if roles and motion_block.get("mover") and motion_block.get("mover") == motion_block.get("seconder"):
            flag("questionable_motion", "Mover and seconder are the same person; verify the motion record.")
        if re.search(r"\b(?:moved|seconded)\b", line, re.IGNORECASE) and re.search(r"\b(?:maybe|possibly|apparently|probably)\b", line, re.IGNORECASE):
            flag("questionable_motion", "Motion attribution is tentative; verify against the transcript.")
    return findings


def write_report(directory: Path, findings: list[Finding]) -> None:
    report = {"document": "minutes-draft.md", "status": "needs_review" if findings else "passed", "findings": [asdict(f) for f in findings]}
    (directory / "minutes-qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Draft Minutes QA", "", "Review required." if findings else "No automated QA findings.", ""]
    for finding in findings:
        lines.append(f"- Line {finding.line} [{finding.code}]: {finding.message}")
    (directory / "minutes-qa.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
