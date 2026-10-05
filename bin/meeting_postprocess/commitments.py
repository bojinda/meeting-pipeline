"""Small, source-backed future undertakings for meeting reduce prompts only."""
from __future__ import annotations

import re

from .sections import ADJOURNMENT, BUSINESS


_FUTURE = r"(?:i['’]ll|i will|i(?:['’]m| am) (?:going to|gonna))"
_COMMITMENT = re.compile(
    r"^(?:[^.!?]{1,300}\b(?:but|and then|so),?\s+)?"
    r"(?:(?:okay|ok|yeah|so|well|uh|um|oh|and|but|then)[,.]?\s+)*"
    r"(?:(?:when|once|if)\b[^.!?]{1,160}?[\s,]+)?"
    r"(?:" + _FUTURE + r"\s*,\s*){0,2}" + _FUTURE + r"\s+"
    r"(?:try to\s+)?(?P<task>.+)$", re.IGNORECASE,
)
_TASK_VERBS = (
    r"(?:approve|request|contact|arrange|schedule|send|circulate|prepare|share|review|"
    r"check|investigate|report|update|submit|gather|collect|isolate|identify|file|grieve|"
    r"ask|call|confirm|coordinate|inform|notify|follow up|touch base|reach out|meet with|find out)"
)
_TASK = re.compile(
    r"^" + _TASK_VERBS + r"\b|"
    r"^be (?:filing|grieving|contacting|sending|preparing|reviewing|investigating)\b|"
    r"^get (?:our|my|the) (?:team|staff|people|guys)\s+"
    r"(?:to\s+(?:start|begin|do|use|follow|" + _TASK_VERBS + r")|"
    r"(?:start|begin)\s+(?:doing|sending|using|following)|doing|sending|using|following)\b|"
    r"^let (?:(?:all|both) )?(?:the |our )?(?:members|team|staff|participants|committee|board) know\b|"
    r"^get (?:this|that|it) out\b",
    re.IGNORECASE,
)
_UNSUPPORTED = re.compile(
    r"\b(?:joking|joke|kidding|hypothetically|pretend|suppose|"
    r"said|says|told|quoted|i think|if i (?:had|could|were))\b", re.IGNORECASE,
)
_QUOTED_FRAME = re.compile(
    r"\b(?:(?:i|we|you)\s+(?:(?:would|might|could)(?:\s+then)?\s+)?(?:say|reply|respond|tell)|"
    r"for example|for instance|as an example|imagine|let['’]s say|let us say)\b",
    re.IGNORECASE,
)
_THIRD_PARTY_STEP = re.compile(
    r"(?:\s*;\s*|,\s*(?:and\s+)?|\s+and\s+)"
    r"(?:(?:if|when|once)\b[^.!?;]{1,180}?(?:,\s*|\s+))?(?:then\s+)?"
    r"(?P<actor>they|he|she|(?:the|our) [a-z]+(?: [a-z]+){0,3}|"
    r"(?!i\b|i['’]|we\b|we['’])(?-i:[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,3}))\s+"
    r"(?:(?:will|would|can|may)\s+)?(?:sends?|forwards?|gives?|delivers?|pass(?:es)?|provides?|shares?|hands?|redacts?)\b",
    re.IGNORECASE,
)


def commitment_evidence(chunks: list[dict]) -> list[str]:
    """Retain speaker and their qualified wording; do not generate action rows.

    Only already-redacted, classified current sections may contribute. A narrow
    task-verb check excludes conversational promises and capability statements.
    The bounded, deduplicated evidence is supplemental, never an assignment.
    """
    evidence = []
    for chunk in chunks:
        if chunk.get("meeting_section") not in {BUSINESS, ADJOURNMENT}:
            continue
        for line in chunk.get("text", "").splitlines():
            turn = re.match(r"^\[([^\]\n]+)\]\s*(.*)", line)
            if not turn:
                continue
            speaker, text = turn.groups()
            commitment = _COMMITMENT.fullmatch(text)
            if not commitment or not _TASK.search(commitment["task"]) or _UNSUPPORTED.search(text):
                continue
            # A conditional undertaking may be genuine. Reject speech/example
            # framing before the selected promise, not every conditional clause.
            if _QUOTED_FRAME.search(text[:commitment.start("task")]):
                continue
            if len(text) > 1000:
                continue  # Do not truncate a qualification in a long ASR turn.
            # Quote only the owner's undertaking when a later delivery clause
            # changes actor. Its recipient/transformation is not this task's.
            for step in _THIRD_PARTY_STEP.finditer(text, commitment.start("task")):
                if step["actor"].casefold() != speaker.casefold():
                    text = text[:step.start()].rstrip(" ,;")
                    break
            record = f"[{speaker}] {text}"
            if record not in evidence:
                evidence.append(record)
                if len(evidence) == 32:
                    return evidence
    return evidence
