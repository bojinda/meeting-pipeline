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


_CONTEXT_PROMISE = re.compile(r"^do (?:that|it)(?:[.!?]|$|\s+(?:after|before|when|once|tomorrow|later|next)\b)", re.IGNORECASE)
_SHARING_CONTEXT = re.compile(
    r"\b(?:i|we)\s+(?:(?:can|will|should|need to|want to)\s+)?"
    r"(?:share|send|circulate)\s+[^.!?;]*?"
    r"\b(?:minutes|reports?|notes|documents?|information|files?|lists?|packages?)\b"
    r"[^.!?;]*?\b(?:to|with)\s+"
    r"(?!(?:(?:all|both|some|any) (?:of )?)?(?:you|them|him|her|it|that|this|us)\b)[a-z][^.!?;]*",
    re.IGNORECASE,
)
_CONTEXT_HYPOTHETICAL = re.compile(r"\b(?:if (?:i|we|you|they) (?:had|could|were)|would|could)\b", re.IGNORECASE)


def _sharing_context(text: str) -> bool:
    # Resolve only an adjacent, concrete sharing task with an explicit recipient.
    # Never turn a bare pronoun or a quoted/example conversation into a task.
    return bool(
        len(list(_SHARING_CONTEXT.finditer(text))) == 1
        and len(re.findall(r"\b(?:share|send|circulate)\b", text, re.IGNORECASE)) == 1
        and not (_UNSUPPORTED.search(text) or _QUOTED_FRAME.search(text)
                 or _CONTEXT_HYPOTHETICAL.search(text) or _THIRD_PARTY_STEP.search(text))
        and not any(quote in text for quote in ('"', '“', '”'))
    )


def commitment_evidence(chunks: list[dict], *, include_context: bool = False,
                        blocked_context_source_ids: set[str] | None = None) -> list[str]:
    """Retain speaker and their qualified wording; do not generate action rows.

    Only already-redacted, classified current sections may contribute. A narrow
    task-verb check excludes conversational promises and capability statements.
    The bounded, deduplicated evidence is supplemental, never an assignment.
    """
    # Contextual recovery is enabled only by the meeting map/reduce caller;
    # existing single-record evidence validation retains its strict contract.
    evidence = []
    blocked = blocked_context_source_ids or set()
    for chunk in chunks:
        if chunk.get("meeting_section") not in {BUSINESS, ADJOURNMENT}:
            continue
        previous = None
        context_safe = include_context and not (
            str(chunk.get("source_chunk_id", chunk.get("chunk_id", ""))) in blocked
            or chunk.get("speaker_review_gap") or chunk.get("redaction_gap")
        )
        for line in chunk.get("text", "").splitlines():
            turn = re.match(r"^\[([^\]\n]+)\]\s*(.*)", line)
            if not turn:
                previous = None
                continue
            speaker, text = turn.groups()
            prior = previous
            previous = (speaker, text)
            commitment = _COMMITMENT.fullmatch(text)
            if not commitment or _UNSUPPORTED.search(text):
                continue
            context = None
            if not _TASK.search(commitment["task"]):
                if not context_safe or not _CONTEXT_PROMISE.search(commitment["task"]):
                    continue
                futures = list(re.finditer(_FUTURE, text[:commitment.start("task")], re.IGNORECASE))
                prefix = text[:futures[-1].start()] if futures else ""
                if _sharing_context(prefix):
                    context = ""  # The task and undertaking are already in this exact source line.
                elif re.fullmatch(r"(?:(?:okay|ok|yeah|so|well|uh|um|oh|and|then)[,.]?\s*)*", prefix.strip(), re.IGNORECASE) and prior and prior[0] == speaker and _sharing_context(prior[1]):
                    context = prior[1]
                else:
                    continue
            # A conditional undertaking may be genuine. Reject speech/example
            # framing before the selected promise, not every conditional clause.
            if _QUOTED_FRAME.search(text[:commitment.start("task")]):
                continue
            if len(text) + len(context or "") > 1000:
                continue  # Do not truncate a qualification in a long ASR turn.
            # Quote only the owner's undertaking when a later delivery clause
            # changes actor. Its recipient/transformation is not this task's.
            for step in _THIRD_PARTY_STEP.finditer(text, commitment.start("task")):
                if step["actor"].casefold() != speaker.casefold():
                    text = text[:step.start()].rstrip(" ,;")
                    break
            record = f"[{speaker}] {text}"
            if context:
                record = f"[{speaker}] {context}\n" + record
            if record not in evidence:
                evidence.append(record)
                if len(evidence) == 32:
                    return evidence
    return evidence
