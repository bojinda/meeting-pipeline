"""Conservative, stateful section labels with reviewable boundary evidence."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .normalization import prepare_text


PRE_MEETING = "pre_meeting_chatter"
RECAP = "previous_meeting_recap"
BUSINESS = "current_meeting_business"
ADJOURNMENT = "adjournment"

_CURRENT = re.compile(
    r"\b(?:call (?:this |the )?meeting to order|meeting (?:is|has been) called to order|"
    r"(?:let['’]s|let us|we (?:will|can|shall)) (?:begin|start) (?:the |this )?meeting|"
    r"(?:move|moving|turn|turning) (?:on )?to (?:the )?(?:new|current|today['’]s) business|"
    r"next (?:item|topic)(?: on (?:the|our) agenda)?|(?:new|current) business\s*[:.,]|"
    r"(?:approve|adopt|accept|approval of|adoption of) (?:the )?(?:previous |last (?:meeting['’]s )?)?minutes)(?=$|\W)",
    re.IGNORECASE,
)
_RECAP_START = re.compile(
    r"^(?:(?:okay|ok|so|now)[,.]?\s+)?(?:"
    r"(?:recap|review|summary) of (?:the )?(?:previous|last) meeting|"
    r"(?:let['’]s|let us|we (?:will|shall)) (?:recap|review|go over) (?:the )?(?:previous|last) meeting|"
    r"(?:let['’]s|let us|we (?:will|shall)) (?:review|read|go over) (?:the )?minutes (?:from|of) (?:the |our )?(?:previous|last) meeting|"
    r"(?:at|during|in) (?:the |our )?(?:previous|last) (?:meeting|month['’]s meeting)|"
    r"(?:last|previous) meeting[,.:])",
    re.IGNORECASE,
)
_HISTORICAL = re.compile(
    r"\b(?:was|were|had|did|approved|agreed|decided|discussed|reported|carried|defeated|"
    r"withdrawn|tabled|seconded|assigned|previous meeting|last meeting|last month)\b",
    re.IGNORECASE,
)
_CURRENT_INTENT = re.compile(
    r"^(?:(?:so|now|okay)[,.]?\s+)?(?:"
    r"(?:we|I) (?:need|should|must|will|shall|move|propose|second)\b|"
    r"(?:let['’]s|let us) (?:discuss|consider|decide|vote)\b|"
    r"(?:today|this meeting|for (?:this|today['’]s) meeting)\b)",
    re.IGNORECASE,
)
_CHATTER = re.compile(
    r"^(?:hello\b|hi\b|good (?:morning|afternoon|evening)\b|"
    r"(?:can|could) (?:you|everyone) hear me\b|(?:we['’]re|we are|still) waiting (?:for|on)\b|"
    r"(?:before|until) (?:we|the meeting) (?:start|starts|begin|begins)\b)",
    re.IGNORECASE,
)
_ADJOURN = re.compile(
    r"^(?:(?:okay|ok|so|now)[,.]?\s+)?(?:"
    r"(?:the |this )?meeting (?:is|stands|has been) adjourned\b|"
    r"(?:we are|we['’]re) adjourned\b|(?:that|this) concludes (?:the |our |this )?meeting\b)",
    re.IGNORECASE,
)
_TURN = re.compile(r"^\[([^\]\n]+)\]\s*(.*)$")


@dataclass
class SectionClassifier:
    section: str = BUSINESS
    business_started: bool = False

    def classify(self, text: str) -> tuple[str, str]:
        if self.section == ADJOURNMENT:
            return ADJOURNMENT, "After explicit adjournment"
        if _ADJOURN.search(text):
            self.section = ADJOURNMENT
            return self.section, "Explicit adjournment"
        if _CURRENT.search(text) or _CURRENT_INTENT.search(text):
            self.section = BUSINESS
            self.business_started = True
            return self.section, "Explicit current-business boundary"
        if _RECAP_START.search(text):
            self.section = RECAP
            self.business_started = True
            return self.section, "Explicit previous-meeting recap"
        if self.section == RECAP and _HISTORICAL.search(text):
            return RECAP, "Historical continuation of explicit recap"
        if not self.business_started and _CHATTER.search(text):
            return PRE_MEETING, "Opening greeting, setup, or waiting chatter"
        self.section = BUSINESS
        self.business_started = True
        return BUSINESS, "No explicit section cue; retained as current business"


def prepare_chunks(chunks: list[dict], aliases: dict[str, str]) -> list[dict]:
    """Split at section boundaries while preserving every word and chunk order.

    Source times remain the containing chunk's range: the chunk index does not
    supply timestamps for each utterance, so no precise boundary is invented.
    """
    classifier = SectionClassifier()
    prepared = []
    for index, chunk in enumerate(chunks, start=1):
        runs: list[dict] = []
        speaker = ""
        for line in prepare_text(chunk.get("text", ""), aliases).splitlines():
            match = _TURN.match(line)
            if match:
                speaker, line = match.groups()
            if not line.strip():
                continue
            # A chunk turn may contain several phases merged by the chunker.
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z‘’\"'])", line.strip()):
                section, reason = classifier.classify(sentence)
                if not runs or runs[-1]["meeting_section"] != section:
                    runs.append({"meeting_section": section, "lines": [], "speakers": [], "section_evidence": []})
                run = runs[-1]
                run["lines"].append(f"[{speaker}] {sentence}" if speaker else sentence)
                if speaker and speaker not in run["speakers"]:
                    run["speakers"].append(speaker)
                if reason not in run["section_evidence"]:
                    run["section_evidence"].append(reason)
        for part, run in enumerate(runs, start=1):
            row = dict(chunk)
            source_id = chunk.get("chunk_id", index)
            row.update(
                chunk_id=source_id if len(runs) == 1 else f"{source_id}.{part}",
                source_chunk_id=source_id,
                meeting_section=run["meeting_section"],
                section_evidence=run["section_evidence"],
                speaker_span=" -> ".join(run["speakers"]) or prepare_text(chunk.get("speaker_span", ""), aliases),
                text="\n".join(run["lines"]),
            )
            if run["speakers"]:
                row["speaker_count"] = len(run["speakers"])
                row["chunk_type"] = "single_speaker" if len(run["speakers"]) == 1 else "discussion"
            row["word_count"] = len(row["text"].split())
            prepared.append(row)
    return prepared
