"""Conservative, stateful section labels with reviewable boundary evidence."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .normalization import prepare_text


PRE_MEETING = "pre_meeting_chatter"
RECAP = "previous_meeting_recap"
BUSINESS = "current_meeting_business"
ADJOURNMENT = "adjournment"

_DISCOURSE = r"(?:okay|ok|yeah|yes|so|now|well|all right|alright|uh|um|oh)"
_LEAD_IN = r"^(?:" + _DISCOURSE + r"[,.]?\s+)*"
_FORMAL_START = re.compile(
    _LEAD_IN + r"(?:(?:i (?:guess|think)|i suppose)\s+)?(?:"
    r"(?:i(?:['’]ll| will)? )?call (?:this |the )?meeting to order|"
    r"(?:the |this )?meeting (?:is|has been) called to order|"
    r"(?:let['’]s|let us|we(?:['’]ll| will| can| shall)) "
    r"(?:get (?:the |this )?meeting started\b|(?:begin|start) (?:the |this )?meeting\b|"
    r"get started(?: (?:there|then|now))?[.!?]*$))",
    re.IGNORECASE,
)
_READY_PREFIX = (
    _LEAD_IN + r"(?:(?:is|are)\s+)?(?:you\s+)?(?:guys|everybody|everyone|folks),?\s+"
    r"(?:are\s+)?(?:you\s+)?all set(?: here| to (?:start|begin)(?: the meeting)?)?"
)
_CHAIR_READY = re.compile(_READY_PREFIX + r"[.!?]*$", re.IGNORECASE)
_START_CONSTRUCTION = r"(?:(?:let['’]s|let us|we(?:['’]ll| will| can| shall))\s+)?start (?:off )?with "
_CHAIR_START_PREFIX = _LEAD_IN + _START_CONSTRUCTION
_OPENING_OBJECT = r"(?:a |the |our )?(?:recap|agenda|roll call|minutes|introductions|reports|first (?:report|item))\b"
_CHAIR_AGENDA_START = re.compile(
    _CHAIR_START_PREFIX + _OPENING_OBJECT,
    re.IGNORECASE,
)
_JOINED_CHAIR_START = re.compile(
    _READY_PREFIX + r"[\s,;:]+(?:" + _DISCOURSE + r"[,.]?\s+){0,3}(?P<agenda>" + _START_CONSTRUCTION + _OPENING_OBJECT + r".*)",
    re.IGNORECASE,
)


def _formal_start(text: str) -> bool:
    return bool(_FORMAL_START.search(text) or _CHAIR_READY.search(text) or _CHAIR_AGENDA_START.search(text) or _JOINED_CHAIR_START.search(text))


_CURRENT_TRANSITION = re.compile(
    r"\b(?:"
    r"(?:move|moving|turn|turning) (?:on )?to (?:the )?(?:new|current|today['’]s) business|"
    r"(?:we(?:['’]ll| will)|let['’]s|let us) move (?:right )?(?:along|on)(?:\s+then)?|"
    r"moving (?:right )?(?:along|on)(?:\s+then)?|"
    r"next (?:item|topic)(?: on (?:the|our) agenda)?|(?:new|current) business\s*[:.,]|"
    r"(?:approve|adopt|accept|approval of|adoption of) (?:the )?(?:previous |last (?:meeting['’]s )?)?minutes)(?=$|\W)",
    re.IGNORECASE,
)
_FIRST_REPORT = re.compile(
    _LEAD_IN + r"(?:"
    r"(?:does )?(?:anybody|anyone) (?:want|wants|like) to (?:volunteer to )?go first|"
    r"(?:would|can) (?:anybody|anyone) (?:volunteer to )?go first|"
    r"i (?:can|will|could) go first)\b",
    re.IGNORECASE,
)
_ACKNOWLEDGMENT = re.compile(
    _LEAD_IN + r"thanks(?: (?:a lot|very much|everyone))?[.!]?$", re.IGNORECASE,
)
_PREVIOUS_MEETING = r"(?:the |our )?(?:previous|last)(?: (?:month|week)['’]s)? meeting"
_RETROSPECTIVE = re.compile(
    _LEAD_IN + r"(?:(?:at|during|in)\s+)?" + _PREVIOUS_MEETING + r"(?=$|[\s,.:])",
    re.IGNORECASE,
)
_RECAP_INTENT = re.compile(
    _CHAIR_START_PREFIX + r"(?:a |the |our )?(?:little |brief )?recap(?: of " + _PREVIOUS_MEETING + r")?[.!?]*$",
    re.IGNORECASE,
)
_HISTORICAL_REPORT = re.compile(
    _LEAD_IN + r"(?:we|i|they|[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,2})\s+"
    r"(?:discussed|reported|approved|agreed|decided|reviewed|noted|raised)\b",
    re.IGNORECASE,
)
_CURRENT_DISCUSSION = re.compile(
    r"\b(?:currently|right now|today|yesterday|recently|last night|this (?:morning|afternoon|evening|week)|"
    r"(?:i|we|they)(?:['’]ve| have)? just (?:had|saw|discussed|noticed|been))\b",
    re.IGNORECASE,
)
_FILLER = re.compile(r"^(?:" + _DISCOURSE + r"[,.]?\s*)+[.!?]?$", re.IGNORECASE)
_RECAP_START = re.compile(
    _LEAD_IN + r"(?:"
    r"(?:recap|review|summary) of " + _PREVIOUS_MEETING + r"|"
    r"(?:let['’]s|let us|we (?:will|shall)) (?:recap|review|go over) " + _PREVIOUS_MEETING + r"|"
    r"(?:let['’]s|let us|we (?:will|shall)) (?:review|read|go over) (?:the )?minutes (?:from|of) " + _PREVIOUS_MEETING + r"|"
    r"(?:do (?:you|we) have|have you got) (?:the |any )?(?:notes|minutes) (?:for|from|of) " + _PREVIOUS_MEETING + r"|"
    r"(?:at|during|in) " + _PREVIOUS_MEETING + r"|"
    r"(?:last|previous) meeting[,.:])",
    re.IGNORECASE,
)
_RECAP_REQUEST = re.compile(
    _LEAD_IN + r"(?:"
    r"if you (?:wouldn['’]t|would not) mind (?:doing|giving(?: us)?) (?:a )?(?:little |brief )?recap|"
    r"(?:can|could|would) you (?:do|give(?: us)?) (?:a )?(?:little |brief )?recap)\b",
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
_ADJOURN_MOTION = re.compile(
    _LEAD_IN + r"(?:a )?motion to adjourn(?: the meeting)?[.!]?$", re.IGNORECASE,
)
_TURN = re.compile(r"^\[([^\]\n]+)\]\s*(.*)$")


@dataclass
class SectionClassifier:
    section: str = BUSINESS
    business_started: bool = False
    awaiting_formal_start: bool = False
    opening_recap_window: bool = False
    recap_pending: bool = False

    def classify(self, text: str, following_text: str = "") -> tuple[str, str]:
        if self.section == ADJOURNMENT:
            return ADJOURNMENT, "After explicit adjournment"
        if _formal_start(text):
            self.awaiting_formal_start = False
            self.business_started = True
            self.opening_recap_window = True
            self.section = BUSINESS
            joined = _JOINED_CHAIR_START.search(text)
            if _RECAP_INTENT.search(text) or (joined and _RECAP_INTENT.search(joined["agenda"])):
                self.section = RECAP
                self.recap_pending = True
                return RECAP, "Formal start announcing an intended recap; awaiting historical content"
            self.recap_pending = False
            return BUSINESS, "Explicit formal start (including informal get-started wording)"
        if self.awaiting_formal_start:
            self.section = PRE_MEETING
            return PRE_MEETING, "Conversation before a recognized formal start"
        if _ADJOURN.search(text) or _ADJOURN_MOTION.fullmatch(text):
            self.section = ADJOURNMENT
            return self.section, "Explicit adjournment"
        transition = _CURRENT_TRANSITION.search(text) or _FIRST_REPORT.search(text)
        acknowledgment = _ACKNOWLEDGMENT.search(text) and (
            _CURRENT_TRANSITION.search(following_text) or _FIRST_REPORT.search(following_text)
        )
        if transition or acknowledgment:
            self.section = BUSINESS
            self.business_started = True
            self.opening_recap_window = False
            self.recap_pending = False
            return self.section, "Explicit current-business boundary"
        if _RETROSPECTIVE.search(text) or (_FILLER.fullmatch(text) and _RETROSPECTIVE.search(following_text)):
            self.section = RECAP
            self.business_started = True
            self.recap_pending = False
            return RECAP, "Explicit retrospective report begins/resumes previous-meeting recap"
        if _RECAP_START.search(text) or (
            _RECAP_REQUEST.search(text) and (self.opening_recap_window or self.section == RECAP)
        ):
            self.section = RECAP
            self.business_started = True
            self.recap_pending = True
            return self.section, "Previous-meeting recap requested; awaiting historical content"
        if _CURRENT_DISCUSSION.search(text):
            self.section = BUSINESS
            self.business_started = True
            self.opening_recap_window = False
            self.recap_pending = False
            return BUSINESS, "Explicit current/recent discussion interrupts any recap"
        if self.recap_pending and _HISTORICAL_REPORT.search(text):
            self.section = RECAP
            self.recap_pending = False
            return RECAP, "Historical report follows the requested recap"
        if self.section == RECAP and not self.recap_pending:
            return RECAP, "Continuation of explicit recap until a current-business boundary"
        if self.recap_pending and _FILLER.fullmatch(text):
            return RECAP, "Brief acknowledgment while awaiting the requested recap"
        if not self.business_started and _CHATTER.search(text):
            return PRE_MEETING, "Opening greeting, setup, or waiting chatter"
        self.section = BUSINESS
        self.business_started = True
        self.opening_recap_window = False
        self.recap_pending = False
        return BUSINESS, "No explicit section cue; retained as current business"


def _utterances(text: str) -> list[tuple[str, str]]:
    utterances = []
    speaker = ""
    for line in text.splitlines():
        match = _TURN.match(line)
        if match:
            speaker, line = match.groups()
        if line.strip():
            # A turn can cross phases; lowercase sentences are common in ASR.
            utterances.extend((speaker, sentence) for sentence in re.split(r"(?<=[.!?])\s+", line.strip()))
    return utterances


def prepare_chunks(chunks: list[dict], aliases: dict[str, str]) -> list[dict]:
    """Split at section boundaries while preserving every word and chunk order.

    Source times remain the containing chunk's range: the chunk index does not
    supply timestamps for each utterance, so no precise boundary is invented.
    """
    inputs = [(chunk, _utterances(prepare_text(chunk.get("text", ""), aliases))) for chunk in chunks]
    sentences = [text for _, utterances in inputs for _, text in utterances]
    # Look ahead for a formal opening before interpreting any opening tasks as
    # business. Recordings that start mid-meeting keep the BUSINESS fallback.
    classifier = SectionClassifier(awaiting_formal_start=any(_formal_start(text) for text in sentences))
    sentence_index = 0
    prepared = []
    for index, (chunk, utterances) in enumerate(inputs, start=1):
        runs: list[dict] = []
        for speaker, sentence in utterances:
            following = sentences[sentence_index + 1] if sentence_index + 1 < len(sentences) else ""
            sentence_index += 1
            section, reason = classifier.classify(sentence, following)
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
