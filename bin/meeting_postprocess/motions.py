"""Preserve explicit named role announcements, including those by the chair."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .normalization import SPEAKER_LABEL


_ROLE = re.compile(r"\b(moved\s+by|motion\s+by|seconded\s+by|mover|seconder)\b\s*:?\s*", re.IGNORECASE)
_ACKNOWLEDGMENT = re.compile(
    r"^(?:(?:okay|ok|yeah|yes|so|well|uh|um|oh)[,.]?\s*)+[.!?]?$", re.IGNORECASE,
)


def role_names(line: str) -> list[tuple[str, str]]:
    plain = line.replace("**", "").replace("__", "")
    matches = list(_ROLE.finditer(plain))
    names = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(plain)
        name = re.split(r"[;|,()]|\.(?:\s|$)", plain[match.end():end], maxsplit=1)[0]
        name = re.sub(r"\s+and\s*$", "", name, flags=re.IGNORECASE).strip(" .:-*[]")
        # Trailing discourse particles in a named role announcement are not
        # part of the announced person's identity.
        name = re.sub(r"(?:\s+(?:there|then|here|okay|ok|yeah|thanks))+$", "", name, flags=re.IGNORECASE)
        role = "seconder" if match.group(1).lower().startswith(("seconded", "seconder")) else "mover"
        names.append((role, name))
    return names


@dataclass(frozen=True)
class AdjournmentAnnouncement:
    mover: str
    seconder: str
    outcome: str | None

    def render(self) -> str:
        record = f"Motion to adjourn moved by {self.mover}, seconded by {self.seconder}."
        return record + (" " + self.outcome + "." if self.outcome else "")


def adjournment_announcements(source: str) -> list[AdjournmentAnnouncement]:
    turns = [re.match(r"^\[([^\]]+)\]\s*(.*)$", line) for line in source.splitlines() if line.strip()]
    original = [line for line in source.splitlines() if line.strip()]
    speakers = [turn[1] if turn else "" for turn in turns]
    lines = [turn[2] if turn else line for turn, line in zip(turns, original)]
    # ASR can split acknowledgments into separate turns between the cue and
    # its named announcement. Ignore only filler in this evidence view; keep
    # the existing bounds on substantive context and the source unchanged.
    kept = [i for i, line in enumerate(lines) if not _ACKNOWLEDGMENT.fullmatch(line)]
    speakers, lines = [speakers[i] for i in kept], [lines[i] for i in kept]
    announcements = []
    for index, line in enumerate(lines):
        if re.search(r"\b(?:if|hypothetically|maybe|joking|joke|would be)\b", line, re.IGNORECASE):
            continue
        roles = dict(role_names(line))
        role_end = index
        if len(roles) == 1:
            for following in range(index + 1, min(len(lines), index + 5)):
                if speakers[following] != speakers[index]:
                    break
                extra = dict(role_names(lines[following]))
                if extra and re.search(r"\b(?:if|hypothetically|maybe|joking|joke|would be)\b", lines[following], re.IGNORECASE):
                    break
                if extra and not set(extra) & set(roles):
                    roles.update(extra)
                    role_end = following
                    break
                filler = lines[following].strip(" .!")
                if filler not in roles.values() and not re.fullmatch(r"perfect|thanks|thank you", filler, re.IGNORECASE):
                    break
        if not all(roles.get(role) for role in ("mover", "seconder")):
            continue
        if any(SPEAKER_LABEL.search(name) or not re.fullmatch(r"[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,3}", name) for name in roles.values()):
            continue
        context = " ".join(lines[max(0, index - (4 if role_end != index else 2)):role_end + 2])
        if not re.search(r"\badjourn(?:ment)?\b", context, re.IGNORECASE):
            continue
        outcome_text = " ".join(lines[index:role_end + 3])
        outcome = None
        for phrase in ("Not carried", "Defeated", "Withdrawn", "Tabled"):
            if re.search(r"\b" + phrase + r"\b", outcome_text, re.IGNORECASE):
                outcome = phrase
                break
        if outcome is None:
            for phrase in ("Carried", "Passed", "Approved", "Adopted", "Accepted", "Ratified"):
                if re.search(
                    r"(?:^|[.!;]\s+)(?:(?:the |that )?motion (?:is |was |has been |has )?)?"
                    r"(?:unanimously )?" + phrase + r"(?: unanimously| without objection)?(?:[.!;]|$)",
                    outcome_text, re.IGNORECASE,
                ):
                    outcome = phrase
                    break
        announcement = AdjournmentAnnouncement(roles["mover"], roles["seconder"], outcome)
        if announcement not in announcements:
            announcements.append(announcement)
    return announcements


_MOTION_TITLE = re.compile(
    r"^(\s*(?:(?:[-+*]|\d+\.)\s+)?)(?:motion to adjourn(?: the meeting)?|"
    r"motion for adjournment|adjournment(?: motion)?)(?=\W|$)", re.IGNORECASE,
)
_OUTCOME = r"(?:not carried|carried(?: unanimously| without objection)?|defeated|withdrawn|tabled|passed|approved|adopted|accepted|ratified)"
_CLOSE = r"(?:the |this )?meeting (?:(?:is|was|has been|formally|officially)\s+)*adjourned"
_OUTCOME_LINE = re.compile(
    r"^\s*(?:outcome\s*[:–—-]\s*)?(?:(?:(?:the |that )?motion (?:is |was |has been )?)?" + _OUTCOME +
    r"(?:[.;,–—-]\s*(?:and\s+)?" + _CLOSE + r")?|" + _CLOSE + r")[.!]?\s*$",
    re.IGNORECASE,
)
_ADJOURNMENT_REFERENCE = r"(?:motion (?:to adjourn(?: the meeting)?|for adjournment)|adjournment motion)"
_ASSERTED_OUTCOME = r"(?:not carried|carried|passed|approved|adopted|accepted|ratified|defeated|withdrawn|tabled)"
_EXCEPTION_OUTCOME = re.compile(
    r"\bNo (?P<formal>formal )?motions (?:were|have been|had been) " + _ASSERTED_OUTCOME +
    r"(?: unanimously)? (?:other than|except(?: for)?|apart from) (?:the )?(?P<motion>" + _ADJOURNMENT_REFERENCE + r")\b",
    re.IGNORECASE,
)
_ONLY_OUTCOME = re.compile(
    r"\bThe only (?P<formal>formal )?motion (?:(?:that|which) )?(?:was )?" + _ASSERTED_OUTCOME +
    r" was (?:the )?(?P<motion>" + _ADJOURNMENT_REFERENCE + r")\b", re.IGNORECASE,
)
_DIRECT_OUTCOME = re.compile(
    r"\b(?P<motion>(?:the |this |a )?" + _ADJOURNMENT_REFERENCE + r")\s+"
    r"(?:(?:was|is|has been|had been)\s+)?(?:unanimously\s+)?" + _ASSERTED_OUTCOME +
    r"(?:\s+(?:unanimously|without objection))?\b", re.IGNORECASE,
)
_OUTCOME_ADJECTIVE = re.compile(
    r"\b(?P<article>the|this|a) " + _ASSERTED_OUTCOME + r" (?P<motion>" + _ADJOURNMENT_REFERENCE + r")\b",
    re.IGNORECASE,
)
_ACTIVE_OUTCOME = re.compile(
    r"\b(?:(?:the )?(?:members|participants|committee|chair)|we|they|"
    r"(?-i:[A-Z][\w'’.-]*(?:\s+(?:and\s+)?[A-Z][\w'’.-]*){0,3}))\s+"
    r"(?:unanimously\s+)?" + _ASSERTED_OUTCOME + r" (?:the|a|this) " +
    r"(?P<motion>" + _ADJOURNMENT_REFERENCE + r")\b", re.IGNORECASE,
)


def _neutral_adjournment_sentence(line: str) -> str:
    # Match outcome grammar tied to this motion, never unrelated uses of a
    # voting verb. Keep hypothetical discussion and other sentence text intact.
    if re.search(r"\b(?:if|whether|would|could|might|hypothetically)\b", line, re.IGNORECASE):
        return line
    def recorded_only(match: re.Match) -> str:
        return "The only " + (match["formal"] or "") + "motion recorded was the " + match["motion"]
    line = _EXCEPTION_OUTCOME.sub(recorded_only, line)
    line = _ONLY_OUTCOME.sub(recorded_only, line)
    line = _DIRECT_OUTCOME.sub(lambda match: match["motion"] + " was recorded", line)
    line = _OUTCOME_ADJECTIVE.sub(lambda match: match["article"] + " " + match["motion"], line)
    return _ACTIVE_OUTCOME.sub(lambda match: "The " + match["motion"] + " was recorded", line)


def _neutral_adjournment_prose(line: str) -> str:
    return "".join(_neutral_adjournment_sentence(part) for part in re.split(r"(?<=[.!?])(\s+)", line))


def _explicit_closing(source: str) -> bool:
    for line in source.splitlines():
        body = re.sub(r"^\[[^\]]+\]\s*", "", line)
        if re.match(r"^(?:(?:okay|ok|so|now)[,.]?\s+)?(?:" + _CLOSE + r"|we (?:are|stand) adjourned|we['’]re adjourned|this concludes the meeting)[.!]?$", body, re.IGNORECASE):
            return True
    return False


def _neutral_closing(line: str) -> str:
    def sentence(part):
        if re.search(r"\b(?:if|whether|would|could|might|hypothetically)\b", part, re.IGNORECASE):
            return part
        part = re.sub(r"\b" + _CLOSE + r"(?: on (?:a )?motion)?\b", "An adjournment motion was recorded", part, flags=re.IGNORECASE)
        if re.search(r"\b(?:meeting|session)\b", part, re.IGNORECASE) and re.search(r"\bbefore adjourning\b", part, re.IGNORECASE):
            part = re.sub(r"\bclosed with\b", "included", part, flags=re.IGNORECASE)
            part = re.sub(r"\s+before adjourning\b", "", part, flags=re.IGNORECASE)
        return part
    return "".join(sentence(part) for part in re.split(r"(?<=[.!?])(\s+)", line))


def correct_adjournment_roles(content: str, source: str) -> str:
    announcements = adjournment_announcements(source)
    if len(announcements) != 1:
        return content  # Conflicting/multiple records require review, not guessing.
    lines = []
    historical = False
    in_motion = False
    closed = _explicit_closing(source)
    for line in content.splitlines():
        if re.match(r"^##\s+", line):
            historical = "recap of previous meeting" in line.casefold()
        if historical or "recap of previous meeting:" in line.casefold():
            lines.append(line)
            in_motion = False
            continue
        plain = line.replace("**", "").replace("__", "")
        match = _MOTION_TITLE.match(plain)
        if match:
            lines.append(match[1] + announcements[0].render())
            in_motion = True
        elif in_motion and (
            re.match(r"^\s*(?:mover|seconder|moved by|seconded by|outcome)\b", plain, re.IGNORECASE)
            or _OUTCOME_LINE.fullmatch(plain)
        ):
            continue
        elif in_motion and not line.strip():
            lines.append(line)
        else:
            neutral = _neutral_adjournment_prose(line) if announcements[0].outcome is None else line
            lines.append(neutral if closed else _neutral_closing(neutral))
            in_motion = False
    return "\n".join(lines)
