"""Source-based guards against completed outreach and uncommitted proposals."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class CompletedRequest:
    recipients: list[str]
    topic: str


@dataclass
class TentativeProposal:
    topic: str
    speaker: str
    addressee: str | None
    source_line: int


_TENTATIVE = re.compile(
    r"\b(?:i think (?:we|i) (?:should|have to|need to)|"
    r"we (?:probably|possibly) (?:need|have) to|we should(?: consider)?)\s+(?P<topic>.+)",
    re.IGNORECASE,
)
_SELF_COMMITMENT = re.compile(r"\b(?:i['’]ll|i will|i agree to|i volunteer to|i commit to)\b", re.IGNORECASE)
_ACCEPTANCE = re.compile(r"^(?:yes|sure|agreed|okay|ok|i agree|i['’]ll do (?:it|that)|i will do (?:it|that))(?:[,.!]|$)", re.IGNORECASE)
_DECLINE = re.compile(r"\b(?:will not|won['’]t|do not|don['’]t|decline|cannot|can['’]t|disagree)\b", re.IGNORECASE)
_TASK_FILLER = {"i", "we", "think", "should", "have", "need", "probably", "possibly", "another", "with", "mr", "ms", "mrs", "dr", "that"}


def _source_turns(source: str) -> list[tuple[str, str]]:
    turns = []
    for line in source.splitlines():
        match = re.match(r"^\[([^\]]+)\]\s*(.*)", line)
        turns.append((match[1], match[2]) if match else ("", line))
    return turns


def _evidence_text(turns: list[tuple[str, str]], index: int) -> str:
    speaker, text = turns[index]
    # The section splitter may separate an honorific from its surname. Join
    # this continuation only for evidence lookup; retain original source text.
    if re.search(r"\b(?:Mr|Ms|Mrs|Dr)\.$", text) and index + 1 < len(turns) and turns[index + 1][0] == speaker:
        text += " " + turns[index + 1][1]
    return text


def tentative_proposals(source: str) -> list[TentativeProposal]:
    turns = _source_turns(source)
    proposals = []
    for index, (speaker, text) in enumerate(turns):
        text = _evidence_text(turns, index)
        match = _TENTATIVE.search(text)
        if match:
            addressee = text[:match.start()].strip(" ,:.").split(",")[-1].strip()
            if not re.fullmatch(r"[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,3}", addressee):
                addressee = None
            proposals.append(TentativeProposal(match["topic"], speaker, addressee, index))
    return proposals


def _same_task(topic: str, candidate: str) -> bool:
    words = _topic_words(topic) - _TASK_FILLER
    candidate_words = _topic_words(candidate) - _TASK_FILLER
    targets = re.findall(r"\b(?:Mr|Ms|Mrs|Dr)\.?\s+([A-Z][\w'’.-]*)", topic)
    if any(target.casefold().rstrip("s.") not in candidate_words for target in targets):
        return False
    return bool(words) and len(words & candidate_words) >= min(2, len(words))


def _named_commitment(text: str, name: str) -> bool:
    # Only a contiguous subject (possibly coordinated) may supply the owner.
    # A name elsewhere in the sentence cannot borrow another person's verb.
    person = r"(?-i:[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,3})"
    coordinated = r"(?:,\s*(?:and\s+)?|\s+(?:and|&)\s+)" + person
    return bool(re.search(
        r"(?<!\w)" + re.escape(name) + r"(?!\w)(?:" + coordinated + r")*,?\s+"
        r"(?:will|must|agreed to|has agreed to|is assigned to|is responsible for|please)\b",
        text, re.IGNORECASE,
    ))


def _proposal_supported(source: str, proposal: TentativeProposal, owner: str | None, owner_aliases: tuple[str, ...] = ()) -> bool:
    turns = _source_turns(source)
    for index, (speaker, _) in enumerate(turns):
        text = _evidence_text(turns, index)
        if _DECLINE.search(text) or _TENTATIVE.search(text):
            continue
        same_task = _same_task(proposal.topic, text)
        # A nearby explicit reply can refer back to the proposed task with
        # "it" or "that", while still repeating its action verb.
        commitment = _SELF_COMMITMENT.search(text)
        action = re.match(r"\s+(?:try to\s+)?([a-z]+)\b", text[commitment.end():], re.IGNORECASE) if commitment else None
        proposed_action = re.match(r"([a-z]+)\b", proposal.topic, re.IGNORECASE)
        referring_reply = (
            proposal.source_line < index <= proposal.source_line + 3
            and re.search(r"\b(?:it|that)\b", text, re.IGNORECASE)
            and action and proposed_action and action[1].casefold() == proposed_action[1].casefold()
            and not any(_TENTATIVE.search(turn[1]) for turn in turns[proposal.source_line + 1:index])
        )
        if not same_task:
            if referring_reply and owner and speaker.casefold() == owner.casefold():
                return True
            continue
        if owner is None and re.search(r"\b(?:we['’]ll|we will|scheduled|agreed to)\b", text, re.IGNORECASE) and not _TENTATIVE.search(text):
            return True
        if (owner is None or speaker.casefold() == owner.casefold()) and _SELF_COMMITMENT.search(text):
            return True
        if owner:
            name = re.escape(owner)
            named = _named_commitment(text, owner)
            assigned = re.search(r"\b(?:assign|assigned|ask)\s+" + name + r"\s+to\b", text, re.IGNORECASE)
            directive = re.search(r"(?<!\w)" + name + r",\s+(?:(?:please\s+)?(?:request|contact|arrange|send|schedule)\b|i (?:want|am asking) you to\b)", text, re.IGNORECASE)
            if named or assigned or directive:
                return True
    # Scan the bounded reply window for this owner's own short acceptance.
    # Other participants' replies cannot supply support or stop the search.
    for speaker, text in turns[proposal.source_line + 1:proposal.source_line + 4]:
        if speaker == proposal.speaker:
            continue
        if owner is not None and speaker.casefold() != owner.casefold():
            continue
        if owner is not None and proposal.addressee and proposal.addressee.casefold() not in {name.casefold() for name in (owner, *owner_aliases)}:
            break
        if _DECLINE.search(text):
            return False
        if _ACCEPTANCE.search(text) and not _TENTATIVE.search(text):
            return True
    return False


_OUTREACH = re.compile(
    r"\b(?:messaged|contacted|emailed|e-mailed|texted|asked|requested (?:information|details) from)\s+"
    r"(?P<recipients>[^.!?\n]+?)\s+(?:for|about|regarding|to)\s+(?P<topic>[^.!?\n]+)",
    re.IGNORECASE,
)
_PASSIVE_OUTREACH = re.compile(
    r"(?P<recipients>[A-Z][\w'’., &-]+?)\s+(?:have|had|were|was)\s+(?:already\s+)?(?:been\s+)?"
    r"(?:messaged|contacted|emailed|texted|asked)\s+(?:for|about|regarding|to)\s+(?P<topic>[^.!?\n]+)",
)
_FULFILL = re.compile(r"\b(?:provide|send|submit|supply|confirm|share|respond|give)\b", re.IGNORECASE)
_STOP_WORDS = {"a", "an", "the", "for", "to", "their", "his", "her", "our", "of", "and", "please", "provide", "send", "me", "us"}


def _topic_words(text: str) -> set[str]:
    return {word.rstrip("s") for word in re.findall(r"[a-z]+", text.lower()) if word not in _STOP_WORDS}


def completed_requests(source: str) -> list[CompletedRequest]:
    requests = []
    matches = list(_OUTREACH.finditer(source)) + list(_PASSIVE_OUTREACH.finditer(source))
    for match in matches:
        names = [name.strip() for name in re.split(r",\s*|\s+(?:and|&)\s+", match["recipients"])]
        # Pronouns cannot establish identities, and hypothetical contacts cannot
        # establish completed outreach. Require an explicit past-tense report.
        if not names or not all(re.fullmatch(r"[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,3}", name) for name in names):
            continue
        start = max(source.rfind("\n", 0, match.start()), source.rfind(".", 0, match.start())) + 1
        lead = source[start:match.start()]
        if re.search(r"\b(?:if|unless|not|never|haven['’]t|hadn['’]t)\b", lead, re.IGNORECASE):
            continue
        requests.append(CompletedRequest(names, match["topic"]))
    return requests


def _explicit_assignment(source: str, name: str, topic: str) -> bool:
    for line in source.splitlines():
        speaker = re.match(r"^\[([^\]]+)\]\s*(.*)", line)
        text = speaker[2] if speaker else line
        if _TENTATIVE.search(text) or _DECLINE.search(text):
            continue
        named_commitment = _named_commitment(text, name)
        own_commitment = speaker and speaker[1].casefold() == name.casefold() and re.search(r"\b(?:I['’]ll|I will|I agree to)\b", text, re.IGNORECASE)
        direct_request = re.search(r"\b(?:ask|assign|assigned)\s+" + re.escape(name) + r"\s+to\b", text, re.IGNORECASE)
        if (named_commitment or own_commitment or direct_request) and _topic_words(topic) & _topic_words(text):
            return True
    return False


def _plain_cell(text: str) -> str:
    return text.replace("**", "").replace("__", "").strip(" *_`")


def _owner_groups(owner: str) -> tuple[tuple[str, ...], ...]:
    groups = []
    # Split people outside parenthetical speaker/role annotations. Each
    # person's name and speaking label are alternate source identities.
    parts = re.split(r"(?:,\s*|\s+(?:and|&|/)\s+)(?![^()]*\))", _plain_cell(owner))
    for part in parts:
        name = re.sub(r"\([^)]*\)", "", part).strip(" []")
        identities = [name] if re.fullmatch(r"[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,3}", name) else []
        identities.extend(re.findall(r"\bSPEAKER_\d+\b", part))
        groups.append(tuple(dict.fromkeys(identities)))
    return tuple(groups)


def _remove_action(owner: str, task: str, source: str, requests: list[CompletedRequest], proposals: list[TentativeProposal], future_evidence=()) -> bool:
    owner, task = _plain_cell(owner), _plain_cell(task)
    owners = _owner_groups(owner)
    for request in requests:
        affected = any(name.casefold() == identity.casefold() for name in request.recipients for group in owners for identity in group)
        if affected and _FULFILL.search(task) and _topic_words(request.topic) & _topic_words(task):
            if not all(any(_explicit_assignment(source, identity, request.topic) for identity in group) for group in owners):
                return True
    for proposal in proposals:
        if _same_task(proposal.topic, task):
            # Raising a proposal for discussion is not implementing its outcome.
            # Only source-bound, gap-safe quotations may establish that narrower task.
            def raises_only(identity):
                if not re.match(r"(?:raise|bring)\b", task, re.IGNORECASE) or not re.search(r"\bmeeting\b", task, re.IGNORECASE):
                    return False
                for record in future_evidence:
                    quoted = record.splitlines()
                    own = re.match(r"^\[([^\]]+)\]\s*(.*)$", quoted[-1]) if quoted else None
                    if (own and own[1].casefold() == identity.casefold()
                            and re.search(r"\b(?:i['’]ll|i will) bring (?:that|this|it) up (?:at|in)\b", own[2], re.IGNORECASE)
                            and all(line in source.splitlines() for line in quoted)
                            and _same_task(proposal.topic, record)):
                        return True
                return False
            if all(any(raises_only(identity) for identity in group) for group in owners):
                continue
            if not all(
                any(_proposal_supported(source, proposal, identity, group) for identity in group)
                if group else len(owners) == 1 and _proposal_supported(source, proposal, None)
                for group in owners
            ):
                return True
    return False


_OWNER_COLUMNS = {"owner", "assigned to", "assignee", "responsible", "responsible person", "responsible party"}
_TASK_COLUMNS = {"action", "action item", "action items", "task", "task description", "description"}


def _table_cells(line: str) -> list[str] | None:
    if not re.search(r"(?<!\\)\|", line):
        return None
    parts = re.split(r"(?<!\\)\|", line.strip())
    if parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return [part.strip() for part in parts]


def _table_separator(cells: list[str] | None) -> bool:
    return bool(cells and all(re.fullmatch(r":?-+:?", cell.replace(" ", "")) for cell in cells))


def action_entries(content: str) -> list[tuple[str, str]]:
    """Read action bullets/tables for cross-document QA, without assigning owners."""
    result, columns = [], None
    active, depth, fence = False, 0, False
    for line in content.splitlines():
        if re.match(r"^\s*(?:```|~~~)", line):
            fence = not fence
            continue
        if fence:
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            if "action items" in heading[2].casefold():
                active, depth = True, len(heading[1])
            elif len(heading[1]) <= depth:
                active = False
            columns = None
            continue
        if not active:
            continue
        cells = _table_cells(line)
        if cells and not _table_separator(cells):
            names = [_plain_cell(cell).casefold() for cell in cells]
            owner = next((i for i, name in enumerate(names) if name in _OWNER_COLUMNS), None)
            task = next((i for i, name in enumerate(names) if name in _TASK_COLUMNS), None)
            if owner is not None and task is not None:
                columns = (owner, task)
            elif columns and len(cells) > max(columns):
                result.append((_plain_cell(cells[columns[0]]), _plain_cell(cells[columns[1]])))
        elif re.match(r"^\s*(?:[-+*]|\d+\.)\s+", line):
            body = re.sub(r"^\s*(?:[-+*]|\d+\.)\s+", "", _plain_cell(line))
            split = re.split(r"\s+[–—-]\s+|[:;]\s*", body, maxsplit=1)
            if len(split) == 2:
                result.append(tuple(split))
    return result


def filter_completed_request_tasks(content: str, source: str, action_sections_only: bool = False, *, future_evidence=()) -> str:
    requests, proposals = completed_requests(source), tentative_proposals(source)
    if not requests and not proposals:
        return content
    original = content.splitlines()
    lines: list[str] = []
    in_actions = not action_sections_only
    section_start = section_depth = None
    section_kept = section_table_removed = 0
    removed_any = table_removed = saw_action_heading = False
    fence = None

    def finish_section() -> None:
        nonlocal section_start, section_depth, section_kept, section_table_removed
        if section_start is not None and section_table_removed and not section_kept:
            lines[section_start:] = ["None noted.", ""]
        section_start = section_depth = None
        section_kept = section_table_removed = 0

    index = 0
    while index < len(original):
        line = original[index]
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker[1][0]
                if in_actions:
                    section_kept += 1  # Preserve literal examples, not action candidates.
            elif marker[1][0] == fence:
                fence = None
            lines.append(line)
            index += 1
            continue
        if fence is not None:
            lines.append(line)
            index += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading:
            depth = len(heading[1])
            if section_depth is not None and depth <= section_depth:
                finish_section()
                in_actions = False
            if "action items" in heading[2].casefold():
                finish_section()
                in_actions = saw_action_heading = True
                lines.append(line)
                section_start, section_depth = len(lines), depth
                index += 1
                continue
            if section_depth is None and (action_sections_only or saw_action_heading):
                in_actions = False
        header = _table_cells(line) if in_actions else None
        separator = _table_cells(original[index + 1]) if header and index + 1 < len(original) else None
        if header and _table_separator(separator) and len(header) == len(separator):
            names = [re.sub(r"\s+", " ", _plain_cell(cell).casefold()) for cell in header]
            owner_column = next((i for i, name in enumerate(names) if name in _OWNER_COLUMNS), None)
            task_column = next((i for i, name in enumerate(names) if name in _TASK_COLUMNS), None)
            rows = []
            kept = removed = 0
            end = index + 2
            while end < len(original) and (cells := _table_cells(original[end])) is not None:
                if owner_column is not None and task_column is not None and len(cells) == len(header):
                    if _remove_action(cells[owner_column], cells[task_column], source, requests, proposals, future_evidence):
                        removed += 1
                        end += 1
                        continue
                    if cells[owner_column] or cells[task_column]:
                        kept += 1
                else:
                    kept += 1  # Unknown layouts are preserved, never guessed.
                rows.append(original[end])
                end += 1
            if not removed or kept:
                lines.extend(original[index:index + 2] + rows)
            removed_any = removed_any or bool(removed)
            table_removed = table_removed or bool(removed)
            section_table_removed += removed
            section_kept += kept
            index = end
            continue
        if in_actions and re.match(r"^\s*(?:[-+*]|\d+\.)\s+", line):
            body = re.sub(r"^\s*(?:[-+*]|\d+\.)\s+", "", _plain_cell(line))
            body = re.sub(r"^Owner:\s*", "", body, flags=re.IGNORECASE)
            candidate = re.split(r"\s+[–—-]\s+|[:;]\s*", body, maxsplit=1)
            owner, task = candidate[0], candidate[-1]
            if _remove_action(owner, task, source, requests, proposals, future_evidence):
                removed_any = True
                index += 1
                continue
            section_kept += 1
        lines.append(line)
        index += 1
    finish_section()
    if not removed_any:
        return content
    result = "\n".join(lines)
    if action_sections_only or table_removed:
        result = re.sub(
            r"(^#{1,6}\s+Action [Ii]tems[^\n]*)(?:\n[ \t]*)*(?=\n#{1,6}\s|\Z)",
            r"\1\nNone noted.\n", result, flags=re.MULTILINE,
        )
        if table_removed and not any(line.strip() and not line.startswith("#") for line in result.splitlines()):
            result = result.rstrip() + ("\n" if result.strip() else "") + "None noted."
    elif not any(line.strip() and not line.startswith("#") for line in lines):
        result = "# Action Items\nNo clear action items identified."
    return result
