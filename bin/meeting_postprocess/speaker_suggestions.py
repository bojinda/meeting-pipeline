"""Per-meeting text evidence for advisory identity review, never voice matching."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from .normalization import SPEAKER_LABEL, normalize_text
from .redaction import redact_chunks

PRIVATE_SUGGESTIONS_FILENAME = "speaker-suggestions.json"
PRIVATE_ROSTER_FILENAME = "speaker_roster.private.json"
NAME = r"(?-i:[A-ZÀ-ÖØ-Þ][\w'’-]*(?:\s+[A-ZÀ-ÖØ-Þ][\w'’-]*){0,3})"
# Parse discourse before looking for a vocative. This removes courtesy and
# conjunction syntax only at sentence openings, rather than blacklisting names.
_DISCOURSE = re.compile(r"^(?:(?:okay|ok|yeah|yes|so|well|uh|um|oh|hey|and|but|however|pardon(?: me)?|excuse me|sorry)[,.!?]?\s+)+", re.IGNORECASE)
_SELF = r"^(?:my name is|i['’]m|i am|that['’]s me[, :]*|that is me[, :]*|this is)\s+(?P<name>NAME)(?=\s*(?:[,.!?;]|$)|\s+(?:speaking|here|from)\b)"
_INTRO = r"^(?:i['’]d like to introduce|let me introduce|next (?:we have|up is)|joining us(?: today)? is|i['’]ll hand over to|let['’]s hear from|we['’]ll hear from)\s+(?P<name>NAME)(?=\s*(?:[,.!?;]|$)|\s+(?:speaking|here|from|who|next)\b)"
_PERSONAL_QUESTION = r"i(?: have (?:just )?got|['’]ve (?:just )?got| have) (?:a quick |a |another )?question for you"
_ADDRESS = r"^(?P<name>NAME)(?:\s*\?\s*$|[, :]\s*(?:are you|can you|could you|would you|will you|do you|have you|please|your report|" + _PERSONAL_QUESTION + r"|what happened|how did|what do you|did you)\b)"
# Consume only a grammatical question-closing preface, never search for names
# inside arbitrary prose. The name after its comma is the new addressee.
_PREFIXED_QUESTION = r"^if there(?: (?:are no|aren['’]t any)|['’]s no) (?:further |more |other )?questions(?: for NAME)?\s*,\s*(?:then\s+)?(?P<name>NAME)\s*,\s*" + _PERSONAL_QUESTION + r"\b"
_BROAD_INVITE = r"^(?:we would like|we['’]d like|i would like|we are waiting for)\s+(?P<name>NAME)\s+to\s+(?:speak|report|go next|present)\b"
_RESPONSE = re.compile(r"^(?:yes|yeah|sure|okay|ok|agreed|thanks|thank you|hello|hi|good morning|i can|i will|i['’]ll|i['’]m here|that['’]s me)\b", re.IGNORECASE)
_BROAD_RESPONSE = re.compile(r"^(?:i\b|i['’]|we\b|we['’]|the (?:claim|report|issue|grievance)\b)", re.IGNORECASE)
_DENIAL = re.compile(r"\b(?:not me|not here|isn['’]t here|i am not|i['’]m not|no[,!.])", re.IGNORECASE)
_FRAGMENT = re.compile(r"^(?:that['’]s me|that is me)[.!]?$", re.IGNORECASE)
_IDENTITY_QUESTION = re.compile(r"^(?:who(?:['’]s| is) (?:that|speaking)|what(?:['’]s| is) your name)\s*\?", re.IGNORECASE)
_NON_NAMES = {"the", "a", "i", "we", "you", "they", "yes", "no", "okay", "good", "really", "tired", "ready", "sorry", "going", "gonna", "here", "back", "available", "chair", "president", "member", "engineer", "conductor", "unknown"}


def _plain_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Speaker names must be nonempty strings")
    name = normalize_text(value.strip())
    if SPEAKER_LABEL.search(name) or any(char in name for char in "\r\n[]#*`<>"):
        raise ValueError("Speaker names must be plain, single-line names")
    return name


def load_roster(transcript_dir: Path, explicit_path: Path | None = None) -> list[dict]:
    path = explicit_path or transcript_dir / PRIVATE_ROSTER_FILENAME
    if explicit_path is None and not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    people = data.get("people") if isinstance(data, dict) else data
    if not isinstance(people, list):
        raise ValueError("Private roster must contain a people list")
    roster = []
    for person in people:
        person = {"name": person} if isinstance(person, str) else person
        if not isinstance(person, dict):
            raise ValueError("Invalid roster person")
        aliases, role = person.get("aliases", []), person.get("role", "")
        if not isinstance(aliases, list) or not isinstance(role, str) or any(char in role for char in "\r\n"):
            raise ValueError("Invalid roster aliases or role")
        roster.append({"name": _plain_name(person.get("name")), "aliases": [_plain_name(alias) for alias in aliases], "role": role.strip()})
    return roster


def source_digest(chunks: list[dict]) -> str:
    visible = speaker_input(chunks)
    text = json.dumps([(chunk["chunk_id"], chunk["text"], chunk["speaker_review_gap"]) for chunk in visible], ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def speaker_input(chunks: list[dict]) -> list[dict]:
    """The ingestion boundary: downstream review receives only redacted text."""
    inputs = [dict(chunk, chunk_id=chunk.get("chunk_id", index + 1)) for index, chunk in enumerate(chunks)]
    redacted = redact_chunks(inputs)
    ids, blocked = [str(chunk["chunk_id"]) for chunk in inputs], set()
    for record in redacted.redactions:
        start, end = ids.index(str(record["start_chunk_id"])), ids.index(str(record["end_chunk_id"]))
        blocked.update(ids[start:end + 1])
    prior_gaps = {str(chunk["chunk_id"]) for chunk in inputs if chunk.get("speaker_review_gap") is True}
    # An entirely redacted chunk disappears from the visible transcript. Carry
    # its boundary forward so an address and response cannot pair across it.
    visible_ids = {str(chunk["chunk_id"]) for chunk in redacted.chunks}
    pending_gap = False
    for identifier in ids:
        if identifier not in visible_ids:
            pending_gap = pending_gap or identifier in blocked | prior_gaps
        elif pending_gap:
            blocked.add(identifier)
            pending_gap = False
    return [dict(chunk, speaker_review_gap=str(chunk["chunk_id"]) in blocked | prior_gaps) for chunk in redacted.chunks]


def safe_groups(chunks: list[dict]) -> list[dict]:
    groups = []
    for chunk in speaker_input(chunks):
        for line in chunk["text"].splitlines():
            match = re.match(r"^\[([^\]]+)\]\s*(.+)", line)
            if not match:
                continue
            speaker, text = match.groups()
            safe = not chunk["speaker_review_gap"]
            if groups and safe and groups[-1]["safe_pair"] and groups[-1]["speaker_label"] == speaker:
                groups[-1]["lines"].append(text)
                groups[-1]["text"] += "\n" + text
            else:
                position = len(groups)
                groups.append({"id": f"E{position:06d}", "position": position, "speaker_label": speaker, "text": text, "lines": [text], "safe_pair": safe})
    return groups


def _sentences(text: str) -> list[str]:
    return [_DISCOURSE.sub("", part.strip()) for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _patterns(known: list[str]) -> tuple:
    name = "(?:" + "|".join(re.escape(item) for item in sorted(set(known), key=len, reverse=True)) + "|" + NAME + ")" if known else NAME
    return tuple(re.compile(pattern.replace("NAME", name), re.IGNORECASE) for pattern in (_SELF, _INTRO, _ADDRESS, _BROAD_INVITE, r"^(?P<name>NAME)[.!]?$", _PREFIXED_QUESTION))


def grounding_events(groups: list[dict], approved: dict, roster: list[dict], broad: bool = False) -> list[dict]:
    known = list(approved.values())
    for person in roster:
        known.extend([person["name"], person["name"].split()[0], *person["aliases"]])
    patterns = _patterns(known)
    attested = []
    prior_identities = {}
    for group in groups:
        found = []
        for sentence in _sentences(group["text"]):
            for pattern in patterns[:2]:
                match = pattern.search(sentence)
                if match:
                    if pattern is patterns[0] and re.match(r"this is\b", match[0], re.IGNORECASE) and not re.match(r"\s+(?:speaking|here)\b", sentence[match.end():], re.IGNORECASE):
                        continue
                    found.append(match["name"].strip(" ."))
        if group["safe_pair"]:
            for first, second in zip(group["lines"], group["lines"][1:]):
                match = patterns[4].fullmatch(second.strip())
                if _FRAGMENT.fullmatch(_DISCOURSE.sub("", first.strip())) and match:
                    found.append(match["name"].strip(" ."))
        prior_identities[group["position"]] = found
        attested.extend(found)
    patterns = _patterns(known + attested)

    def resolve(spoken: str) -> list[str]:
        name = spoken.strip(" .")
        if not name or SPEAKER_LABEL.search(name) or any(word.casefold() in _NON_NAMES for word in name.split()):
            return []
        for pool in (roster, [{"name": item, "aliases": []} for item in [*approved.values(), *attested]]):
            hits = [person["name"] for person in pool if name.casefold() in {item.casefold() for item in [person["name"], person["name"].split()[0], *person.get("aliases", [])]}]
            if hits:
                return list(dict.fromkeys(hits))
        return [name]

    events = []
    def add(group: dict, spoken: str, kind: str, anchors: dict[str, str], strength: str, uncertain: bool = False) -> None:
        names = resolve(spoken)
        sentences = _sentences(group["text"])
        uncertain = uncertain or bool(re.search(r"\[SPEAKER_\d+\]", group["text"])) or len(sentences) > 1 and any(_IDENTITY_QUESTION.match(sentence) for sentence in sentences)
        if names and SPEAKER_LABEL.fullmatch(group["speaker_label"]) and group["speaker_label"] not in approved:
            events.append({"speaker_label": group["speaker_label"], "candidate_names": names, "type": kind, "evidence_ids": list(anchors), "anchors": anchors, "confidence": strength, "origin": "heuristic", "uncertain_source_attribution": uncertain})

    for index, group in enumerate(groups):
        for sentence in _sentences(group["text"]):
            match = patterns[0].search(sentence)
            if not match:
                continue
            if re.match(r"this is\b", match[0], re.IGNORECASE) and not re.match(r"\s+(?:speaking|here)\b", sentence[match.end():], re.IGNORECASE):
                continue
            add(group, match["name"], "self_identification", {group["id"]: match[0]}, "high")
        if group["safe_pair"]:
            for first, second in zip(group["lines"], group["lines"][1:]):
                name = patterns[4].fullmatch(second.strip())
                if _FRAGMENT.fullmatch(_DISCOURSE.sub("", first.strip())) and name:
                    add(group, name["name"], "fragmented_self_identification", {group["id"]: first + "\n" + second}, "high")
            # ASR may merge an exchange and a fragmented identity into one line.
            # Retain the possible identity, but the label's attribution needs review.
            for line in group["lines"]:
                sentences = re.split(r"(?<=[.!?])\s+", line)
                for first, second in zip(sentences, sentences[1:]):
                    name = patterns[4].fullmatch(second.strip())
                    if _FRAGMENT.fullmatch(_DISCOURSE.sub("", first.strip())) and name:
                        add(group, name["name"], "fragmented_self_identification", {group["id"]: first + " " + second}, "low", uncertain=True)
        if index + 1 >= len(groups):
            continue
        response = groups[index + 1]
        if not group["safe_pair"] or not response["safe_pair"] or group["speaker_label"] == response["speaker_label"] or _DENIAL.search(response["text"]):
            continue
        reply = response["text"].lstrip()
        if not (_RESPONSE.search(reply) or (broad and _BROAD_RESPONSE.search(reply))):
            continue
        cues = []
        for sentence in _sentences(group["text"]):
            for pattern, kind in ((patterns[1], "introduction"), (patterns[2], "direct_address_response"), (patterns[5], "direct_address_response"), *(([(patterns[3], "invited_speaker")] if broad else []))):
                match = pattern.search(sentence)
                if match:
                    if kind == "direct_address_response" and re.fullmatch(r"\s*\?\s*", sentence[match.end("name"):]):
                        independent = known + [name for position, names in prior_identities.items() if position < group["position"] for name in names]
                        if match["name"].casefold() not in {form.casefold() for name in independent for form in (name, name.split()[0])}:
                            continue
                    cues.append((match, kind))
        if cues:
            match, kind = cues[-1]
            add(response, match["name"], kind, {group["id"]: match[0], response["id"]: reply[:80]}, "medium" if kind != "direct_address_response" or not _RESPONSE.search(reply) else "low")
    return events


def summarize_events(groups: list[dict], events: list[dict], approved: dict) -> list[dict]:
    result = []
    labels = sorted({group["speaker_label"] for group in groups if SPEAKER_LABEL.fullmatch(group["speaker_label"]) and group["speaker_label"] not in approved})
    for label in labels:
        items = [event for event in events if event["speaker_label"] == label]
        names = list(dict.fromkeys(name for item in items for name in item["candidate_names"]))
        conflicts = []
        if any(item.get("uncertain_source_attribution") for item in items):
            conflicts.append("Identity occurs inside a merged ASR exchange; source attribution requires explicit manual review")
        if len(names) > 1:
            conflicts.append("Conflicting identities on this diarization label; explicit manual review required")
        if any(name.casefold() in {value.casefold() for value in approved.values()} for name in names):
            conflicts.append("Candidate already has an approved alias on another label")
        ranks = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
        confidence = max((item["confidence"] for item in items), key=ranks.get, default="unknown")
        independent = {tuple(item["evidence_ids"]) for item in items if item["type"] != "role_context"}
        if confidence == "low" and len(independent) >= 2:
            confidence = "medium"
        origins = {item["origin"] for item in items}
        retained = []
        for name in names:
            supporting = [item for item in items if name in item["candidate_names"]]
            best = max(supporting, key=lambda item: (item["type"] != "role_context", ranks[item["confidence"]]))
            if best not in retained:
                retained.append(best)
        retained.extend(item for item in items if item not in retained)
        result.append({"speaker_label": label, "suggested_name": names[0] if len(names) == 1 and not conflicts else None, "confidence": confidence if not conflicts else "unknown", "evidence_types": sorted({item["type"] for item in items}), "evidence": [{"type": item["type"], "candidate_names": item["candidate_names"], "evidence_ids": item["evidence_ids"], "excerpts": [text[:240] for text in item["anchors"].values()], "origin": item["origin"], "uncertain_source_attribution": item.get("uncertain_source_attribution", False)} for item in retained[:16]], "candidates": names, "conflicting_candidates": names if conflicts else [], "ambiguity": conflicts, "origin": "both" if len(origins) > 1 else next(iter(origins), "heuristic"), "requires_manual_review": True})
    for row in result:
        if row["suggested_name"] and any(row["suggested_name"] in other["candidates"] for other in result if other is not row):
            row["ambiguity"].append("Candidate appears on another diarization label; verify manually")
            row["conflicting_candidates"] = row["candidates"]
            row["suggested_name"], row["confidence"] = None, "unknown"
    return result


def build_speaker_suggestions(chunks: list[dict], approved: dict[str, str] | None = None, roster: list[dict] | None = None, meeting_id: str | None = None) -> dict:
    approved, roster = approved or {}, roster or []
    groups = safe_groups(chunks)
    events = grounding_events(groups, approved, roster)
    broad = grounding_events(groups, approved, roster, broad=True)
    for event in broad:
        other = {name for item in events if item["speaker_label"] == event["speaker_label"] for name in item["candidate_names"]}
        if other and set(event["candidate_names"]) - other and event not in events:
            events.append(event)
    for person in roster:
        if not person.get("role"):
            continue
        for group in groups:
            role = re.search(r"\b" + re.escape(person["role"]) + r"\b", group["text"], re.IGNORECASE)
            if role and any(item["speaker_label"] == group["speaker_label"] and person["name"] in item["candidate_names"] for item in events):
                start = max(0, role.start() - 60)
                events.append({"speaker_label": group["speaker_label"], "candidate_names": [person["name"]], "type": "role_context", "evidence_ids": [group["id"]], "anchors": {group["id"]: group["text"][start:start + 200]}, "confidence": "low", "origin": "heuristic"})
    return {"meeting_id": meeting_id, "source_digest": source_digest(chunks), "advisory_only": True, "method": "per_meeting_text_evidence", "suggestions": summarize_events(groups, events, approved)}


def write_private_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise OSError("Private speaker output cannot be a symbolic link")
    descriptor, filename = tempfile.mkstemp(prefix=".private-speaker-", suffix=".tmp", dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
