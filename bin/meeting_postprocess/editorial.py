"""Opt-in editorial reduction: one private register, shared public presentations.

This is part of map/reduce, not an additional extraction pass. All results are
operator-review drafts. Reference checks do not certify paraphrase accuracy.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

from .actions import _explicit_assignment, _named_commitment, _owner_groups, filter_completed_request_tasks
from .commitments import _FUTURE, _QUOTED_FRAME, _THIRD_PARTY_STEP
from .motions import correct_adjournment_roles
from .normalization import clean_speaker_annotations, prepare_text, SPEAKER_LABEL
from .publication import strip_private_references
from .qa import check_minutes, write_report
from .rendering import insert_recap, strip_chunk_references
from .sections import BUSINESS, ADJOURNMENT, RECAP
from .speaker_suggestions import write_private_json


REGISTER = "action-register.private.json"
REGISTER_MD = "action-register.private.md"
REVIEW = "editorial-review.private.json"
CHECKLIST = "operator-review.private.md"
RESPONSE = "editorial-response.private.json"
NOTES_EVIDENCE = "notes-evidence.private.json"
BUDGETS = "editorial-budgets.private.json"
FRAMING = 1024
OUTPUT_TOKENS = {"register": 16384, "notes": 8192, "detailed": 16384, "recap": 8192}
LOCAL_LINES = 3
LOCAL_CHARS = 1800
CATEGORIES = {
    "undertaking": "Explicit undertakings and conditional requests",
    "ongoing": "Existing casework in progress",
    "proposal": "Tentative proposals, unassigned plans and unresolved matters",
    "external": "Reported external undertakings and decisions",
    "completed": "Completed work and in-session dispositions",
    "business": "Recorded meeting business, not assignments",
}
STATUS = {"undertaking": "Recorded undertaking; completion unconfirmed",
          "ongoing": "Reported in progress", "proposal": "Tentative / unassigned",
          "external": "Reported outside undertaking or decision",
          "completed": "Reported completed", "business": "Recorded business"}
CONCERNS = {"source_conflict", "uncertain_identity", "confidentiality", "policy_sensitive", "incomplete_coverage"}
_SENSITIVE = re.compile(
    r"[\w.+-]+@[\w.-]+\.[a-z]{2,}|(?:\+?1[ -]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}|"
    r"\b(?:diagnos\w*|surgery|medical history|cancer|medication|home address|"
    r"pay ?stub|employee (?:number|id)|\d+ (?:demerit|discipline) points)\b", re.I)
_IDENTIFYING_CASE = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+){0,2} (?:received|was assessed|was given|was disciplined|was diagnosed)\b")
_ACTION_CUE = re.compile(r"\b(?:I['’]ll|I will|I['’]m (?:going to|gonna)|"
                         r"already|sent|completed|grieving|grievance|should|propos\w*|"
                         r"agreed|promised|undertook|motion|seconded|review tomorrow)\b", re.I)
_STOP = set("a an the to of and or for with at in on by i i'll will am i'm going gonna we our my it that this be is are was were have has been as from".split())


class EditorialFailure(ValueError):
    """Safe error code only; never attach private response text to an exception."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class RequestBudget:
    """Local tokenizer only; never fall back to byte estimates or resize context."""
    def __init__(self, tokenizer, context, system):
        if not tokenizer or type(context) is not int or context <= 0:
            raise EditorialFailure("editorial_tokenizer_and_context_required")
        try:
            from .chunking import counter
            self.count = counter(tokenizer)
            self.tokenizer_hash = hashlib.sha256(Path(tokenizer).read_bytes()).hexdigest()
        except Exception:
            raise EditorialFailure("editorial_tokenizer_unavailable_or_invalid") from None
        self.context, self.system = context, system
        self.measurements = []

    def measure(self, stage, prompt, structured=False):
        system = self.system + ("\nReturn only the requested JSON; no Markdown wrapper." if structured else "")
        # Count complete text, with saved tokenizer truncation/padding disabled.
        tokens = self.count(system + "\n\n" + prompt)
        required = tokens + FRAMING + OUTPUT_TOKENS[stage]
        row = {"stage": stage, "method": self.count.method, "tokenizer_sha256": self.tokenizer_hash,
               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "input_tokens": tokens,
               "reserved_output": OUTPUT_TOKENS[stage], "reserved_framing": FRAMING,
               "configured_context": self.context, "required_context": required,
               "headroom": self.context - required, "fits": required <= self.context}
        self.measurements.append(row)
        return row


def local_ranges(quotes):
    """Small exact consecutive line ranges; no union across unrelated passages."""
    for start in range(len(quotes)):
        rows, size = [], 0
        for row in quotes[start:start + LOCAL_LINES]:
            if rows:
                last = rows[-1]
                if row["chunk_id"] != last["chunk_id"] or int(row["id"].rsplit(":L", 1)[1]) != int(last["id"].rsplit(":L", 1)[1]) + 1:
                    break
            size += len(row["text"])
            if size > LOCAL_CHARS:
                break
            rows.append(row)
            yield rows[:]


def local_support(text, quotes):
    material = _words(text)
    candidates = list(local_ranges(quotes))
    best = max(candidates, key=lambda rows: (len(material & _words(" ".join(r["body"] for r in rows))), -len(rows)), default=[])
    supported = bool(material) and material <= _words(" ".join(r["body"] for r in best))
    return {"local_source_ids": [r["id"] for r in best],
            "status": "local_lexical_candidate_requires_review" if supported else "uncertain_local_support",
            "review_required": True}


def model_register(register):
    """Compact presentation data; all full evidence remains in private JSON."""
    items = []
    for item in register["items"]:
        support = local_support(item["task"], item["evidence"])
        items.append({**{key: item[key] for key in ("id", "category", "task", "owners", "member_facing", "concerns", "status")},
                      "source_ids": support["local_source_ids"], "support": support["status"]})
    return {"items": items, "evidence_scope": "bounded local candidates; complete provenance retained privately",
            "validation": "operator_semantic_review_required"}


def source_excerpts(records):
    """One copy per real ID, exact whole lines. Oversized lines are explicit gaps."""
    supplied, omitted = {}, []
    for key, row in records.items():
        if len(row["text"]) > LOCAL_CHARS:
            omitted.append(key)
        else:
            # Speaker is already in the exact text; chunk identity is in the ID.
            supplied[key] = {field: row[field] for field in ("id", "section", "text")}
    return {"source_excerpts": list(supplied.values()), "omitted_oversized_source_ids": omitted,
            "evidence_policy": "exact lines; cite small consecutive ranges; gaps require operator review"}


def register_prompt(prompts, records, current, commitments, context):
    evidence = register_input(records, commitments, context)
    payload = {"summaries": current, "source_evidence": [r for r in evidence if len(r["text"]) <= LOCAL_CHARS],
               "omitted_oversized_source_ids": [r["id"] for r in evidence if len(r["text"]) > LOCAL_CHARS]}
    return (prompts / "register_prompt.txt").read_text(encoding="utf-8") + "\n\nUntrusted input JSON:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def notes_prompt(prompts, records, combined, register):
    payload = {"summaries": combined, **source_excerpts(records), "register": model_register(register)}
    return (prompts / "meeting_notes_prompt.txt").read_text(encoding="utf-8") + "\n\nUntrusted input JSON:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def detailed_prompt(prompts, records, current, register):
    prompt = (prompts / "minutes_prompt.txt").read_text(encoding="utf-8").replace("{chunk_summaries}", current)
    payload = {**source_excerpts(records), "register": model_register(register)}
    return prompt + "\n\nExact source excerpts take precedence over map paraphrases. Do not extract an independent Action Items section; Python inserts the canonical undertaking table.\nUntrusted input JSON:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def source_records(chunks):
    """Exact prepared lines: already redacted, classified and speaker-corrected."""
    result = {}
    for chunk in chunks:
        if chunk["meeting_section"] not in {BUSINESS, ADJOURNMENT, RECAP}:
            continue
        for number, line in enumerate(chunk["text"].splitlines(), 1):
            if not line.strip():
                continue
            key = f"{chunk['chunk_id']}:L{number}"
            if key in result:
                raise EditorialFailure("duplicate_source_id")
            turn = re.match(r"^\[([^\]]+)\]\s*(.*)$", line)
            result[key] = {"id": key, "chunk_id": str(chunk["chunk_id"]),
                           "source_chunk_id": str(chunk["source_chunk_id"]),
                           "start_time": chunk.get("start_time"), "end_time": chunk.get("end_time"),
                           "section": chunk["meeting_section"], "speaker": turn[1] if turn else "",
                           "text": line, "body": turn[2] if turn else line}
    return result


def register_input(records, commitments, context):
    # Retain exact, whole source lines and immediate context. This supplements
    # existing map summaries, not a second transcript-wide extraction layer.
    current = [r for r in records.values() if r["section"] != RECAP]
    selected = set()
    supplements = "\n".join(commitments + context)
    for i, row in enumerate(current):
        if _ACTION_CUE.search(row["body"]) or row["text"] in supplements:
            selected.add(i)
            if i and current[i-1]["chunk_id"] == row["chunk_id"]:
                selected.add(i-1)
    return [{"id": r["id"], "speaker": r["speaker"], "text": r["text"]}
            for i, r in enumerate(current) if i in selected]


def _object(value, fields):
    return isinstance(value, dict) and set(value) == set(fields.split())


def _text(value, limit=6000):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit


def _list(value):
    return isinstance(value, list)


def _refs(ids, records, *, recap=False):
    if not _list(ids) or not ids or any(not isinstance(i, str) for i in ids) or len(ids) != len(set(ids)):
        raise EditorialFailure("invalid_source_references")
    if any(i not in records or (not recap and records[i]["section"] == RECAP) for i in ids):
        raise EditorialFailure("invalid_source_references")
    positions = {key: i for i, key in enumerate(records)}
    if ids != sorted(ids, key=positions.get):
        raise EditorialFailure("unordered_source_references")
    return [records[i] for i in ids]


def _words(text):
    return set(re.findall(r"[\w]+", text.casefold())) - _STOP


def _direct_support(quotes, identity, task):
    """Exact local undertaking/assignment; exclude quoted speech and later actors."""
    for row in quotes:
        body = row["body"]
        if _QUOTED_FRAME.search(body) or re.search(r"\b(?:hypothetically|pretend|suppose|joking|kidding)\b", body, re.I):
            continue
        future = re.search(_FUTURE, body, re.I)
        named = _named_commitment(body, identity)
        if not named and not (row["speaker"] == identity and future):
            continue  # Being asked/addressed is not an accepted undertaking.
        safe = body
        step = _THIRD_PARTY_STEP.search(body, future.end() if future else 0)
        if step:
            safe = body[:step.start()].rstrip(" ,;")
        analysis = re.sub(_FUTURE, "I will", safe, flags=re.I)
        if _words(task) <= _words(safe) and _explicit_assignment(f"[{row['speaker']}] {analysis}", identity, task):
            return f"[{row['speaker']}] {safe}"
    return None


def validate_register(raw, records, commitments, source):
    if not _object(raw, "items") or not _list(raw["items"]):
        raise EditorialFailure("invalid_register_schema")
    items, seen, findings = [], set(), []
    for index, item in enumerate(raw["items"]):
        if (not _object(item, "id category task owners source_ids member_facing concerns")
                or not isinstance(item["id"], str) or not re.fullmatch(r"A[1-9]\d{0,4}", item["id"])
                or item["id"] in seen or not isinstance(item["category"], str) or item["category"] not in CATEGORIES
                or not _text(item["task"], 1200) or not _list(item["owners"])
                or any(not _text(n, 120) for n in item["owners"])
                or len(set(item["owners"])) != len(item["owners"])
                or type(item["member_facing"]) is not bool
                or not _list(item["concerns"]) or any(not isinstance(c, str) or c not in CONCERNS for c in item["concerns"])):
            raise EditorialFailure("invalid_register_item")
        seen.add(item["id"])
        quotes = _refs(item["source_ids"], records)
        evidence = "\n".join(r["text"] for r in quotes)
        # An existing source ID does not support arbitrary tasks. Material words
        # added by a paraphrase require operator revision, never silent acceptance.
        material = _words(item["task"])
        if not material or not material <= _words(evidence):
            raise EditorialFailure("unsupported_task_wording")
        support = local_support(item["task"], quotes)
        if support["status"] == "uncertain_local_support":
            findings.append({"code": "action_support_review", "item_id": item["id"], **support})
        if item["category"] == "undertaking":
            own = []
            for proof in commitments:
                lines = proof.splitlines()
                if all(any(line == r["text"] or r["text"].startswith(line) for r in quotes) for line in lines):
                    own.append(proof)
            if not any(material <= _words(proof) for proof in own) and not any(_direct_support(quotes, owner, item["task"]) for owner in item["owners"]):
                raise EditorialFailure("unsupported_undertaking")
            # Every owner must be supported in the cited passage, including
            # distinct people in a model-supplied multi-owner field.
            for owner in item["owners"]:
                if re.search(r"[()]", owner):
                    raise EditorialFailure("unsupported_owner_annotation")
                for identities in _owner_groups(owner):
                    if not identities or not any(
                        any(proof.splitlines()[-1].startswith(f"[{identity}]") and material <= _words(proof) for proof in own)
                        or _direct_support(quotes, identity, item["task"]) for identity in identities
                    ):
                        raise EditorialFailure("unsupported_owner")
            # Preserve conditions/qualification; review rather than upgrading a
            # qualified promise to an unconditional assignment.
            proofs = own + [proof for owner in item["owners"] if (proof := _direct_support(quotes, owner, item["task"]))]
            for proof in proofs:
                for qualifier in ("if", "when", "once", "try"):
                    if re.search(r"\b" + qualifier + r"\b", proof, re.I) and not re.search(r"\b" + qualifier + r"\b", item["task"], re.I):
                        raise EditorialFailure("lost_commitment_qualification")
            candidate = "# Action Items\n- " + (" and ".join(item["owners"]) or "Unassigned") + ": " + item["task"]
            if filter_completed_request_tasks(candidate, source, future_evidence=commitments) != candidate:
                raise EditorialFailure("rejected_action_semantics")
        elif item["category"] == "completed" and not re.search(r"\b(?:already|sent|completed|finished|done|noted|recorded|accepted)\b", evidence, re.I):
            raise EditorialFailure("unsupported_completion")
        elif item["category"] == "ongoing" and not re.search(r"\b(?:grieving|grievances?|pending|ongoing|working|in progress)\b", evidence, re.I):
            raise EditorialFailure("unsupported_ongoing_status")
        elif item["category"] == "external" and not re.search(r"\b(?:said|reported|promised|email|conference|convention|management)\b", evidence, re.I):
            raise EditorialFailure("unsupported_external_status")
        if item["category"] != "undertaking":
            # Named outside owners are reports, not speaker aliases/assignments.
            if any(not all(identity in evidence for group in _owner_groups(owner) for identity in group)
                   for owner in item["owners"]):
                raise EditorialFailure("unsupported_reported_owner")
        public = item["member_facing"]
        if public and (_SENSITIVE.search(item["task"]) or _IDENTIFYING_CASE.search(item["task"]) or "confidentiality" in item["concerns"]):
            raise EditorialFailure("confidential_action_presentation")
        if not public and item["category"] == "undertaking":
            findings.append({"code": "member_selection_review", "item_id": item["id"]})
        if (item["category"] == "undertaking" and not item["owners"]) or any(SPEAKER_LABEL.search(n) for n in item["owners"]):
            findings.append({"code": "unverified_owner", "item_id": item["id"]})
        findings.extend({"code": c, "item_id": item["id"]} for c in item["concerns"])
        items.append({**item, "status": STATUS[item["category"]], "evidence": quotes, "support": support})
    # A model cannot silently omit explicit source evidence to shorten a list.
    used = "\n".join(r["text"] for item in items for r in item["evidence"])
    for number, proof in enumerate(commitments, 1):
        if any(line not in used for line in proof.splitlines()):
            findings.append({"code": "commitment_coverage_review", "evidence_number": number})
    return {"version": 1, "source_hash": digest(records), "categories": CATEGORIES,
            "items": items, "findings": findings, "validation": "references_and_action_guards_checked; operator_semantic_review_required"}


def _plain(text):
    if (not _text(text) or "\n" in text or re.search(r"[#<>|`*_]|!\[|\]\(|(?:https?://)", text)
            or SPEAKER_LABEL.search(text) or re.search(r"\b\d+(?:\.\d+)?:L\d+\b", text)):
        raise EditorialFailure("invalid_plain_text")
    if _SENSITIVE.search(text) or _IDENTIFYING_CASE.search(text):
        raise EditorialFailure("confidential_notes_content")
    return text.strip()


def validate_notes(raw, records, source):
    if not _object(raw, "highlights previous_context issues motions unresolved concerns"):
        raise EditorialFailure("invalid_notes_schema")
    if not _list(raw["concerns"]) or any(not isinstance(c, str) or c not in CONCERNS for c in raw["concerns"]):
        raise EditorialFailure("invalid_review_flags")
    def block(value, historical=False):
        if not _object(value, "text source_ids"):
            raise EditorialFailure("invalid_notes_block")
        _plain(value["text"])
        refs = _refs(value["source_ids"], records, recap=historical)
        if historical and any(r["section"] != RECAP for r in refs):
            raise EditorialFailure("invalid_recap_section")
        # These deterministic repairs use the same explicit motion evidence as
        # detailed minutes. Plain prose remains subject to operator fact review.
        return {**value, "text": correct_adjournment_roles(value["text"], source),
                "support": local_support(value["text"], refs)}
    clean = {"concerns": raw["concerns"]}
    for key in ("highlights", "previous_context", "motions", "unresolved"):
        if not _list(raw[key]):
            raise EditorialFailure("invalid_notes_section")
        clean[key] = [block(b, key == "previous_context") for b in raw[key]]
    if not _list(raw["issues"]) or not raw["issues"]:
        raise EditorialFailure("missing_current_business")
    headings = set()
    clean["issues"] = []
    reserved = {"meeting highlights", "previous-meeting context", "recorded motions and decisions", "recorded undertakings", "unresolved matters", "open questions", "important notes", "action items"}
    for issue in raw["issues"]:
        if not _object(issue, "heading paragraphs") or not _list(issue["paragraphs"]) or not issue["paragraphs"]:
            raise EditorialFailure("invalid_issue_section")
        heading = _plain(issue["heading"])
        if heading.casefold() in reserved | headings:
            raise EditorialFailure("duplicate_or_reserved_heading")
        headings.add(heading.casefold())
        clean["issues"].append({"heading": heading, "paragraphs": [block(p) for p in issue["paragraphs"]]})
    return clean


def undertaking_table(register):
    lines = ["| Undertaking | Responsible person or verified role | Recorded status |", "|---|---|---|"]
    for item in register["items"]:
        if item["category"] != "undertaking" or not item["member_facing"]:
            continue
        task = _plain(item["task"])
        owner = "; ".join(item["owners"]) if item["owners"] and not any(SPEAKER_LABEL.search(n) for n in item["owners"]) else "Owner awaiting confirmation"
        owner = _plain(owner)
        lines.append(f"| {task} | {owner} | {item['status']} |")
    return "\n".join(lines) if len(lines) > 2 else "No verified member-facing undertakings available; review remains pending."


def render_notes(notes, register):
    lines = ["# Division 070 Meeting Notes", "", "Draft for operator review — meeting date, identities and distribution authority awaiting confirmation.", "",
             "These informal notes are separate from the division's official minutes. Status reflects the recorded discussion, not a later completion check.", "", "## Meeting Highlights", ""]
    lines += ["- " + b["text"] for b in notes["highlights"]] or ["None noted."]
    lines += ["", "## Previous-meeting context", ""]
    lines += [b["text"] + "\n" for b in notes["previous_context"]] or ["No previous-meeting context included.\n"]
    for issue in notes["issues"]:
        lines += ["## " + issue["heading"], ""]
        lines += [b["text"] + "\n" for b in issue["paragraphs"]]
    lines += ["## Recorded motions and decisions", ""]
    lines += [b["text"] + "\n" for b in notes["motions"]] or ["No supported motions or decisions recorded.\n"]
    lines += ["## Recorded Undertakings", "", undertaking_table(register), "",
              "Reported outside undertakings, existing casework, proposals and completed work remain separate from this table.", "", "## Unresolved Matters", ""]
    lines += [b["text"] + "\n" for b in notes["unresolved"]] or ["None noted.\n"]
    return "\n".join(lines).strip() + "\n"


def replace_actions(minutes, register):
    lines, output, skip_depth = minutes.splitlines(), [], None
    for line in lines:
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading and skip_depth is not None and len(heading[1]) <= skip_depth:
            skip_depth = None
        if heading and heading[2].strip().casefold() in {"action items", "recorded undertakings"}:
            skip_depth = len(heading[1])
        if skip_depth is None:
            output.append(line)
    return "\n".join(output).rstrip() + "\n\n## Action Items\n\n" + undertaking_table(register) + "\n"


def private_text(path, text):
    # Artifacts are created only in a new isolated private output directory.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.chmod(path, 0o600)


def review_groups(register, notes, flags, records):
    """Publication relevance, with actionable pointers into complete private evidence."""
    groups = [
        {"id": "publication_gates", "priority": 1, "heading": "Meeting date and distribution authority", "entries": [
            {"code": "meeting_date", "instruction": "Confirm the actual meeting date and resolve relative deadlines against source evidence."},
            {"code": "distribution_authority", "instruction": "Confirm authority and audience for informal notes, official minutes and committee attachments separately."}]},
        {"id": "member_undertaking_owners", "priority": 1, "heading": "Member-facing undertaking owners", "entries": []},
        {"id": "substantive_conflicts", "priority": 1, "heading": "Substantive conflicts and uncertain support", "entries": []},
        {"id": "confidentiality", "priority": 1, "heading": "Confidentiality and audience", "entries": [
            {"code": "confidentiality_review", "instruction": "Review member prose and selected undertakings for identifying case details and attachment restrictions."}]},
        {"id": "member_content", "priority": 2, "heading": "Member-facing facts and selection", "entries": []},
        {"id": "private_background", "priority": 3, "heading": "Private background identities and action concerns", "entries": []},
    ]
    targets = {g["id"]: g["entries"] for g in groups}
    by_id = {item["id"]: item for item in register["items"]}
    instructions = {
        "unverified_owner": "Verify the owner against the exact undertaking turn; approve identity separately from speaker labels.",
        "owner_verification": "Confirm the named person or role against the exact undertaking turn and approved identity records.",
        "uncertain_identity": "Resolve the identity only if needed for member-facing wording or ownership.",
        "source_conflict": "Compare the conflicting source passages; preserve corrections and qualifications.",
        "action_support_review": "Check the task as a whole against a small local passage; scattered words do not establish an action.",
        "notes_support_review": "Compare this claim with its local cited passage; revise the claim or references if support is uncertain.",
        "confidentiality": "Confirm whether this detail and any associated document may reach the intended audience.",
    }
    # All selected owners need verification, including names/roles without labels.
    for item in register["items"]:
        if item["member_facing"] and item["category"] == "undertaking":
            flags = flags + [{"code": "owner_verification", "item_id": item["id"]}]
    for flag in flags:
        item = by_id.get(flag.get("item_id"))
        public = item is None or item["member_facing"]
        code = flag["code"]
        if code in {"unverified_owner", "uncertain_identity", "owner_verification"}:
            group = "member_undertaking_owners" if item and public and item["category"] == "undertaking" else "member_content" if public else "private_background"
        elif code == "confidentiality":
            group = "confidentiality"
        elif code in {"source_conflict", "action_support_review", "notes_support_review"}:
            group = "substantive_conflicts" if public else "private_background"
        else:
            group = "member_content" if public else "private_background"
        entry = {**flag, "instruction": instructions.get(code, "Review the cited source, wording, status and publication relevance before release.")}
        if item:
            entry.update(task=item["task"], owners=item["owners"], category=item["category"], member_facing=item["member_facing"],
                         local_source_ids=local_support(item["task"], item["evidence"])["local_source_ids"],
                         evidence_pointer=f"{REGISTER}#/items/{register['items'].index(item)}/evidence")
            if "private_review_detail" in item:
                entry["operator_detail"] = item["private_review_detail"]
        targets[group].append(entry)
    return groups


def notes_blocks(notes):
    for section in ("highlights", "previous_context", "motions", "unresolved"):
        for index, block in enumerate(notes[section]):
            yield f"{section}/{index}", block
    for index, issue in enumerate(notes["issues"]):
        for number, block in enumerate(issue["paragraphs"]):
            yield f"issues/{index}/paragraphs/{number}", block


def finish(directory, register, notes, detailed, source, findings=(), *, records):
    """Shared deterministic rendering, also used by offline response fixtures."""
    table = undertaking_table(register)
    document = render_notes(notes, register)
    checklist = ["# Private Operator Review", "", "Review hold — no publication authorization.", "",
                 "Source-ID and lexical checks do not certify factual accuracy. Review every material claim against its source.",
                 f"Complete provenance: [{REGISTER}]({REGISTER}) and [{NOTES_EVIDENCE}]({NOTES_EVIDENCE}).", ""]
    flags = register["findings"] + [{"code": c} for c in notes["concerns"]]
    if not 1800 <= len(document.split()) <= 2400:
        flags.append({"code": "length_review"})
    flags += [{"code": f.code} for f in findings]
    flags += [{"code": f.code} for f in check_minutes(document, source)]
    for location, block in notes_blocks(notes):
        support = block.get("support", local_support(block["text"], [records[key] for key in block["source_ids"]]))
        if support["status"] == "uncertain_local_support":
            flags.append({"code": "notes_support_review", "notes_location": location, "claim": block["text"],
                          "source_ids": block["source_ids"], **support,
                          "evidence_pointer": f"{NOTES_EVIDENCE}#/notes/{location}"})
    groups = review_groups(register, notes, flags, records)
    for group in groups:
        checklist += [f"## Priority {group['priority']} — {group['heading']}", ""]
        for entry in group["entries"]:
            label = entry["code"] + (": " + entry["item_id"] if "item_id" in entry else "")
            checklist += [f"- [ ] {label} — {entry['instruction']}"]
            if "task" in entry:
                checklist += [f"  Task: {entry['task']} Owners: {', '.join(entry['owners']) or 'Unassigned'}."]
            if "claim" in entry:
                checklist += [f"  Claim: {entry['claim']}"]
            if "evidence_pointer" in entry:
                checklist += [f"  Evidence: {entry['evidence_pointer']}; local candidate: {', '.join(entry.get('local_source_ids', [])) or 'none'}."]
            if "operator_detail" in entry:
                checklist += ["  Private review detail: " + " / ".join(entry["operator_detail"])]
        checklist += [""]
    review = {"status": "review_hold", "source_hash": register["source_hash"], "register_hash": digest(register),
              "notes_word_count": len(document.split()), "findings": flags, "review_groups": groups,
              "required": ["meeting_date", "owner_verification", "semantic_fact_review", "confidentiality", "distribution_authority"]}
    blocks = [b for key in ("highlights", "previous_context", "motions", "unresolved") for b in notes[key]]
    blocks += [b for issue in notes["issues"] for b in issue["paragraphs"]]
    cited = {key for block in blocks for key in block["source_ids"]}
    write_private_json(directory / NOTES_EVIDENCE, {"source_hash": register["source_hash"], "notes": notes,
                       "sources": records,
                       "coverage": {"eligible_records": len(records), "cited_records": len(cited),
                                    "semantic_completeness": "requires_operator_review"}})
    write_private_json(directory / REGISTER, register)
    write_private_json(directory / REVIEW, review)
    private_text(directory / CHECKLIST, "\n".join(checklist) + "\n")
    internal = ["# Private Canonical Action Register", "", "Operator review required; source references are private.", ""]
    for key, heading in CATEGORIES.items():
        internal += ["## " + heading, ""]
        for item in register["items"]:
            if item["category"] == key:
                internal += [f"- {item['id']}: {item['task']}", f"  - Owners: {', '.join(item['owners']) or 'Unassigned'}; {item['status']}",
                             f"  - Source: {', '.join(item['source_ids'])}", ""]
    private_text(directory / REGISTER_MD, "\n".join(internal))
    # No member document is written until all schemas, action guards and
    # confidentiality checks have succeeded. The hold is written first.
    outputs = {"meeting-notes-draft.md": document,
               "action-items.md": "# Recorded Undertakings\n\nDraft for operator review.\n\n" + table + "\n",
               "summary.md": "# Meeting Highlights\n\n" + "\n".join("- " + b["text"] for b in notes["highlights"]) + "\n",
               "minutes-draft.md": replace_actions(detailed, register)}
    for filename, content in outputs.items():
        (directory / filename).write_text(content, encoding="utf-8")
    write_report(directory, check_minutes(outputs["minutes-draft.md"], source) + list(findings))
    return review


def run(directory, chunks, combined, current, recap, commitments, context, aliases, approved, warnings, prompts, generate, keep_recap, *, budget):
    """Reuse action, summary and detailed-minutes reduction slots, in that order."""
    write_private_json(directory / REVIEW, {"status": "review_hold", "phase": "incomplete"})
    records = source_records(chunks)
    source = "\n".join(c["text"] for c in chunks if c["meeting_section"] in {BUSINESS, ADJOURNMENT})
    raw = {}
    def request(stage, prompt, structured=False):
        measured = budget.measure(stage, prompt, structured)
        write_private_json(directory / BUDGETS, {"requests": budget.measurements})
        if not measured["fits"]:
            raise EditorialFailure("editorial_" + stage + "_context_exceeded")
        return generate(prompt, structured=structured, num_predict=measured["reserved_output"])
    try:
        raw["register"] = request("register", register_prompt(prompts, records, current, commitments, context), True)
        proposed = json.loads(raw["register"])
        supplied = {r["id"] for r in register_input(records, commitments, context) if len(r["text"]) <= LOCAL_CHARS}
        if isinstance(proposed, dict) and isinstance(proposed.get("items"), list):
            for item in proposed["items"]:
                if isinstance(item, dict) and isinstance(item.get("source_ids"), list) and any(not isinstance(i, str) or i not in supplied for i in item["source_ids"]):
                    raise EditorialFailure("unsupplied_action_evidence")
        register = validate_register(proposed, records, commitments, source)
        write_private_json(directory / REGISTER, register)
        raw["notes"] = request("notes", notes_prompt(prompts, records, combined, register), True)
        visible = {key: row for key, row in records.items() if len(row["text"]) <= LOCAL_CHARS}
        notes = validate_notes(json.loads(raw["notes"]), visible, source)
        for key in records.keys() - visible.keys():
            register["findings"].append({"code": "source_excerpt_coverage_review", "source_id": key})
        detailed = request("detailed", detailed_prompt(prompts, records, current, register))
        if keep_recap and recap:
            historical = request("recap", (prompts / "recap_prompt.txt").read_text(encoding="utf-8").replace("{chunk_summaries}", recap))
            detailed = insert_recap(detailed, historical)
        detailed = prepare_text(clean_speaker_annotations(detailed, source, aliases, approved_passages=approved), aliases)
        detailed = correct_adjournment_roles(detailed, source)
        detailed = strip_private_references(strip_chunk_references(detailed, [c.get("file_name", "") for c in chunks]))
        review = finish(directory, register, notes, detailed, source, warnings, records=records)
        print(f"[editorial] review hold; {review['notes_word_count']} words; {len(review['findings'])} finding(s)", flush=True)
        return 0
    except Exception as exc:
        category = str(exc) if isinstance(exc, EditorialFailure) else "invalid_json_or_generation_failure"
        # Responses contain private source data. Never print exception/model text.
        write_private_json(directory / RESPONSE, {"source_hash": digest(records), "responses": raw})
        write_private_json(directory / REVIEW, {"status": "review_hold", "phase": "failed", "failure_category": category})
        print(f"[editorial] incomplete: {category}; no publication authorized", flush=True)
        return 1
