"""Opt-in editorial reduction: one private register, shared public presentations.

This is part of map/reduce, not an additional extraction pass. All results are
operator-review drafts. Reference checks do not certify paraphrase accuracy.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy
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
DIAGNOSTICS = "editorial-generation.private.json"
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
        # Initial drafting has no operator-approved assignments. Keep these
        # candidates in private review, not in either member-facing projection.
        if item["category"] == "undertaking":
            continue
        support = item.get("support") or local_support(item["task"], item["evidence"])
        items.append({**{key: item[key] for key in ("id", "category", "task", "owners", "member_facing", "concerns", "status")},
                      "source_ids": support["local_source_ids"], "support": support["status"]})
    return {"items": items, "evidence_scope": "bounded local candidates; complete provenance retained privately",
            "validation": "unapproved_candidates; operator_semantic_review_required",
            "selection_policy": "Undertaking candidates and excluded/private proposals remain in private review. Their absence does not remove substantive issues from source evidence or summaries. Discuss supported issues without promoting omitted actions into assignments."}


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


def _refs(ids, records, *, recap=False, sections=None, canonical=False):
    if not _list(ids) or not ids or any(not isinstance(i, str) for i in ids) or len(ids) != len(set(ids)):
        raise EditorialFailure("invalid_source_references")
    if any(i not in records or (not recap and records[i]["section"] == RECAP)
           or (sections is not None and records[i]["section"] not in sections) for i in ids):
        raise EditorialFailure("invalid_source_references")
    positions = {key: i for i, key in enumerate(records)}
    ordered = sorted(ids, key=positions.get)
    if not canonical and ids != ordered:
        raise EditorialFailure("unordered_source_references")
    return [records[i] for i in ordered]


def _words(text):
    return set(re.findall(r"[\w]+", text.casefold())) - _STOP


def _direct_undertakings(quotes, identity):
    """Source-owned commitment candidates, never proof of a proposed paraphrase."""
    result = []
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
        if _explicit_assignment(f"[{row['speaker']}] {analysis}", identity, analysis):
            result.append(f"[{row['speaker']}] {safe}")
    return result


def _transfer_recipients(text):
    # A narrow concrete-argument guard, not a general paraphrase verifier.
    pattern = (r"\b(?:send|sent|share[ds]?|circulat(?:e[ds]?|ing)|forward(?:ed)?|"
               r"distribut(?:e[ds]?|ing)|deliver(?:ed)?|provid(?:e[ds]?|ing)|give|gave)\b"
               r"[^.!?;\n]*?\b(?:to|with)[,\s]+(?:(?:uh|um)[,\s]+)*(.+?)(?=[,.!?;\n]|$|"
               r"\s+\b(?:after|before|if|when|once|tomorrow|today|next|and then)\b)")
    # ASR fillers after a preposition do not change the explicitly named recipient.
    explicit = [match[1] for match in re.finditer(pattern, text, re.I)]
    # A direct pronoun recipient is recorded, but never mapped to a speaker ID.
    pronouns = re.findall(r"\b(?:send|sent|give|gave)\s+(you|him|her|them|us)\b", text, re.I)
    return explicit + pronouns


def _check_transfer_recipients(task, proofs):
    recipients = [r for proof in proofs for r in _transfer_recipients(proof) if r]
    def words(text):
        return {w.rstrip("s") if len(w) > 3 else w for w in _words(text)} - {"all", "both", "them", "him", "her", "you", "us"}
    # Derive initial-letter equivalents only from recipient phrases in these
    # proofs. No meeting-wide aliases or global acronym-to-person mapping.
    equivalents = {}
    for recipient in recipients:
        for match in re.finditer(r"\b([a-z]{2,})\s+(?:and|&)\s+([a-z]{2,})\b", recipient, re.I):
            key = (match[1][0].casefold(), match[2][0].casefold())
            equivalents.setdefault(key, set()).add((match[1].casefold(), match[2].casefold()))
    review = []
    for recipient in _transfer_recipients(task):
        normalized = words(recipient)
        abbreviation = re.search(r"\b([a-z])\s*&\s*([a-z])\b", recipient, re.I)
        if abbreviation:
            expansions = equivalents.get((abbreviation[1].casefold(), abbreviation[2].casefold()), set())
            if len(expansions) == 1:
                expanded = next(iter(expansions))
                normalized -= {abbreviation[1].casefold(), abbreviation[2].casefold()}
                normalized |= words(" ".join(expanded))
            elif len(expansions) > 1 or not any(normalized <= words(r) for r in recipients):
                review.append("recipient_equivalence_review")
                continue
        if normalized and not any(normalized <= words(r) for r in recipients):
            if any(not words(r) or re.search(r"\b[a-z]\s*&\s*[a-z]\b", r, re.I) for r in recipients):
                review.append("recipient_equivalence_review")
            else:
                raise EditorialFailure("unsupported_recipient")
    return review


def _qualification_parts(text):
    text = re.sub(r"^\[[^]]+\]\s*", "", text.strip())
    future = re.search(_FUTURE, text, re.I)
    condition, action = "", text
    if future:
        lead, action = text[:future.start()].strip(" ,;"), text[future.end():].strip()
        if re.match(r"^(?:if|when|once|unless)\b", lead, re.I):
            condition = lead
    else:
        prefix = re.match(r"^((?:if|when|once|unless|provided|subject to|in the event)\b.*?),\s*(.+)$", text, re.I)
        if prefix:
            condition, action = prefix.groups()
    trailing = re.search(r"\s+((?:if|when|once|unless)\b.*)$", action, re.I)
    if trailing and not re.search(r"\b(?:see|check|ask|find out)$", action[:trailing.start()].strip(), re.I):
        condition, action = trailing[1], action[:trailing.start()]
    return condition, action


def _polarity(text):
    normalized = re.sub(r"\bfail(?:s|ed)?\s+to\b|\b\w+n['’]t\b", "not", text, flags=re.I)
    negative = bool(re.search(r"\b(?:not|never|unless|cannot)\b", normalized, re.I))
    grammar = {"not", "never", "unless", "cannot", "if", "when", "once", "will", "do", "does", "did", "try", "attempt"}
    predicate = {w.rstrip("s") if len(w) > 3 else w for w in _words(normalized) - grammar}
    return negative, predicate


def _check_undertaking_arguments(task, proofs):
    condition, action = _qualification_parts(task)
    scopes = list(dict.fromkeys(piece.strip() for proof in proofs
                               for piece in re.split(r"(?<=[.!?;])\s+|\n|\s+and\s+(?=" + _FUTURE + r")", proof, flags=re.I)
                               if re.search(_FUTURE + r"|\b(?:will|must|agreed to|is assigned to|is responsible for)\b", piece, re.I)))
    scores = [len(_words(action) & _words(_qualification_parts(scope)[1])) for scope in scopes]
    best = max(scores, default=0)
    candidates = [scope for scope, score in zip(scopes, scores) if score == best and score > 0]
    if len(candidates) != 1:
        return ["qualification_relationship_review"] + _check_transfer_recipients(task, proofs)
    selected = candidates[0]
    prior_condition, prior_action = _qualification_parts(selected)
    # Pronoun recovery is not invented here. Keep the original bounded proof
    # available for recipient review when the source says only "do that".
    review = _check_transfer_recipients(task, proofs)
    if sum(score > 0 for score in scores) > 1 and re.search(r"\band\b", action, re.I):
        review.append("qualification_relationship_review")
    if prior_condition and not condition:
        raise EditorialFailure("lost_commitment_qualification")
    if re.search(r"\btry\b", prior_action, re.I) and not re.search(r"\b(?:try|attempt)\b", action, re.I):
        raise EditorialFailure("lost_commitment_qualification")
    for prior, proposed in ((prior_condition, condition), (prior_action, action)):
        before, before_words = _polarity(prior)
        after, after_words = _polarity(proposed)
        if before != after:
            if before_words and before_words == after_words:
                raise EditorialFailure("contradictory_task_polarity")
            review.append("qualification_relationship_review")
    if condition and not prior_condition:
        review.append("qualification_relationship_review")
    return review


def _completion_evidence(quotes):
    for row in quotes:
        for clause in re.split(r"[.!?;]", row["body"]):
            match = re.search(r"\b(?:already|sent|completed|finished|done|noted|recorded|accepted)\b", clause, re.I)
            if match and not re.search(r"\b(?:not|never|haven['’]t|hadn['’]t|didn['’]t|isn['’]t|wasn['’]t|"
                                       r"will|would|should|if|when|once)\b|" + _FUTURE, clause[:match.end()], re.I):
                return True
    return False


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
        # Lexical mismatch is uncertainty, not evidence that a paraphrase is
        # false (or true). Structural actor/status/argument guards stay separate.
        if not _words(item["task"]):
            raise EditorialFailure("invalid_task_wording")
        support = local_support(item["task"], quotes)
        if support["status"] == "uncertain_local_support":
            findings.append({"code": "action_support_review", "item_id": item["id"], **support})
        if item["category"] == "undertaking":
            own = []
            for proof in commitments:
                lines = proof.splitlines()
                if all(any(line == r["text"] or r["text"].startswith(line) for r in quotes) for line in lines):
                    own.append(proof)
            direct = {identity: _direct_undertakings(quotes, identity)
                      for owner in item["owners"] for group in _owner_groups(owner) for identity in group}
            if not own and not any(direct.values()):
                raise EditorialFailure("unsupported_undertaking")
            # Every owner must be supported in the cited passage, including
            # distinct people in a model-supplied multi-owner field.
            for owner in item["owners"]:
                if re.search(r"[()]", owner):
                    raise EditorialFailure("unsupported_owner_annotation")
                for identities in _owner_groups(owner):
                    proofs = [proof for identity in identities for proof in own
                              if proof.splitlines()[-1].startswith(f"[{identity}]")]
                    proofs += [proof for identity in identities for proof in direct.get(identity, [])]
                    if not identities or not proofs:
                        raise EditorialFailure("unsupported_owner")
                    for code in _check_undertaking_arguments(item["task"], proofs):
                        findings.append({"code": code, "item_id": item["id"], "source_ids": item["source_ids"]})
                        support["status"] = "uncertain_local_support"
            if not item["owners"]:
                for code in _check_undertaking_arguments(item["task"], own):
                    findings.append({"code": code, "item_id": item["id"], "source_ids": item["source_ids"]})
                    support["status"] = "uncertain_local_support"
            candidate = "# Action Items\n- " + (" and ".join(item["owners"]) or "Unassigned") + ": " + item["task"]
            if filter_completed_request_tasks(candidate, source, future_evidence=commitments) != candidate:
                raise EditorialFailure("rejected_action_semantics")
        elif item["category"] == "completed" and not _completion_evidence(quotes):
            raise EditorialFailure("unsupported_completion")
        elif item["category"] == "ongoing" and not re.search(r"\b(?:grieving|grievances?|pending|ongoing|working|in progress)\b", evidence, re.I):
            raise EditorialFailure("unsupported_ongoing_status")
        elif item["category"] == "external" and not re.search(r"\b(?:said|reported|promised|email|conference|convention|management)\b", evidence, re.I):
            raise EditorialFailure("unsupported_external_status")
        if item["category"] != "undertaking":
            for code in _check_transfer_recipients(item["task"], [r["body"] for r in quotes]):
                findings.append({"code": code, "item_id": item["id"], "source_ids": item["source_ids"]})
                support["status"] = "uncertain_local_support"
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
        status = STATUS[item["category"]]
        if support["status"] == "uncertain_local_support":
            status = "Proposed " + item["category"] + "; semantic support requires operator review"
        items.append({**item, "status": status, "evidence": quotes, "support": support})
    # A model cannot silently omit explicit source evidence to shorten a list.
    used = "\n".join(r["text"] for item in items for r in item["evidence"])
    for number, proof in enumerate(commitments, 1):
        if any(line not in used for line in proof.splitlines()):
            findings.append({"code": "commitment_coverage_review", "evidence_number": number})
    return {"version": 1, "source_hash": digest(records), "categories": CATEGORIES,
            "items": items, "findings": findings, "validation": "references_and_action_guards_checked; operator_semantic_review_required"}


def audit_register(raw, records, commitments, source, *, supplied=None):
    """Assess every proposal without deleting/rewording it or certifying semantics."""
    if not _object(raw, "items") or not _list(raw["items"]):
        raise EditorialFailure("invalid_register_schema")
    outcomes, seen = [], set()
    for index, item in enumerate(raw["items"]):
        identifier = item.get("id") if isinstance(item, dict) else None
        outcome = {"position": index, "item_id": identifier, "outcome": "review_required", "findings": []}
        try:
            validated = validate_register({"items": [item]}, records, commitments, source)
            if identifier in seen:
                raise EditorialFailure("duplicate_register_id")
            if supplied is not None and any(key not in supplied for key in item["source_ids"]):
                raise EditorialFailure("unsupplied_action_evidence")
            outcome["findings"] = [f for f in validated["findings"] if f["code"] != "commitment_coverage_review"]
            outcome["support"] = validated["items"][0]["support"]
        except EditorialFailure as exc:
            outcome.update(outcome="hard_block", failure_category=str(exc))
        if isinstance(identifier, str):
            seen.add(identifier)
        outcomes.append(outcome)
    blocked = sum(row["outcome"] == "hard_block" for row in outcomes)
    return {"source_hash": digest(records), "proposed_register": raw, "sources": records,
            "outcomes": outcomes, "hard_block_count": blocked, "publication_status": "review_hold",
            "assessment_limit": "First hard violation per action; review findings are not semantic approval.",
            "register": None if blocked else validate_register(raw, records, commitments, source)}


def register_review_checklist(assessment):
    lines = ["# Private Register Review", "", "Publication/export hold — no action is semantically approved.", "",
             "Every original proposal and source record is retained in editorial-response.private.json.",
             "Lexical uncertainty requires checking the task, actor, recipient, status and conditions together.", ""]
    for result, item in zip(assessment["outcomes"], assessment["proposed_register"]["items"]):
        codes = [result["failure_category"]] if result["outcome"] == "hard_block" else [f["code"] for f in result["findings"]]
        lines += [f"- [ ] {result['item_id'] or 'Invalid item'}: {result['outcome']} — {', '.join(codes) or 'operator_semantic_review_required'}"]
        if isinstance(item, dict):
            lines += ["  Original proposal: " + json.dumps(item, ensure_ascii=False)]
        lines += [f"  Complete evidence: {RESPONSE}#/register_assessment/sources; proposal index {result['position']}.", ""]
    return "\n".join(lines) + "\n"


def triage_register(assessment, records, commitments, source):
    """Automatic candidate projection; original failures are retained, not waived."""
    proposed = assessment["proposed_register"]["items"]
    identifiers = [i.get("id") for i in proposed if isinstance(i, dict) and isinstance(i.get("id"), str)]
    candidates, exclusions = [], []
    for outcome in assessment["outcomes"]:
        position = outcome["position"]
        item = proposed[position]
        reason = outcome.get("failure_category") if outcome["outcome"] == "hard_block" else None
        if reason is None:
            if identifiers.count(item["id"]) != 1:
                reason = "ambiguous_duplicate_id"
            elif not item["member_facing"]:
                reason = "private_action_selection"
            else:
                reason = next((c for c in item["concerns"] if c in {
                    "source_conflict", "confidentiality", "uncertain_identity", "incomplete_coverage"}), None)
            if reason is None and any(_SENSITIVE.search(records[key]["text"]) or
                                      _IDENTIFYING_CASE.search(records[key]["text"]) for key in item["source_ids"]):
                reason = "source_confidentiality_review"
            # Financial-document references stay private even in plural/hyphenated
            # form. This presentation exclusion does not change the original audit.
            if reason is None and any(re.search(r"\bpay[ -]?stubs?\b", text, re.I) for text in
                                      [item["task"], *(records[key]["text"] for key in item["source_ids"])]):
                reason = "financial_confidentiality_review"
            if reason is None:
                try:
                    # A safe source candidate can still be unsafe to present,
                    # e.g. a task containing an unresolved speaker ID or markup.
                    _plain(item["task"])
                except EditorialFailure as exc:
                    reason = str(exc)
        if reason is not None:
            exclusions.append({"code": "excluded_action_review", "item_id": outcome["item_id"],
                               "proposal_index": position, "reason": reason,
                               "original_outcome": outcome["outcome"],
                               "evidence_pointer": f"{REGISTER}#/original_assessment/proposed_register/items/{position}"})
        else:
            candidates.append(deepcopy(item))
    # Reuse all action guards and coverage checks for the selected projection.
    # An empty projection is valid: source-grounded issue notes can still run.
    register = validate_register({"items": candidates}, records, commitments, source)
    for item in register["items"]:
        item["status"] = "Unapproved " + item["category"] + " candidate; wording, ownership and status require operator review"
    register.update(automatic_triage=True, original_assessment=deepcopy(assessment), exclusions=exclusions)
    register["findings"].extend(exclusions)
    return register


def _plain(text):
    if (not _text(text) or "\n" in text or re.search(r"[#<>|`*_]|!\[|\]\(|(?:https?://)", text)
            or SPEAKER_LABEL.search(text) or re.search(r"\b\d+(?:\.\d+)?:L\d+\b", text)):
        raise EditorialFailure("invalid_plain_text")
    if _SENSITIVE.search(text) or _IDENTIFYING_CASE.search(text):
        raise EditorialFailure("confidential_notes_content")
    return text.strip()


def validate_notes(raw, records, source, *, reference_changes=None):
    if not _object(raw, "highlights previous_context issues motions unresolved concerns"):
        raise EditorialFailure("invalid_notes_schema")
    if not _list(raw["concerns"]):
        raise EditorialFailure("invalid_review_flags")
    def references(ids, sections, location):
        # Validate every ID and its context before doing the sole allowed repair.
        refs = _refs(ids, records, recap=RECAP in sections, sections=sections, canonical=True)
        ordered = [r["id"] for r in refs]
        if ids != ordered and reference_changes is not None:
            reference_changes.append({"code": "notes_reference_order_normalized", "notes_location": location,
                                      "original_source_ids": list(ids), "canonical_source_ids": ordered})
        return refs
    concerns = []
    for index, concern in enumerate(raw["concerns"]):
        if (not _object(concern, "category text source_ids")
                or not isinstance(concern["category"], str) or concern["category"] not in CONCERNS
                or not _text(concern["text"])):
            raise EditorialFailure("invalid_notes_concern")
        # Private concern text may contain the sensitive detail that must never
        # enter member prose. Do not apply _plain's public-content guard here.
        refs = references(concern["source_ids"], {RECAP, BUSINESS, ADJOURNMENT}, f"concerns/{index}")
        concerns.append({**concern, "source_ids": [r["id"] for r in refs]})
    def block(value, location, historical=False):
        if not _object(value, "text source_ids"):
            raise EditorialFailure("invalid_notes_block")
        _plain(value["text"])
        refs = references(value["source_ids"], {RECAP} if historical else {BUSINESS, ADJOURNMENT}, location)
        # These deterministic repairs use the same explicit motion evidence as
        # detailed minutes. Plain prose remains subject to operator fact review.
        return {**value, "source_ids": [r["id"] for r in refs], "text": correct_adjournment_roles(value["text"], source),
                "support": local_support(value["text"], refs)}
    clean = {"concerns": concerns}
    for key in ("highlights", "previous_context", "motions", "unresolved"):
        if not _list(raw[key]):
            raise EditorialFailure("invalid_notes_section")
        clean[key] = [block(b, f"{key}/{index}", key == "previous_context") for index, b in enumerate(raw[key])]
    if not _list(raw["issues"]) or not raw["issues"]:
        raise EditorialFailure("missing_current_business")
    headings = set()
    clean["issues"] = []
    reserved = {"meeting highlights", "previous-meeting context", "recorded motions and decisions", "recorded undertakings", "unresolved matters", "open questions", "important notes", "action items"}
    for index, issue in enumerate(raw["issues"]):
        if not _object(issue, "heading paragraphs") or not _list(issue["paragraphs"]) or not issue["paragraphs"]:
            raise EditorialFailure("invalid_issue_section")
        heading = _plain(issue["heading"])
        if heading.casefold() in reserved | headings:
            raise EditorialFailure("duplicate_or_reserved_heading")
        headings.add(heading.casefold())
        clean["issues"].append({"heading": heading, "paragraphs": [block(p, f"issues/{index}/paragraphs/{number}")
                                                               for number, p in enumerate(issue["paragraphs"])]})
    return clean


def undertaking_table(register):
    # All current registers contain proposals or held candidates, not approved
    # assignments. Validation, selection and member_facing are not approval.
    # There is no assignment-approval input to this initial-draft renderer.
    return "Undertaking candidates remain in the private review register pending operator approval."


def notes_concern_findings(notes):
    return [{"code": concern["category"], "concern_text": concern["text"], "source_ids": concern["source_ids"],
             "evidence_pointer": f"{NOTES_EVIDENCE}#/notes/concerns/{index}"}
            for index, concern in enumerate(notes["concerns"])]


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
              "Reported outside undertakings, existing casework, proposals and completed work remain separate from division assignments.", "", "## Unresolved Matters", ""]
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
        "qualification_relationship_review": "Match each condition and qualification to its actual undertaking; do not borrow one from another action by the same speaker.",
        "recipient_equivalence_review": "Verify the local recipient phrase or acronym, or the person addressed by a pronoun; do not invent an alias or speaker mapping.",
        "notes_support_review": "Compare this claim with its local cited passage; revise the claim or references if support is uncertain.",
        "confidentiality": "Confirm whether this detail and any associated document may reach the intended audience.",
        "excluded_action_review": "Review the original proposal, cited sources and exclusion reason. It is withheld from the action projection, not approved or discarded. Assess any underlying workplace issue separately for appropriately qualified, non-identifying discussion.",
    }
    # All selected owners need verification, including names/roles without labels.
    for item in register["items"]:
        if item["member_facing"] and item["category"] == "undertaking":
            flags = flags + [{"code": "owner_verification", "item_id": item["id"]}]
    for flag in flags:
        identifier = flag.get("item_id")
        item = by_id.get(identifier) if isinstance(identifier, str) else None
        public = item is None or item["member_facing"]
        code = flag["code"]
        if code == "excluded_action_review":
            reason = flag["reason"]
            group = "private_background" if reason == "private_action_selection" else "confidentiality" if "confidential" in reason else "substantive_conflicts"
        elif code in {"unverified_owner", "uncertain_identity", "owner_verification"}:
            group = "member_undertaking_owners" if item and public and item["category"] == "undertaking" else "member_content" if public else "private_background"
        elif code == "confidentiality":
            group = "confidentiality"
        elif code in {"source_conflict", "action_support_review", "notes_support_review", "qualification_relationship_review", "recipient_equivalence_review"}:
            group = "substantive_conflicts" if public else "private_background"
        else:
            group = "member_content" if public else "private_background"
        entry = {**flag, "instruction": instructions.get(code, "Review the cited source, wording, status and publication relevance before release.")}
        if code == "excluded_action_review":
            entry["instruction"] += f" Exclusion: {flag['reason']}; original audit: {flag['original_outcome']}."
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
    flags = register["findings"] + notes_concern_findings(notes)
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
            label = entry["code"] + (": " + str(entry["item_id"]) if "item_id" in entry else "")
            checklist += [f"- [ ] {label} — {entry['instruction']}"]
            if "task" in entry:
                checklist += [f"  Task: {entry['task']} Owners: {', '.join(entry['owners']) or 'Unassigned'}."]
            if "claim" in entry:
                checklist += [f"  Claim: {entry['claim']}"]
            if "concern_text" in entry:
                checklist += [f"  Sources: {', '.join(entry['source_ids'])}."]
            if "evidence_pointer" in entry:
                checklist += [f"  Evidence: {entry['evidence_pointer']}; local candidate: {', '.join(entry.get('local_source_ids', [])) or 'none'}."]
            if "operator_detail" in entry:
                checklist += ["  Private review detail: " + " / ".join(entry["operator_detail"])]
        checklist += [""]
    review = {"status": "review_hold", "source_hash": register["source_hash"], "register_hash": digest(register),
              "notes_word_count": len(document.split()), "findings": flags, "review_groups": groups,
              "required": ["meeting_date", "owner_verification", "semantic_fact_review", "confidentiality", "distribution_authority"]}
    assessment = register.get("original_assessment")
    if assessment is not None:
        review.update(register_outcomes=assessment["outcomes"], hard_block_count=assessment["hard_block_count"],
                      action_exclusions=register["exclusions"], candidate_approval="none")
        checklist += ["## Complete original register audit", "", register_review_checklist(assessment)]
    blocks = [b for key in ("highlights", "previous_context", "motions", "unresolved") for b in notes[key]]
    blocks += [b for issue in notes["issues"] for b in issue["paragraphs"]]
    blocks += notes["concerns"]
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
    if assessment is not None:
        internal += ["## Complete original model register and validation findings", "", register_review_checklist(assessment)]
    private_text(directory / REGISTER_MD, "\n".join(internal))
    # All undertaking candidates and original blocked actions remain private.
    # Notes safeguards must succeed before any member draft.
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
    raw, assessment, diagnostics, reference_changes = {}, None, [], []
    phase = "register"
    def request(stage, prompt, structured=False):
        measured = budget.measure(stage, prompt, structured)
        write_private_json(directory / BUDGETS, {"requests": budget.measurements})
        if not measured["fits"]:
            raise EditorialFailure("editorial_" + stage + "_context_exceeded")
        def record_response(info):
            diagnostics.append({"stage": stage, **info})
            write_private_json(directory / DIAGNOSTICS, {"requests": diagnostics, "notes_reference_changes": reference_changes})
        # CPU fixtures may supply the legacy callable; the production adapter
        # explicitly accepts the response callback. No backend retry is made.
        import inspect
        parameters = inspect.signature(generate).parameters
        kwargs = {"response_callback": record_response} if "response_callback" in parameters else {}
        answer = generate(prompt, structured=structured, num_predict=measured["reserved_output"], **kwargs)
        if not isinstance(answer, str) or not answer.strip():
            raise EditorialFailure("ollama_empty_final_answer")
        return answer
    try:
        raw["register"] = request("register", register_prompt(prompts, records, current, commitments, context), True)
        try:
            proposed = json.loads(raw["register"])
        except json.JSONDecodeError:
            raise EditorialFailure("editorial_register_invalid_json") from None
        supplied = {r["id"] for r in register_input(records, commitments, context) if len(r["text"]) <= LOCAL_CHARS}
        assessment = audit_register(proposed, records, commitments, source, supplied=supplied)
        write_private_json(directory / RESPONSE, {"source_hash": digest(records), "responses": raw, "register_assessment": assessment})
        register = triage_register(assessment, records, commitments, source)
        write_private_json(directory / REGISTER, register)
        phase = "notes"
        raw["notes"] = request("notes", notes_prompt(prompts, records, combined, register), True)
        visible = {key: row for key, row in records.items() if len(row["text"]) <= LOCAL_CHARS}
        try:
            proposed_notes = json.loads(raw["notes"])
        except json.JSONDecodeError:
            raise EditorialFailure("editorial_notes_invalid_json") from None
        try:
            notes = validate_notes(proposed_notes, visible, source, reference_changes=reference_changes)
        finally:
            write_private_json(directory / DIAGNOSTICS, {"requests": diagnostics, "notes_reference_changes": reference_changes})
        for key in records.keys() - visible.keys():
            register["findings"].append({"code": "source_excerpt_coverage_review", "source_id": key})
        phase = "detailed"
        raw["detailed"] = request("detailed", detailed_prompt(prompts, records, current, register))
        detailed = raw["detailed"]
        if keep_recap and recap:
            phase = "recap"
            raw["recap"] = request("recap", (prompts / "recap_prompt.txt").read_text(encoding="utf-8").replace("{chunk_summaries}", recap))
            historical = raw["recap"]
            detailed = insert_recap(detailed, historical)
        detailed = prepare_text(clean_speaker_annotations(detailed, source, aliases, approved_passages=approved), aliases)
        detailed = correct_adjournment_roles(detailed, source)
        detailed = strip_private_references(strip_chunk_references(detailed, [c.get("file_name", "") for c in chunks]))
        write_private_json(directory / RESPONSE, {"source_hash": digest(records), "responses": raw, "register_assessment": assessment})
        review = finish(directory, register, notes, detailed, source, warnings, records=records)
        print(f"[editorial] review hold; {review['notes_word_count']} words; {len(review['findings'])} finding(s)", flush=True)
        return 0
    except Exception as exc:
        from .ollama_response import GenerationFailure
        category = str(exc) if isinstance(exc, (EditorialFailure, GenerationFailure)) else "editorial_internal_failure"
        # Responses contain private source data. Never print exception/model text.
        write_private_json(directory / RESPONSE, {"source_hash": digest(records), "responses": raw, "register_assessment": assessment})
        write_private_json(directory / REVIEW, {"status": "review_hold", "phase": "failed", "failure_category": category,
                                               "failed_stage": phase,
                                               "register_outcomes": assessment["outcomes"] if assessment else []})
        if assessment and not (directory / CHECKLIST).exists():
            private_text(directory / CHECKLIST, register_review_checklist(assessment))
        print(f"[editorial] incomplete: {category}; no publication authorized", flush=True)
        return 1
