"""Explicit offline dispositions for held drafts, independent of model output.

This module does not resume inference or approve publication. An excluded private
proposal keeps its original hard failure; every selected derivative must pass
the existing register guards. Speaker tokens are never converted to names.
"""
from copy import deepcopy
from pathlib import Path

from . import editorial as ed


DISPOSITIONS = "editorial-dispositions.private.json"
TREATMENTS = {"selected_undertaking", "private", "issue_note", "recorded_business"}


def reviewed_register(raw, records, commitments, source, decisions):
    """Return a checked selected view plus the complete unchanged original audit."""
    original = ed.audit_register(raw, records, commitments, source)
    if not ed._object(decisions, "version scope source_hash commitments_hash original_register_hash original_outcomes_hash authorization rows"):
        raise ed.EditorialFailure("invalid_disposition_schema")
    if (type(decisions["version"]) is not int or decisions["version"] != 1 or decisions["scope"] != "held_draft_only"
            or not ed._text(decisions["authorization"], 12000)):
        raise ed.EditorialFailure("disposition_authorization_required")
    if (decisions["source_hash"] != ed.digest(records)
            or decisions["commitments_hash"] != ed.digest(commitments)
            or decisions["original_register_hash"] != ed.digest(raw)
            or decisions["original_outcomes_hash"] != ed.digest(original["outcomes"])):
        raise ed.EditorialFailure("stale_disposition_identity")
    if [line for line in source.splitlines() if line.strip()] != [
            r["text"] for r in records.values() if r["section"] in {ed.BUSINESS, ed.ADJOURNMENT}]:
        raise ed.EditorialFailure("disposition_source_text_mismatch")
    rows = decisions["rows"]
    ids = [item.get("id") for item in raw["items"]]
    if (not isinstance(rows, list) or len(ids) != len(set(ids))
            or any(not isinstance(row, dict) for row in rows)
            or [row.get("id") for row in rows] != ids):
        raise ed.EditorialFailure("incomplete_disposition_coverage")
    selected, selections = [], []
    for item, outcome, row in zip(raw["items"], original["outcomes"], rows):
        if (not ed._object(row, "id treatment classification reason context_source_ids candidate")
                or row["treatment"] not in TREATMENTS
                or not ed._text(row["classification"], 200)
                or not ed._text(row["reason"], 6000)):
            raise ed.EditorialFailure("invalid_disposition_row")
        context = ed._refs(row["context_source_ids"], records)
        if not set(item["source_ids"]) <= {r["id"] for r in context}:
            raise ed.EditorialFailure("lost_original_disposition_references")
        if row["treatment"] != "selected_undertaking":
            if row["candidate"] is not None:
                raise ed.EditorialFailure("private_disposition_has_candidate")
            continue
        # An operator selection cannot erase a pre-existing hard violation.
        if outcome["outcome"] == "hard_block":
            raise ed.EditorialFailure("selected_original_hard_block")
        candidate = row["candidate"]
        if (not ed._object(candidate, "task source_ids qualification")
                or not ed._text(candidate["qualification"], 1200)
                or not isinstance(candidate["source_ids"], list)
                or not set(candidate["source_ids"]) <= {r["id"] for r in context}
                or not set(item["source_ids"]) <= set(candidate["source_ids"])):
            raise ed.EditorialFailure("invalid_selected_candidate")
        ed._plain(candidate["qualification"])
        derived = {**deepcopy(item), "category": "undertaking", "member_facing": True,
                   "task": candidate["task"], "source_ids": candidate["source_ids"]}
        # Owners and concerns are inherited unchanged. No aliases, status waiver,
        # edited source records, or reduced commitment evidence are accepted here.
        checked = ed.validate_register({"items": [derived]}, records, commitments, source)
        selected.append(derived)
        selections.append({"id": item["id"], "original_hash": ed.digest(item),
                           "candidate": derived, "validation": checked["validation"],
                           "findings": checked["findings"],
                           "qualification": candidate["qualification"],
                           "identity_status": "source_actor_checked; final_identity_unconfirmed"})
    if not selected:
        raise ed.EditorialFailure("empty_reviewed_selection")
    register = ed.validate_register({"items": selected}, records, commitments, source)
    register["editorial_scope"] = "operator_selected_held_draft; semantic_review_remains_required"
    register["original_register_hash"] = ed.digest(raw)
    register["original_outcomes_hash"] = ed.digest(original["outcomes"])
    return original, register, selections


def prepare_held_draft(directory, raw, records, commitments, source, decisions, notes):
    """Write to a new private directory only; no model, backend or publication API."""
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    # Place the existing fail-closed publication marker before doing any work.
    ed.write_private_json(directory / ed.REVIEW, {"status": "review_hold", "phase": "incomplete"})
    original = ed.audit_register(raw, records, commitments, source)
    ed.write_private_json(directory / ed.RESPONSE, {"register_assessment": original, "notes_proposal": deepcopy(notes)})
    ed.write_private_json(directory / DISPOSITIONS, deepcopy(decisions))
    try:
        original, register, selections = reviewed_register(raw, records, commitments, source, decisions)
        reference_changes = []
        try:
            checked_notes = ed.validate_notes(notes, records, source, reference_changes=reference_changes)
        finally:
            ed.write_private_json(directory / ed.DIAGNOSTICS, {"requests": [], "notes_reference_changes": reference_changes})
        # Identity confirmation is outside this bounded mechanism. Even a name
        # in source text is not automatically promoted to a public owner label.
        presentation = deepcopy(register)
        for item, selection in zip(presentation["items"], selections):
            item["owners"] = []
            item["status"] = "Held candidate. " + selection["qualification"]
        document = ed.render_notes(checked_notes, presentation)
        document = document.replace("Draft for operator review — meeting date, identities and distribution authority awaiting confirmation.",
                                    "Held draft for operator review — meeting date, final owner names, audience and distribution authority awaiting confirmation. Publication and export are on hold.")
        register["selected_records"] = selections
        flags = list(register["findings"])
        for location, block in ed.notes_blocks(checked_notes):
            flags.append({"code": "notes_semantic_review", "notes_location": location,
                          "source_ids": block["source_ids"], "support": block["support"]})
        flags += ed.notes_concern_findings(checked_notes)
        review = {"status": "review_hold", "phase": "held_draft_prepared",
                  "source_hash": ed.digest(records), "dispositions_hash": ed.digest(decisions),
                  "original_hard_block_count": original["hard_block_count"],
                  "original_outcomes": original["outcomes"], "selected_ids": [s["id"] for s in selections],
                  "findings": flags, "notes_word_count": len(document.split()),
                  "required": ["meeting_date", "final_owner_names", "semantic_fact_review", "confidentiality", "audience", "distribution_authority"],
                  "publication_authorized": False, "inference_calls": 0}
        ed.write_private_json(directory / ed.REGISTER, register)
        ed.write_private_json(directory / ed.NOTES_EVIDENCE, {"notes": checked_notes, "sources": records,
                              "source_hash": ed.digest(records), "semantic_approval": False})
        ed.write_private_json(directory / ed.REVIEW, review)
        checklist = ["# Held draft operator review", "", "D1–D4 authorize editorial preparation only. Publication/export remain on hold.",
                     "Original proposals, sources and original hard failures are retained in editorial-response.private.json.",
                     "Every selected derivative passed existing hard guards; this does not certify semantic accuracy or owner identity.", ""]
        for row, outcome in zip(decisions["rows"], original["outcomes"]):
            checklist.append(f"- {row['id']}: {row['classification']}; {row['treatment']}; original {outcome['outcome']}"
                             + (f" ({outcome['failure_category']})" if outcome['outcome'] == "hard_block" else "")
                             + f". {row['reason']}")
        checklist += ["", "Remaining operator decisions:"] + ["- [ ] " + key for key in review["required"]]
        ed.private_text(directory / ed.CHECKLIST, "\n".join(checklist) + "\n")
        ed.private_text(directory / "meeting-notes-draft.md", document)
        ed.private_text(directory / "action-items.md", "# Recorded Undertakings\n\nHeld draft; owners and distribution unconfirmed.\n\n" + ed.undertaking_table(presentation) + "\n")
        return review
    except Exception as exc:
        code = str(exc) if isinstance(exc, ed.EditorialFailure) else "invalid_offline_disposition_input"
        ed.write_private_json(directory / ed.REVIEW, {"status": "review_hold", "phase": "failed", "failure_category": code,
                              "original_outcomes": original["outcomes"]})
        raise
