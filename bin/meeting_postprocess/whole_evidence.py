"""Compact evidence references hydrated from immutable, source-bound records."""
from __future__ import annotations

from copy import deepcopy
import re

from .whole_source import encode, validate, SourceEncodingError
from .commitments import commitment_evidence

FORMAT = "meeting-evidence-v2"
FIELDS = {"id", "kind", "section", "primary_record_id", "supporting_record_ids", "owners", "mover", "seconder", "outcome"}


def schema(legacy):
    original = legacy["properties"]["items"]["items"]["properties"]
    fields = {key: deepcopy(original[key]) for key in ("id", "kind", "section", "owners", "mover", "seconder", "outcome")}
    reference = {"type": "string", "pattern": r"^[1-9][0-9]*$", "description": "Canonical decimal compact source ID; never an evidence ID"}
    fields["kind"]["enum"] = ["motion", "action", "decision", "issue", "health_safety", "qualification", "recap", "topic"]
    fields["primary_record_id"] = reference
    fields["supporting_record_ids"] = {"type": "array", "items": reference, "uniqueItems": True, "description": "Relevant supporting source IDs; exclude primary ID and routine repetitions"}
    return {"type": "object", "additionalProperties": False, "required": ["format", "items"], "properties": {"format": {"type": "string", "const": FORMAT}, "items": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": sorted(FIELDS), "properties": fields}}}}


def hydrate(data, records, encoded, structure_check, legacy_fields):
    if not isinstance(data, dict) or set(data) != {"format", "items"} or data.get("format") != FORMAT or not isinstance(data["items"], list):
        return [], [{"item_index": 0, "category": "invalid_evidence_format", "field": "format"}], {}, {}
    encoded = encoded if encoded is not None else encode(records)
    mapping = validate(encoded, records)
    known = {row["id"]: row for row in records}
    restored, rejected, provenance, positions, seen = [], [], {}, {}, set()
    topic_sets = set()
    for index, row in enumerate(data["items"]):
        if not isinstance(row, dict) or set(row) != FIELDS:
            rejected.append({"item_index": index, "category": "missing_or_extra_fields", "field": "item"})
            continue
        candidate = {key: row[key] for key in ("id", "kind", "section", "owners", "mover", "seconder", "outcome")}
        candidate.update(statement="reference", quotes=[{"record_id": "1", "text": "reference"}])
        errors = structure_check(candidate, seen, legacy_fields)
        support = row["supporting_record_ids"]
        if not isinstance(support, list) or not all(isinstance(ref, str) for ref in support):
            errors.append({"category": "invalid_supporting_references", "field": "supporting_record_ids"})
        if errors:
            rejected.extend({"item_index": index, **error} for error in errors)
            continue
        refs = [row["primary_record_id"], *support]
        if any(not isinstance(ref, str) or ref not in mapping for ref in refs):
            error = SourceEncodingError("unknown_compact_source_reference")
            error.item_index = index
            raise error
        if len(set(refs)) != len(refs):
            rejected.append({"item_index": index, "category": "duplicate_source_reference", "field": "supporting_record_ids"})
            continue
        primary = known[mapping[refs[0]]]
        cited = [known[mapping[ref]] for ref in refs]
        group = (row["kind"], row["section"], frozenset(refs))
        if row["kind"] in {"topic", "recap"} and group in topic_sets:
            rejected.append({"item_index": index, "category": "duplicate_topic_evidence", "field": "supporting_record_ids"})
            continue
        topic_sets.add(group)
        statement = primary["text"]
        # An exact deterministic own-task span preserves qualifications while
        # excluding later third-party workflow steps. Nothing is paraphrased.
        if row["kind"] == "action":
            undertakings = commitment_evidence([{"meeting_section": primary["section"], "text": f"[{primary['speaker']}] {statement}"}])
            if undertakings:
                statement = re.sub(r"^\[[^\]]+\]\s*", "", undertakings[0])
                own_evidence = commitment_evidence([{"meeting_section": row["section"], "text": "\n".join(f"[{record['speaker']}] {record['text']}" for record in cited)}])
                if primary["speaker"] not in row["owners"] or not all(f"[{owner}] {statement}" in own_evidence for owner in row["owners"]):
                    rejected.append({"item_index": index, "category": "unsupported_primary_action_owner", "field": "owners"})
                    continue
        candidate.update(statement=statement, quotes=[{"record_id": record["id"], "text": record["text"]} for record in cited])
        positions[len(restored)] = index
        restored.append(candidate)
        provenance[row["id"]] = {"primary_record_id": primary["id"], "supporting_record_ids": [record["id"] for record in cited[1:]], "source_statement": statement}
    return restored, rejected, provenance, positions
