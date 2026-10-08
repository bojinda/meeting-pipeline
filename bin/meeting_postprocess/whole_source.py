"""Lossless model-facing source encoding and read-only token accounting."""
from __future__ import annotations

import copy
import hashlib
import json

from .sections import BUSINESS, ADJOURNMENT, RECAP

VERSION = "meeting-source-v1"
SECTIONS = {"B": BUSINESS, "R": RECAP, "A": ADJOURNMENT}
COLUMNS = ["record_id", "speaker_index", "text"]


class SourceEncodingError(ValueError):
    pass


def serialized(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def fingerprint(records):
    return hashlib.sha256(serialized(records).encode("utf-8")).hexdigest()


def encode(records):
    speakers, runs, mapping, gaps = [], [], {}, []
    codes = {section: code for code, section in SECTIONS.items()}
    for number, record in enumerate(records, 1):
        if record["section"] not in codes or not isinstance(record["id"], str) or not isinstance(record["speaker"], str) or not isinstance(record["text"], str):
            raise SourceEncodingError("invalid_original_source_record")
        if record["speaker"] not in speakers:
            speakers.append(record["speaker"])
        code = codes[record["section"]]
        if not runs or runs[-1][0] != code:
            runs.append([code, []])
        runs[-1][1].append([number, speakers.index(record["speaker"]), record["text"]])
        mapping[str(number)] = record["id"]
        if record.get("redaction_gap"):
            gaps.append(number)
    payload = {"format": VERSION, "columns": list(COLUMNS), "sections": dict(SECTIONS), "speakers": speakers, "runs": runs, "redaction_gaps": gaps}
    result = {"payload": payload, "compact_to_original": mapping, "original_records_digest": fingerprint(records)}
    validate(result, records)
    return result


def validate(encoded, records):
    """Verify bijection, record order, attribution, boundaries and exact bodies."""
    try:
        originals = [row["id"] for row in records]
        mapping = encoded["compact_to_original"]
        payload = encoded["payload"]
        if len(set(originals)) != len(originals) or mapping != {str(i): row["id"] for i, row in enumerate(records, 1)} or encoded["original_records_digest"] != fingerprint(records):
            raise SourceEncodingError("ambiguous_or_missing_source_mapping")
        if set(payload) != {"format", "columns", "sections", "speakers", "runs", "redaction_gaps"} or payload["format"] != VERSION or payload["columns"] != COLUMNS or payload["sections"] != SECTIONS:
            raise SourceEncodingError("invalid_source_encoding")
        speakers = payload["speakers"]
        if not isinstance(speakers, list) or not all(isinstance(name, str) for name in speakers) or len(set(speakers)) != len(speakers):
            raise SourceEncodingError("ambiguous_speaker_mapping")
        number = 0
        previous = None
        for code, rows in payload["runs"]:
            if code not in SECTIONS or code == previous or not isinstance(rows, list) or not rows:
                raise SourceEncodingError("invalid_section_boundary")
            previous = code
            for sid, speaker, text in rows:
                number += 1
                if number > len(records) or type(sid) is not int or sid != number or type(speaker) is not int or not 0 <= speaker < len(speakers):
                    raise SourceEncodingError("missing_or_colliding_compact_record")
                original = records[number - 1]
                if SECTIONS[code] != original["section"] or speakers[speaker] != original["speaker"] or not isinstance(text, str) or text != original["text"]:
                    raise SourceEncodingError("source_encoding_mismatch")
        if number != len(records) or not isinstance(payload["redaction_gaps"], list) or not all(type(i) is int for i in payload["redaction_gaps"]) or payload["redaction_gaps"] != [i for i, row in enumerate(records, 1) if row.get("redaction_gap")]:
            raise SourceEncodingError("missing_source_record_or_redaction_boundary")
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        if isinstance(exc, SourceEncodingError):
            raise
        raise SourceEncodingError("invalid_source_encoding") from None
    return mapping


def resolve_evidence(data, encoded, records):
    mapping = validate(encoded, records)
    resolved = copy.deepcopy(data)
    if not isinstance(resolved, dict) or not isinstance(resolved.get("items"), list):
        return resolved  # The existing evidence schema validator rejects it.
    for item in resolved["items"]:
        if not isinstance(item, dict) or not isinstance(item.get("quotes"), list):
            continue
        for quote in item["quotes"]:
            if not isinstance(quote, dict):
                continue
            reference = quote.get("record_id")
            if not isinstance(reference, str) or reference not in mapping:
                raise SourceEncodingError("unknown_compact_source_reference")
            quote["record_id"] = mapping[reference]
    return resolved


def verbose_payload(records):
    return {"meeting_records": [{key: row[key] for key in ("id", "section", "speaker", "text")} for row in records]}


def source_breakdown(payload, counter, compact):
    """Exact marginal ablations; BPE effects are order-dependent, not ignored."""
    value = copy.deepcopy(payload)
    total = previous = counter(serialized(value))
    result = {}
    def rows():
        return [row for _, run in value["runs"] for row in run] if compact else value["meeting_records"]
    for row in rows():
        if compact:
            row[2] = ""
        else:
            row["text"] = ""
    current = counter(serialized(value))
    result["transcript_text"] = previous - current
    previous = current
    for row in rows():
        if compact:
            row[0] = ""
        else:
            row["id"] = ""
    current = counter(serialized(value))
    result["record_ids"] = previous - current
    previous = current
    if compact:
        value["sections"] = {key: "" for key in value["sections"]}
        for run in value["runs"]:
            run[0] = ""
    else:
        for row in rows():
            row["section"] = ""
    current = counter(serialized(value))
    result["section_identifiers"] = previous - current
    previous = current
    if compact:
        value["speakers"] = ["" for _ in value["speakers"]]
        for row in rows():
            row[1] = ""
    else:
        for row in rows():
            row["speaker"] = ""
    current = counter(serialized(value))
    result["speaker_identifiers"] = previous - current
    result["json_fields_structure_and_empty_placeholders"] = current
    result["serialized_source_total"] = total
    return result


def budget(records, encoded, counter, system, legacy_system, schema, context, output):
    validate(encoded, records)
    compact = source_breakdown(encoded["payload"], counter, True)
    verbose = source_breakdown(verbose_payload(records), counter, False)
    schema_tokens = counter(json.dumps(schema))
    instructions = counter(system)
    legacy_instructions = counter(legacy_system)
    required = compact["serialized_source_total"] + instructions + schema_tokens + output + 1024
    legacy_required = verbose["serialized_source_total"] + legacy_instructions + schema_tokens + output + 1024
    return {"token_count_method": counter.method, "component_method": "ordered_marginal_ablation_text_ids_sections_speakers",
            "compact_source": compact, "previous_verbose_source": verbose,
            "plain_transcript_text_standalone": counter("\n".join(row["text"] for row in records)),
            "system_instructions": instructions, "previous_system_instructions": legacy_instructions,
            "schema": schema_tokens, "reserved_generation": output, "reserved_framing": 1024,
            "configured_context": context, "required_context": required, "previous_required_context": legacy_required,
            "tokens_saved": legacy_required - required, "headroom": context - required,
            "selected_mode": "whole" if required <= context else "map_reduce_fallback"}
