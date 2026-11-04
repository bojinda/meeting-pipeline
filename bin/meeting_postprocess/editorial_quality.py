"""Conservative review cues, never semantic approval or automatic shortening."""
import re

TRACE = "notes-source-review.private.md"
REVISION = (
    "Preserve the original response and draft. Revise a separate copy: consolidate repeated topics, "
    "keep previous-meeting context only where it explains current business, and retain material union "
    "business, outcomes, qualifications and confidentiality protections. Compare every revised claim "
    "with its cited source, check coverage and action status, and record any justified length exception. "
    "Never truncate or pad to a word count; human fact review and distribution approval remain required."
)


def blocks(notes):
    for section in ("highlights", "previous_context", "motions", "unresolved"):
        for index, block in enumerate(notes[section]):
            yield f"{section}/{index}", block
    for index, issue in enumerate(notes["issues"]):
        for number, block in enumerate(issue["paragraphs"]):
            yield f"issues/{index}/paragraphs/{number}", block


def review_quality(notes, records, document):
    """Heuristic warnings only: even matching words/numbers cannot prove support."""
    flags = []
    count = len(document.split())
    recap = sum(len(b["text"].split()) for b in notes["previous_context"])
    substantial = sum(len(r["text"].split()) for r in records.values()) >= 4000
    metrics = {"document_words": count, "previous_context_words": recap,
               "target_words": [1800, 2400], "substantial_source": substantial,
               "automatic_revision": False}
    if count > 2400 or (substantial and count < 1800):
        flags.append({"code": "length_review", "word_count": count,
                      "target_words": [1800, 2400], "instruction": REVISION})
    if recap > 180 or recap > max(80, count * .15):
        flags.append({"code": "previous_context_length_review", "word_count": recap,
                      "instruction": "Review the recap for relevance to current business; retain necessary context and preserve the full source. " + REVISION})
    seen = {}
    for index, (location, block) in enumerate(blocks(notes), 1):
        text = block["text"]
        source = "\n".join(records[key]["text"] for key in block["source_ids"])
        def flag(code, instruction):
            flags.append({"code": code, "notes_location": location,
                          "source_ids": list(block["source_ids"]),
                          "evidence_pointer": f"{TRACE}#block-{index}", "instruction": instruction})
        if re.search(r"\b(?:chair(?:person)?|presiding officer|president)\b", text, re.I):
            flag("role_attribution_review", "Verify the role and attribution against this exact turn and approved identity evidence. Nearby names and speaker labels do not establish the chair; use a participant when uncertain.")
        if re.search(r"\b(?:alleg\w*|rumou?rs?|hearsay|unverified|suspect\w*|false information|fraud\w*|theft|dishonest\w*)\b", source + "\n" + text, re.I):
            flag("allegation_status_review", "Check that allegations, disputed accounts and reported interpretations remain attributed and qualified throughout this paragraph; they are not confirmed facts.")
        if re.search(r"\b(?:\d+(?:\.\d+)?\s*(?:times|to\s+\d+|:)|(?:two|three|four|five|six|seven|eight|nine|ten) times|ratio|twice|double|triple|percent(?:age)?)\b|%", text, re.I):
            flag("numerical_basis_review", "Verify both quantities, units, denominators, time periods and comparison bases. Do not derive a new ratio or change the basis; matching numbers alone do not establish support.")
        if re.search(r"\b(?:began|begun|started|since|commenced)\b", text, re.I) and re.search(r"\b(?:audit\w*|investigat\w*|practice|policy)\b", text + "\n" + source, re.I):
            flag("chronology_review", "Verify which event began at the stated time. The start of a practice is distinct from its discovery, investigation or audit; preserve uncertainty and later corrections.")
        key = " ".join(re.findall(r"\w+", text.casefold()))
        if key in seen:
            flag("repeated_notes_block", f"This repeats {seen[key]}; consolidate in a reviewed copy without losing qualifications or material business.")
        seen[key] = location
    return metrics, flags


def _literal(text):
    fence = "`" * max(3, 1 + max((len(s) for s in re.findall(r"`+", str(text))), default=0))
    return f"{fence}text\n{text}\n{fence}"


def source_review(notes, records, source_hash):
    """Readable private claim-to-source view; original source text is untouched."""
    lines = ["# Private Notes Source Review", "", "Review hold. Contains private evidence; never distribute with member notes.",
             "Times are containing chunk bounds, not exact line timings. References are candidate support, not proof of accuracy.",
             f"Source hash: {source_hash}", ""]
    entries = list(blocks(notes)) + [(f"concerns/{i}", b) for i, b in enumerate(notes["concerns"])]
    for index, (location, block) in enumerate(entries, 1):
        lines += [f"## Block {index}", "", f"Notes location: {location}", "", _literal(block["text"]), ""]
        for key in block["source_ids"]:
            row = records[key]
            lines += [_literal(f"{key} | original chunk {row.get('source_chunk_id', row.get('chunk_id'))} | "
                              f"{row.get('start_time')}–{row.get('end_time')} seconds | {row.get('section')}\n{row['text']}"), ""]
    return "\n".join(lines)
