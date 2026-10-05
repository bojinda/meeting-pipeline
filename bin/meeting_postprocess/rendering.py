"""Final-document presentation, independent of model compliance."""

import re


_CHUNK_REFERENCE = re.compile(
    r"\b(?:(?:from|see|per)\s+)?(?:(?:source[ _-]+)?chunks?(?:[ _-]+(?:ids?|labels?))?|source[ _-]+ids?)"
    r"\s*[:=#-]?\s*(?:chunk-)?\d+(?:\.\d+)*(?:\s*(?:,|and|&)\s*(?:\d+(?:\.\d+)*))*\b",
    re.IGNORECASE,
)


def strip_chunk_references(content: str, source_filenames: list[str] | None = None) -> str:
    references = _CHUNK_REFERENCE
    if source_filenames:
        references = re.compile(
            references.pattern + "|(?<![\\w.-])(?:" + "|".join(re.escape(name) for name in source_filenames) + r")(?![\w.-])",
            re.IGNORECASE,
        )
    lines = []
    for line in content.splitlines():
        # Remove wrappers only when their entire content is provenance, so
        # parenthesized meeting facts survive next to a chunk citation.
        line = re.sub(
            r"\([^()]*\)|\[[^\[\]]*\]",
            lambda m: "" if references.search(m.group()) and not re.sub(
                r"[\s;,.:\[\]()]+|\b(?:from|see|source|file|filename)\b", "", references.sub("", m.group()), flags=re.IGNORECASE
            ) else m.group(),
            line,
        )
        line = references.sub("", line)
        line = re.sub(r"\(\s*\)|\[\s*\]", "", line)
        line = re.sub(r"[,;]\s*([)\]])", r"\1", line)
        line = re.sub(r"[,;:]\s*([.!?])", r"\1", line)
        line = re.sub(r"[ \t]+([,.;!?])", r"\1", line).rstrip()
        if re.fullmatch(r"\s*(?:#{1,6}|[-+*]|\d+\.)?\s*", line) and line.strip():
            continue
        if re.fullmatch(r"\s*(?:[-+*]\s*)?(?:source(?: file)?|file(?:name)?)\s*:\s*", line, re.IGNORECASE):
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def insert_recap(minutes: str, recap: str) -> str:
    # The caller owns the heading even if the model returns its own title.
    recap = "\n".join(line for line in recap.splitlines() if not line.lstrip().startswith("#")).strip()
    section = "## Recap of Previous Meeting\n\n" + (recap or "None noted.")
    title, _, body = minutes.lstrip().partition("\n")
    if title.startswith("# "):
        return title + "\n\n" + section + "\n\n" + body.lstrip()
    return "# Draft Minutes\n\n" + section + "\n\n" + minutes
