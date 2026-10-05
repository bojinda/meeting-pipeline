"""Deterministically label historical recap separately from current minutes."""


def insert_recap(minutes: str, recap: str) -> str:
    # The caller owns the heading even if the model returns its own title.
    recap = "\n".join(line for line in recap.splitlines() if not line.lstrip().startswith("#")).strip()
    section = "## Recap of Previous Meeting\n\n" + (recap or "None noted.")
    title, _, body = minutes.lstrip().partition("\n")
    if title.startswith("# "):
        return title + "\n\n" + section + "\n\n" + body.lstrip()
    return "# Draft Minutes\n\n" + section + "\n\n" + minutes
