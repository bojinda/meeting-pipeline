"""The single file-selection boundary for meeting export/publication payloads."""
from __future__ import annotations

from pathlib import Path
import re
import zipfile

from .redaction import PRIVATE_REDACTION_FILENAME
from .speaker_suggestions import PRIVATE_SUGGESTIONS_FILENAME, PRIVATE_ROSTER_FILENAME
from .speaker_turns import CORRECTIONS_FILE, TURNS_FILE, CONFLICTS_FILE


PUBLIC_MEETING_FILENAMES = ("summary.md", "action-items.md", "minutes-draft.md")
PRIVATE_MEETING_FILENAMES = frozenset({PRIVATE_REDACTION_FILENAME.casefold(), PRIVATE_SUGGESTIONS_FILENAME.casefold(), PRIVATE_ROSTER_FILENAME.casefold(), "speaker_aliases.json", CORRECTIONS_FILE, TURNS_FILE, CONFLICTS_FILE, "whole-source.json", "whole-evidence.json", "whole-plan.json", "whole-run.json"})


def strip_private_references(content: str) -> str:
    """Do not publish links/attachments to the private review record either."""
    private = "(?:" + "|".join(re.escape(name) for name in sorted(PRIVATE_MEETING_FILENAMES)) + ")"
    content = re.sub(r"\[[^\]\n]*\]\([^\n)]*" + private + r"[^\n)]*\)", "", content, flags=re.IGNORECASE)
    content = re.sub(r"<a\b[^>]*" + private + r"[^>]*>.*?</a>", "", content, flags=re.IGNORECASE | re.DOTALL)
    # Also reject reference-link definitions, autolinks, and other embedded
    # filename references. A private-record reference must never be exported.
    return "\n".join(line for line in content.splitlines() if not any(name in line.casefold() for name in PRIVATE_MEETING_FILENAMES))


def _public_content(path: Path) -> str:
    return strip_private_references(path.read_text(encoding="utf-8"))


def public_meeting_files(directory: Path) -> list[Path]:
    files = []
    private_files = [path for path in directory.iterdir() if path.name.casefold() in PRIVATE_MEETING_FILENAMES] if directory.is_dir() else []
    for name in PUBLIC_MEETING_FILENAMES:
        path = directory / name
        # Explicit private-name denial plus a positive filename allowlist. No
        # recursive copy, suffix-only filtering, or symlink traversal is allowed.
        if any(part.casefold() in PRIVATE_MEETING_FILENAMES for part in path.parts):
            continue
        if path.is_symlink() or any(part.casefold() in PRIVATE_MEETING_FILENAMES for part in path.resolve().parts):
            continue
        if path.exists() and any(private.exists() and path.samefile(private) for private in private_files):
            continue
        if path.is_file():
            files.append(path)
    return files


def publication_payload(directory: Path) -> dict:
    return {"documents": [{"name": path.name, "content": _public_content(path)} for path in public_meeting_files(directory)]}


def export_documents(directory: Path, destination: Path) -> list[Path]:
    files = public_meeting_files(directory)
    if any(part.casefold() in PRIVATE_MEETING_FILENAMES for part in destination.parts):
            raise ValueError("Export destination cannot be a private review path")
    if directory.resolve() == destination.resolve():
        raise ValueError("Export destination must differ from the private processing directory")
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for source in files:
        target = destination / source.name
        if target.is_symlink():
            raise ValueError("Export targets cannot be symbolic links")
        target.write_text(_public_content(source).rstrip() + "\n", encoding="utf-8")
        written.append(target)
    return written


def export_archive(directory: Path, destination: Path) -> None:
    if any(part.casefold() in PRIVATE_MEETING_FILENAMES for part in destination.parts) or destination.is_symlink():
        raise ValueError("Archive destination cannot be a private review path or symbolic link")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in public_meeting_files(directory):
            archive.writestr(path.name, _public_content(path).rstrip() + "\n")
