#!/usr/bin/env python3
"""Export only the allowlisted public meeting documents, never private review."""
import argparse
from pathlib import Path

from meeting_postprocess.publication import export_archive, export_documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meeting_dir", type=Path, help="Private meeting summary directory")
    parser.add_argument("destination", type=Path, help="Output directory, or ZIP path with --archive")
    parser.add_argument("--archive", action="store_true", help="Write a ZIP containing only public meeting documents")
    args = parser.parse_args()
    try:
        if args.archive:
            export_archive(args.meeting_dir, args.destination)
        else:
            export_documents(args.meeting_dir, args.destination)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: Meeting export failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
