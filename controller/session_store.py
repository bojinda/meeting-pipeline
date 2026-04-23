import json
from pathlib import Path

from .config import (
    CURRENT_SESSION_PATH,
    LAST_RECORDING_PATH,
    LESSON_SUMMARIES_ROOT,
    LESSON_TRANSCRIPTS_ROOT,
    SESSION_HISTORY_DIR,
)
from .models import SessionRecord


def load_current_session() -> SessionRecord | None:
    if not CURRENT_SESSION_PATH.exists():
        return None
    return SessionRecord.model_validate_json(
        CURRENT_SESSION_PATH.read_text(encoding="utf-8")
    )


def save_session(session: SessionRecord) -> None:
    payload = session.model_dump_json(indent=2)
    CURRENT_SESSION_PATH.write_text(payload, encoding="utf-8")
    history_path = SESSION_HISTORY_DIR / f"{session.session_id}.json"
    history_path.write_text(payload, encoding="utf-8")


def _build_metadata_payload(
    session: SessionRecord,
    *,
    recording_file: str | None = None,
    session_name: str | None = None,
) -> dict:
    payload = session.model_dump(mode="json")
    if recording_file:
        payload["recording_file"] = recording_file
    if session_name:
        payload["transcript_dir"] = session_name
        payload["summary_dir"] = session_name
    return payload


def sync_session_artifacts(session: SessionRecord) -> list[str]:
    """
    Update metadata sidecars after the controller has the final session state.
    This refreshes:
      - lesson-recordings/<name>.metadata.json
      - lesson-transcripts/<name>/session-metadata.json   (if it exists)
      - lesson-summaries/<name>/session-metadata.json     (if it exists)
    """
    written: list[str] = []

    if not LAST_RECORDING_PATH.exists():
        return written

    raw = LAST_RECORDING_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return written

    recording_path = Path(raw)
    if not recording_path.exists():
        return written

    session_name = recording_path.stem
    payload = _build_metadata_payload(
        session,
        recording_file=recording_path.name,
        session_name=session_name,
    )
    text = json.dumps(payload, indent=2)

    wav_sidecar = recording_path.with_suffix(".metadata.json")
    wav_sidecar.write_text(text, encoding="utf-8")
    written.append(str(wav_sidecar))

    transcript_meta = LESSON_TRANSCRIPTS_ROOT / session_name / "session-metadata.json"
    if transcript_meta.parent.exists():
        transcript_meta.write_text(text, encoding="utf-8")
        written.append(str(transcript_meta))

    summary_meta = LESSON_SUMMARIES_ROOT / session_name / "session-metadata.json"
    if summary_meta.parent.exists():
        summary_meta.write_text(text, encoding="utf-8")
        written.append(str(summary_meta))

    return written