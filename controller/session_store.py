from pathlib import Path

from .config import CURRENT_SESSION_PATH, SESSION_HISTORY_DIR
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