from pathlib import Path
import os

BASE_DIR = Path(
    os.environ.get("MEETING_PIPELINE_BASE_DIR", str(Path.home() / "meeting-pipeline"))
).expanduser()

LESSON_STATE_DIR = Path(
    os.environ.get("LESSON_STATE_DIR", str(BASE_DIR / "lesson-state"))
).expanduser()

CURRENT_SESSION_PATH = LESSON_STATE_DIR / "current-session.json"
SESSION_HISTORY_DIR = LESSON_STATE_DIR / "sessions"

LESSON_RECORDINGS_ROOT = Path(
    os.environ.get("LESSON_RECORDINGS_ROOT", str(BASE_DIR / "lesson-recordings"))
).expanduser()

LESSON_TRANSCRIPTS_ROOT = Path(
    os.environ.get("LESSON_TRANSCRIPTS_ROOT", str(BASE_DIR / "lesson-transcripts"))
).expanduser()

LESSON_SUMMARIES_ROOT = Path(
    os.environ.get("LESSON_SUMMARIES_ROOT", str(BASE_DIR / "lesson-summaries"))
).expanduser()

LAST_RECORDING_PATH = LESSON_RECORDINGS_ROOT / "last_recording.txt"

LESSON_CONTROLLER_HOST = os.environ.get("LESSON_CONTROLLER_HOST", "0.0.0.0")
LESSON_CONTROLLER_PORT = int(os.environ.get("LESSON_CONTROLLER_PORT", "8765"))
LESSON_CONTROLLER_API_KEY = os.environ.get("LESSON_CONTROLLER_API_KEY", "").strip()
LESSON_CONTROLLER_SCRIPT_TIMEOUT = int(
    os.environ.get("LESSON_CONTROLLER_SCRIPT_TIMEOUT", "30")
)

def ensure_controller_dirs() -> None:
    LESSON_STATE_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_HISTORY_DIR.mkdir(parents=True, exist_ok=True)