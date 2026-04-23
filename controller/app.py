from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    LESSON_CONTROLLER_API_KEY,
    ensure_controller_dirs,
)
from .models import (
    LessonActionResponse,
    SessionRecord,
    StartLessonRequest,
    StopLessonRequest,
)
from .registry import build_default_registry
from .session_store import load_current_session, save_session, sync_session_artifacts
from .utils import build_session_id, slugify, utc_now

app = FastAPI(title="Meeting Pipeline Controller", version="1.0.0")
registry = build_default_registry()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if LESSON_CONTROLLER_API_KEY and x_api_key != LESSON_CONTROLLER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.on_event("startup")
def on_startup() -> None:
    ensure_controller_dirs()


@app.get("/api/v1/health")
def health() -> dict:
    return {
        "ok": True,
        "actions": registry.list_actions(),
    }


@app.get("/api/v1/lessons/current", dependencies=[Depends(require_api_key)])
def get_current_lesson() -> dict:
    session = load_current_session()
    return {
        "ok": True,
        "session": None if session is None else session.model_dump(mode="json"),
    }


@app.post(
    "/api/v1/lessons/start",
    response_model=LessonActionResponse,
    dependencies=[Depends(require_api_key)],
)
def start_lesson(payload: StartLessonRequest) -> LessonActionResponse:
    current = load_current_session()
    if current and current.state == "recording" and current.stopped_at is None:
        raise HTTPException(
            status_code=409,
            detail="A lesson session is already active.",
        )

    started_at = payload.started_at or utc_now()
    slug_source = payload.user_label or payload.title or "lesson"
    slug = slugify(slug_source)
    session_id = build_session_id(slug, started_at)

    session = SessionRecord(
        session_id=session_id,
        slug=slug,
        title=payload.title,
        url=payload.url,
        source=payload.source,
        user_label=payload.user_label,
        started_at=started_at,
        extra=payload.extra,
        state="starting",
    )
    save_session(session)

    result = registry.run("lesson_start")
    if result.returncode != 0:
        session.state = "start_failed"
        save_session(session)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "lesson_start failed",
                "stderr": result.stderr,
                "stdout": result.stdout,
            },
        )

    session.state = "recording"
    save_session(session)
    return LessonActionResponse(session=session, action=result)


@app.post(
    "/api/v1/lessons/stop",
    response_model=LessonActionResponse,
    dependencies=[Depends(require_api_key)],
)
def stop_lesson(payload: StopLessonRequest) -> LessonActionResponse:
    session = load_current_session()
    if session is None:
        raise HTTPException(status_code=404, detail="No active lesson session found.")

    if payload.session_id and payload.session_id != session.session_id:
        raise HTTPException(
            status_code=409,
            detail="Session ID does not match the current lesson session.",
        )

    session.stopped_at = payload.stopped_at or utc_now()
    session.state = "stopping"
    save_session(session)

    result = registry.run("lesson_stop")
    if result.returncode != 0:
        session.state = "stop_failed"
        save_session(session)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "lesson_stop failed",
                "stderr": result.stderr,
                "stdout": result.stdout,
            },
        )

    session.state = "stopped"
    save_session(session)
    sync_session_artifacts(session)
    return LessonActionResponse(session=session, action=result)


@app.post("/api/v1/actions/{action_name}", dependencies=[Depends(require_api_key)])
def run_action(action_name: str) -> dict:
    try:
        result = registry.run(action_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"{action_name} failed",
                "stderr": result.stderr,
                "stdout": result.stdout,
            },
        )

    return {
        "ok": True,
        "action": result.model_dump(),
    }