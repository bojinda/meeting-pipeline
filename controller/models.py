from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class StartLessonRequest(BaseModel):
    title: str
    url: str
    source: str = "chrome_extension"
    user_label: str | None = None
    started_at: datetime | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class StopLessonRequest(BaseModel):
    session_id: str | None = None
    stopped_at: datetime | None = None


class SessionRecord(BaseModel):
    session_id: str
    slug: str
    title: str
    url: str
    source: str
    user_label: str | None = None
    started_at: datetime
    stopped_at: datetime | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    state: str = "recording"


class ActionRunResult(BaseModel):
    action: str
    returncode: int
    stdout: str = ""
    stderr: str = ""


class LessonActionResponse(BaseModel):
    ok: bool = True
    session: SessionRecord
    action: ActionRunResult