from dataclasses import dataclass
from pathlib import Path
import subprocess

from .config import BASE_DIR, LESSON_CONTROLLER_SCRIPT_TIMEOUT
from .models import ActionRunResult


@dataclass(frozen=True)
class ActionSpec:
    name: str
    command: list[str]
    cwd: Path


class CommandRegistry:
    def __init__(self) -> None:
        self._actions: dict[str, ActionSpec] = {}

    def register(self, name: str, command: list[str], cwd: Path | None = None) -> None:
        self._actions[name] = ActionSpec(
            name=name,
            command=command,
            cwd=cwd or BASE_DIR,
        )

    def list_actions(self) -> list[str]:
        return sorted(self._actions.keys())

    def get(self, name: str) -> ActionSpec:
        if name not in self._actions:
            raise KeyError(f"Unknown action: {name}")
        return self._actions[name]

    def run(self, name: str) -> ActionRunResult:
        spec = self.get(name)
        result = subprocess.run(
            spec.command,
            cwd=spec.cwd,
            capture_output=True,
            text=True,
            timeout=LESSON_CONTROLLER_SCRIPT_TIMEOUT,
            check=False,
        )
        return ActionRunResult(
            action=name,
            returncode=result.returncode,
            stdout=(result.stdout or "").strip(),
            stderr=(result.stderr or "").strip(),
        )


def build_default_registry() -> CommandRegistry:
    registry = CommandRegistry()

    registry.register("lesson_start", [str(BASE_DIR / "bin" / "ha-start-lesson.sh")])
    registry.register("lesson_stop", [str(BASE_DIR / "bin" / "ha-stop-lesson.sh")])

    registry.register("meeting_start", [str(BASE_DIR / "bin" / "ha-start-meeting.sh")])
    registry.register("meeting_stop", [str(BASE_DIR / "bin" / "ha-stop-meeting.sh")])

    return registry