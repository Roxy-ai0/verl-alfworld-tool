import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .utils import build_single_game_env, close_env_quietly


@dataclass
class AlfworldSession:
    request_id: str
    env: Any
    task_description: str
    current_observation: str
    admissible_actions: list[str]
    last_score: float
    step_id: int
    done: bool
    success: bool
    invalid_action_count: int
    session_init: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def touch(self) -> None:
        self.updated_at = time.time()


_REGISTRY_LOCK = threading.Lock()
_SESSIONS: dict[str, AlfworldSession] = {}


def _ttl_seconds(default: int = 900) -> int:
    return default


def _max_sessions(default: int = 4096) -> int:
    return default


def _purge_expired_locked(now: float, ttl_seconds: int, max_sessions: int) -> None:
    expired_request_ids = [
        request_id
        for request_id, session in _SESSIONS.items()
        if session.finished_at is not None and now - session.finished_at >= ttl_seconds
    ]
    for request_id in expired_request_ids:
        close_env_quietly(_SESSIONS[request_id].env)
        del _SESSIONS[request_id]

    if len(_SESSIONS) <= max_sessions:
        return

    done_sessions = sorted(
        ((request_id, session) for request_id, session in _SESSIONS.items() if session.finished_at is not None),
        key=lambda item: item[1].finished_at or item[1].updated_at,
    )
    for request_id, session in done_sessions:
        if len(_SESSIONS) <= max_sessions:
            break
        close_env_quietly(session.env)
        del _SESSIONS[request_id]


def get_or_create_session(
    request_id: str,
    session_init: dict[str, Any],
    *,
    ttl_seconds: int | None = None,
    max_sessions: int | None = None,
) -> AlfworldSession:
    ttl_seconds = _ttl_seconds() if ttl_seconds is None else ttl_seconds
    max_sessions = _max_sessions() if max_sessions is None else max_sessions

    with _REGISTRY_LOCK:
        now = time.time()
        _purge_expired_locked(now, ttl_seconds=ttl_seconds, max_sessions=max_sessions)

        session = _SESSIONS.get(request_id)
        if session is not None:
            session.touch()
            return session

        env, current_observation, admissible_actions, task_description = build_single_game_env(session_init)
        session = AlfworldSession(
            request_id=request_id,
            env=env,
            task_description=task_description,
            current_observation=current_observation,
            admissible_actions=admissible_actions,
            last_score=0.0,
            step_id=0,
            done=False,
            success=False,
            invalid_action_count=0,
            session_init=session_init,
        )
        _SESSIONS[request_id] = session
        return session


def mark_finished(session: AlfworldSession) -> None:
    session.done = True
    session.finished_at = time.time()
    session.touch()


def snapshot(session: AlfworldSession) -> dict[str, Any]:
    return {
        "task_description": session.task_description,
        "current_observation": session.current_observation,
        "admissible_actions": list(session.admissible_actions),
        "final_env_score": float(session.last_score),
        "step_id": int(session.step_id),
        "done": bool(session.done),
        "success": bool(session.success),
        "invalid_action_count": int(session.invalid_action_count),
        "request_id": session.request_id,
        "session_init": dict(session.session_init),
    }
