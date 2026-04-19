from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _as_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes, dict)):
        return [float(value) for value in values]
    return [float(values)]


def compute_score(
    data_source: str | None,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    *,
    success_reward: float = 10.0,
    **kwargs,
) -> dict[str, Any]:
    extra_info = extra_info or {}
    tool_rewards = _as_float_list(extra_info.get("tool_rewards"))
    success = bool(extra_info.get("success", False))
    invalid_action_count = int(extra_info.get("invalid_action_count", 0))
    final_env_score = float(extra_info.get("final_env_score", 0.0))

    score = (success_reward if success else 0.0) + sum(tool_rewards)
    return {
        "score": float(score),
        "success": success,
        "invalid_action_count": invalid_action_count,
        "final_env_score": final_env_score,
    }
