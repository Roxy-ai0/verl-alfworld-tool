from __future__ import annotations

from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .agent_loop import INVALID_TOOL_RESPONSE_TEXT
from .session_registry import get_or_create_session, mark_finished, snapshot
from .utils import (
    admissible_action_map,
    extract_task_and_observation,
    first_batch_item,
    format_tool_response,
    normalize_action_list,
    normalize_action,
)


class AlfworldStepTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.completed_session_ttl_seconds = int(config.get("completed_session_ttl_seconds", 900))
        self.max_sessions = int(config.get("max_sessions", 4096))

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        if agent_data is None:
            raise ValueError("agent_data is required for AlfworldStepTool.execute")

        request_id = agent_data.request_id
        session_init = (
            agent_data.tools_kwargs.get(self.name, {}).get("create_kwargs", {}).get("session_init", {})
            if agent_data.tools_kwargs
            else {}
        )
        if not session_init:
            raise ValueError("extra_info.tools_kwargs.alfworld_step.create_kwargs.session_init is required")

        session = get_or_create_session(
            request_id,
            session_init,
            ttl_seconds=self.completed_session_ttl_seconds,
            max_sessions=self.max_sessions,
        )

        raw_action = parameters.get("action")
        display_action = "" if raw_action is None else str(raw_action)

        if session.done:
            self._update_agent_extra_fields(agent_data, session)
            return (
                ToolResponse(
                    text=format_tool_response(
                        action=display_action,
                        current_observation=f"The episode is already finished.\n\n{session.current_observation}",
                        admissible_actions=session.admissible_actions,
                        done=session.done,
                        success=session.success,
                        score=session.last_score,
                        step_id=session.step_id,
                    )
                ),
                0.0,
                snapshot(session),
            )

        normalized_map = admissible_action_map(session.admissible_actions)
        invalid = False
        if raw_action is None or not isinstance(raw_action, str):
            invalid = True
        else:
            normalized_action = normalize_action(raw_action)
            if not normalized_action or normalized_action not in normalized_map:
                invalid = True

        if invalid:
            session.invalid_action_count += 1
            session.touch()
            self._update_agent_extra_fields(agent_data, session)
            return (
                ToolResponse(text=INVALID_TOOL_RESPONSE_TEXT),
                0.0,
                snapshot(session),
            )

        action = normalized_map[normalize_action(raw_action)]
        observations, scores, dones, infos = session.env.step([action])
        stepped_observation = str(first_batch_item(observations)).strip()
        _, session.current_observation = extract_task_and_observation(stepped_observation)
        admissible_commands = (
            infos.get("admissible_commands", [session.admissible_actions])
            if isinstance(infos, dict)
            else [session.admissible_actions]
        )
        session.admissible_actions = normalize_action_list(first_batch_item(admissible_commands))
        session.last_score = float(first_batch_item(scores))
        session.done = bool(first_batch_item(dones))
        won = infos.get("won", [False]) if isinstance(infos, dict) else [False]
        session.success = bool(first_batch_item(won)) if session.done else False
        session.step_id += 1
        session.touch()
        if session.done:
            mark_finished(session)

        self._update_agent_extra_fields(agent_data, session)
        return (
            ToolResponse(
                text=format_tool_response(
                    action=action,
                    current_observation=session.current_observation,
                    admissible_actions=session.admissible_actions,
                    done=session.done,
                    success=session.success,
                    score=session.last_score,
                    step_id=session.step_id,
                )
            ),
            0.0,
            snapshot(session),
        )

    def _update_agent_extra_fields(self, agent_data: Any, session: Any) -> None:
        agent_data.extra_fields.update(snapshot(session))
