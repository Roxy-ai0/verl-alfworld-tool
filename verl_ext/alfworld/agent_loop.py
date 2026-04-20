import json

from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.experimental.agent_loop.agent_loop import register
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput


INVALID_TOOL_RESPONSE_TEXT = 'obs: ""'
INVALID_TOOL_CALL_NAME = "__alfworld_invalid_tool_call__"


@register("alfworld_tool_agent")
class AlfworldToolAgentLoop(ToolAgentLoop):
    def _has_complete_tool_call_wrapper(self, text: str) -> bool:
        return "<tool_call>" in text and "</tool_call>" in text

    async def _handle_generating_state(self, agent_data: AgentData, sampling_params: dict, ignore_termination: bool = False):
        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )

        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tools = [tool.tool_schema for tool in active_tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)

        if not agent_data.tool_calls:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            if self._has_complete_tool_call_wrapper(assistant_message):
                agent_data.tool_calls = [FunctionCall(name=INVALID_TOOL_CALL_NAME, arguments="{}")]
                return AgentState.PROCESSING_TOOLS

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            agent_data.messages.extend([{"role": "assistant", "content": assistant_message}])
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict, agent_data: AgentData):
        if tool_call.name == INVALID_TOOL_CALL_NAME:
            return ToolResponse(text=INVALID_TOOL_RESPONSE_TEXT), 0.0, {}

        tool, instance_id = None, None
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        try:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = active_tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
        except Exception:
            return ToolResponse(text=INVALID_TOOL_RESPONSE_TEXT), 0.0, {}
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        tool_response_kwargs = {"text": tool_response_text}
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res
