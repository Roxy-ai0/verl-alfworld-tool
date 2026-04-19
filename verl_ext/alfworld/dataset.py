import json
from typing import Any

from .utils import render_actions_json


SYSTEM_PROMPT_TEMPLATE = """In this environment you have access to a set of tools you can use to assist with the user query. You may perform multiple rounds of function calls. In each round, you should call exactly one function.

Here are available functions in JSONSchema format:

{tool_schema}
In your response, you need to first conduct function calling to perform the next action if needed. The function calling is enclosed within <tool_call> </tool_call> tags.

The results of the function calls will be given back to you after execution, and you can continue to call functions until you get the final answer for the user's query.

Important rules:

At each round, call exactly one function.
The function arguments must contain exactly one action.
You must choose the action from the admissible actions provided in the current situation.
Do not invent actions that are not in the admissible action list.
If the environment indicates the task is finished, stop calling functions and provide a short plain-text final answer.
Keep your output concise and focused on the next action.

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


USER_PROMPT_TEMPLATE = """Your task is: {task_description}

Your current observation is: {current_observation}

Your admissible actions of the current situation are:
{admissible_actions}

Please choose exactly one admissible action and call the tool to execute it.
Continue this process until you complete the task."""


def get_alfworld_tool_schema_dict() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "alfworld_step",
            "description": "Execute exactly one admissible action in the current ALFWorld TextWorld episode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "One admissible action from the current ALFWorld situation.",
                    }
                },
                "required": ["action"],
            },
            "strict": True,
        },
    }


def build_system_prompt() -> str:
    tool_schema = json.dumps([get_alfworld_tool_schema_dict()], ensure_ascii=False, indent=2)
    return SYSTEM_PROMPT_TEMPLATE.format(tool_schema=tool_schema)


def build_user_prompt(task_description: str, current_observation: str, admissible_actions: list[str]) -> str:
    return USER_PROMPT_TEMPLATE.format(
        task_description=task_description,
        current_observation=current_observation,
        admissible_actions=render_actions_json(admissible_actions),
    )


def build_dataset_row(
    *,
    split: str,
    index: int,
    task_description: str,
    current_observation: str,
    admissible_actions: list[str],
    session_init: dict[str, Any],
    task_metadata: dict[str, Any],
    data_source: str = "alfworld_textworld",
) -> dict[str, Any]:
    return {
        "data_source": data_source,
        "agent_name": "tool_agent",
        "index": index,
        "split": split,
        "prompt": [
            {"role": "system", "content": build_system_prompt()},
            {
                "role": "user",
                "content": build_user_prompt(
                    task_description=task_description,
                    current_observation=current_observation,
                    admissible_actions=admissible_actions,
                ),
            },
        ],
        "reward_model": {"style": "rule", "ground_truth": task_description},
        "extra_info": {
            "split": split,
            "index": index,
            "need_tools_kwargs": True,
            "tool_selection": ["alfworld_step"],
            "tools_kwargs": {"alfworld_step": {"create_kwargs": {"session_init": session_init}}},
            "task_metadata": task_metadata,
        },
    }
