import argparse
import asyncio
import json
import random
import re
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.chat_template import apply_chat_template
from verl_ext.alfworld.utils import render_actions_json

DEFAULT_SYSTEM_PROMPT_TEMPLATE = """In this environment you have access to a set of tools you can use to assist with the user query. You may perform multiple rounds of function calls. In each round, you should call exactly one function.

Here are available functions in JSONSchema format:

{tool_schema}
In your response, you need to first think about the reasoning process in the mind and then conduct function calling to get the information or perform the actions if needed. The reasoning process and function calling are enclosed within <think> </think> and <tool_call> </tool_call> tags.

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

DEFAULT_USER_PROMPT_TEMPLATE = """Your task is: {task_description}

Your current observation is: {current_observation}

Your admissible actions of the current situation are:
{admissible_actions}

Please choose exactly one admissible action and call the tool to execute it.
Continue this process until you complete the task."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug whether a model can emit <think> and <tool_call> across one or more ALFWorld turns."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_file", required=True, help="Path to train/valid parquet generated for ALFWorld.")
    parser.add_argument(
        "--tool_config_path",
        default="examples/alfworld_multiturn/configs/tool_config/alfworld_tool_config.yaml",
    )
    parser.add_argument("--index", type=int, default=None, help="Use a fixed sample index instead of random pick.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_turns", type=int, default=1, help="Maximum assistant turns to debug.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--show_rendered_prompt", action="store_true")
    parser.add_argument("--system_prompt_path", default=None, help="Optional text file to override system prompt.")
    parser.add_argument("--user_prompt_path", default=None, help="Optional text file to override user prompt.")
    return parser.parse_args()


def _maybe_json_load(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            return json.loads(stripped)
    return value


def _normalize_nested(value: Any) -> Any:
    value = _maybe_json_load(value)
    if isinstance(value, np.ndarray):
        return [_normalize_nested(item) for item in value.tolist()]
    if isinstance(value, list | tuple):
        return [_normalize_nested(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_nested(item) for key, item in value.items()}
    if hasattr(value, "as_py"):
        return _normalize_nested(value.as_py())
    if hasattr(value, "tolist") and not isinstance(value, str | bytes):
        try:
            return _normalize_nested(value.tolist())
        except Exception:
            pass
    return value


def _pick_row(data_file: str, index: int | None, seed: int) -> tuple[int, dict[str, Any]]:
    dataframe = pd.read_parquet(data_file)
    if len(dataframe) == 0:
        raise ValueError(f"No rows found in {data_file}")

    if index is None:
        index = random.Random(seed).randrange(len(dataframe))
    if index < 0 or index >= len(dataframe):
        raise IndexError(f"index {index} out of range for dataset of size {len(dataframe)}")

    row = dataframe.iloc[index].to_dict()
    row["prompt"] = _normalize_nested(row["prompt"])
    row["extra_info"] = _normalize_nested(row.get("extra_info"))
    row["reward_model"] = _normalize_nested(row.get("reward_model"))
    _populate_fields_from_prompt(row)
    return index, row


def _extract_role_content(messages: list[dict[str, Any]], role: str) -> str:
    for message in messages:
        if message.get("role") == role:
            content = message.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _populate_fields_from_prompt(row: dict[str, Any]) -> None:
    messages = row.get("prompt")
    if not isinstance(messages, list | tuple):
        return

    messages = list(messages)
    row["_orig_system_prompt"] = _extract_role_content(messages, "system")
    user_prompt = _extract_role_content(messages, "user")
    row["_orig_user_prompt"] = user_prompt

    task_match = re.search(r"Your task is:\s*(.*?)\n\nYour current observation is:", user_prompt, re.DOTALL)
    observation_match = re.search(
        r"Your current observation is:\s*(.*?)\n\nYour admissible actions of the current situation are:",
        user_prompt,
        re.DOTALL,
    )
    actions_match = re.search(
        r"Your admissible actions of the current situation are:\s*(.*?)\n\nPlease choose exactly one admissible action",
        user_prompt,
        re.DOTALL,
    )

    row["task_description"] = task_match.group(1).strip() if task_match else ""
    row["current_observation"] = observation_match.group(1).strip() if observation_match else ""
    actions_text = actions_match.group(1).strip() if actions_match else "[]"
    try:
        parsed_actions = json.loads(actions_text)
        row["admissible_actions"] = parsed_actions if isinstance(parsed_actions, list) else []
    except Exception:
        row["admissible_actions"] = []


def _load_tools(tool_config_path: str):
    tools = initialize_tools_from_config(tool_config_path)
    tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tools]
    tool_map = {tool.name: tool for tool in tools}
    return tool_schemas, tool_map


def _read_text_if_needed(path: str | None) -> str | None:
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def _build_system_prompt_text(tool_schemas: list[dict[str, Any]], override_text: str | None) -> str:
    if override_text is not None:
        return override_text
    return DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(
        tool_schema=json.dumps(tool_schemas, ensure_ascii=False, indent=2)
    )


def _build_user_prompt_text(row: dict[str, Any], override_text: str | None) -> str:
    if override_text is not None:
        return override_text.format(
            task_description=row.get("task_description", ""),
            current_observation=row.get("current_observation", ""),
            admissible_actions=render_actions_json(row.get("admissible_actions", [])),
        )

    extra_info = row.get("extra_info") or {}
    task_metadata = extra_info.get("task_metadata") or {}
    return DEFAULT_USER_PROMPT_TEMPLATE.format(
        task_description=task_metadata.get("task_description", row.get("task_description", "")),
        current_observation=row.get("current_observation", ""),
        admissible_actions=render_actions_json(row.get("admissible_actions", [])),
    )


def _select_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def _build_prompt_text(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
) -> str:
    return apply_chat_template(
        tokenizer,
        messages,
        tools=tool_schemas,
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate_one_turn(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    device: str,
    args: argparse.Namespace,
) -> tuple[list[int], str]:
    model_inputs = tokenizer(prompt_text, return_tensors="pt")
    model_inputs = {key: value.to(device) for key, value in model_inputs.items()}

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        outputs = model.generate(**model_inputs, **generate_kwargs)

    prompt_len = int(model_inputs["input_ids"].shape[1])
    new_token_ids = outputs[0, prompt_len:].tolist()
    raw_output = tokenizer.decode(new_token_ids, skip_special_tokens=False)
    return new_token_ids, raw_output


async def _execute_tool_call(
    tool_name: str,
    tool_args: dict[str, Any],
    tool_map: dict[str, Any],
    agent_data: Any,
) -> tuple[ToolResponse, float, dict]:
    tool = tool_map[tool_name]
    kwargs = agent_data.tools_kwargs.get(tool_name, {})
    instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
    try:
        return await tool.execute(instance_id, tool_args, agent_data=agent_data)
    finally:
        await tool.release(instance_id)


def main() -> None:
    args = parse_args()

    sample_index, row = _pick_row(args.data_file, args.index, args.seed)
    tool_schemas, tool_map = _load_tools(args.tool_config_path)
    system_prompt_override = _read_text_if_needed(args.system_prompt_path)
    user_prompt_override = _read_text_if_needed(args.user_prompt_path)
    system_prompt_text = _build_system_prompt_text(tool_schemas, system_prompt_override)
    user_prompt_text = _build_user_prompt_text(row, user_prompt_override)
    messages = [
        {"role": "system", "content": system_prompt_text},
        {"role": "user", "content": user_prompt_text},
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device, dtype = _select_device_and_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    extra_info = row.get("extra_info") or {}
    agent_data = SimpleNamespace(
        request_id=uuid4().hex,
        tools_kwargs=extra_info.get("tools_kwargs", {}),
        extra_fields={},
    )
    parser = ToolParser.get_tool_parser("hermes", tokenizer)

    print("=" * 80)
    print(f"Sample index: {sample_index}")
    print(f"Data source: {row.get('data_source')}")
    print("=" * 80)
    print("System prompt:")
    print(system_prompt_text)
    print("=" * 80)
    print("User prompt:")
    print(user_prompt_text)

    for turn in range(1, args.max_turns + 1):
        prompt_text = _build_prompt_text(tokenizer, messages, tool_schemas)
        if args.show_rendered_prompt:
            print("=" * 80)
            print(f"Rendered model prompt before turn {turn}:")
            print(prompt_text)

        new_token_ids, raw_output = _generate_one_turn(model, tokenizer, prompt_text, device, args)
        parsed_content, tool_calls = asyncio.run(parser.extract_tool_calls(new_token_ids, tool_schemas))

        print("=" * 80)
        print(f"Turn {turn} raw model output:")
        print(raw_output)
        print("-" * 80)
        print(f"Contains <think>: {'<think>' in raw_output and '</think>' in raw_output}")
        print(f"Contains <tool_call>: {'<tool_call>' in raw_output and '</tool_call>' in raw_output}")
        print(f"Parsed tool calls: {len(tool_calls)}")

        messages.append({"role": "assistant", "content": raw_output})

        if not tool_calls:
            print("Parsed content without tool calls:")
            print(parsed_content)
            break

        first_tool_call = tool_calls[0]
        print(f"[tool_call] name={first_tool_call.name} arguments={first_tool_call.arguments}")
        tool_args = json.loads(first_tool_call.arguments)
        tool_response, tool_reward, tool_metrics = asyncio.run(
            _execute_tool_call(first_tool_call.name, tool_args, tool_map, agent_data)
        )
        print("-" * 80)
        print(f"Tool reward: {tool_reward}")
        print(f"Tool metrics: {tool_metrics}")
        print("Tool response:")
        print(tool_response.text or "")

        messages.append({"role": "tool", "content": tool_response.text or ""})

        if agent_data.extra_fields.get("done"):
            print("-" * 80)
            print("Episode marked done by tool session.")
            break


if __name__ == "__main__":
    main()
