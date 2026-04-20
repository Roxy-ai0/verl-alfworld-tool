import argparse
import asyncio
import concurrent.futures
import json
import multiprocessing as mp
import os
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
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.chat_template import apply_chat_template
from verl_ext.alfworld.utils import (
    TASK_TYPES,
    admissible_action_map,
    apply_textworld_overrides,
    build_single_game_env_fast,
    close_env_quietly,
    extract_task_and_observation,
    first_batch_item,
    format_tool_response,
    load_runtime_config,
    normalize_action,
    normalize_action_list,
    render_actions_json,
)

DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are an expert agent operating in the ALFRED Embodied Environment.In this environment you have access to a set of tools you can use to assist with the user query. You may perform multiple rounds of function calls. In each round, you should call exactly one function.

Here are available functions in JSONSchema format:

{tool_schema}
In your response, you should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.

Once you have finished your reasoning, you should choose exactly one admissible action for the current step and call the tool by presenting the function call within <tool_call> </tool_call> tags.

The results of the function calls will be given back to you after execution, and you can continue to call functions until you get the final answer for the user's query."""

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
    parser.add_argument("--data_file", default=None, help="Path to one parquet file for single-sample mode.")
    parser.add_argument(
        "--tool_config_path",
        default="examples/alfworld_multiturn/configs/tool_config/alfworld_tool_config.yaml",
    )
    parser.add_argument("--index", type=int, default=None, help="Use a fixed sample index instead of random pick.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_turns", type=int, default=50, help="Maximum assistant turns to debug.")
    parser.add_argument(
        "--memory_k",
        type=int,
        default=2,
        help="Number of most recent assistant/tool rounds visible to the model each turn.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use in eval-all mode. Each worker uses one GPU.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--show_rendered_prompt", action="store_true")
    parser.add_argument("--system_prompt_path", default=None, help="Optional text file to override system prompt.")
    parser.add_argument("--user_prompt_path", default=None, help="Optional text file to override user prompt.")
    parser.add_argument("--valid_seen_file", default=None, help="Evaluate the full valid_seen split.")
    parser.add_argument("--valid_unseen_file", default=None, help="Evaluate the full valid_unseen split.")
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=-1,
        help="Optional cap for faster eval-all runs. -1 means evaluate all samples.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/alfworld_debug_eval",
        help="Directory for split summaries and full trajectories in eval-all mode.",
    )
    parser.add_argument(
        "--print_all_trajectories",
        action="store_true",
        help="Also print every trajectory to stdout in eval-all mode.",
    )
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


def _load_rows(data_file: str) -> list[dict[str, Any]]:
    dataframe = pd.read_parquet(data_file)
    rows = []
    for _, series in dataframe.iterrows():
        row = series.to_dict()
        row["prompt"] = _normalize_nested(row.get("prompt"))
        row["extra_info"] = _normalize_nested(row.get("extra_info"))
        row["reward_model"] = _normalize_nested(row.get("reward_model"))
        _populate_fields_from_prompt(row)
        rows.append(row)
    return rows


def _maybe_truncate_rows(rows: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if max_samples is None or max_samples < 0:
        return rows
    return rows[:max_samples]


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


def _build_runtime_config_cache(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    runtime_configs = {}
    for row in rows:
        extra_info = row.get("extra_info") or {}
        tools_kwargs = extra_info.get("tools_kwargs") or {}
        session_init = ((tools_kwargs.get("alfworld_step") or {}).get("create_kwargs") or {}).get("session_init") or {}
        config_path = session_init.get("config_path")
        if not config_path or config_path in runtime_configs:
            continue
        runtime_configs[config_path] = apply_textworld_overrides(load_runtime_config(config_path))
    return runtime_configs


def _read_text_if_needed(path: str | None) -> str | None:
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def _load_model_and_tokenizer(args: argparse.Namespace, device: str):
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def _rebuild_args(args_dict: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(**args_dict)


def _build_system_prompt_text(tool_schemas: list[dict[str, Any]], override_text: str | None) -> str:
    template = override_text if override_text is not None else DEFAULT_SYSTEM_PROMPT_TEMPLATE
    return template.format(tool_schema=json.dumps(tool_schemas, ensure_ascii=False, indent=2))


def _initial_state_from_row(row: dict[str, Any]) -> dict[str, Any]:
    extra_info = row.get("extra_info") or {}
    task_metadata = extra_info.get("task_metadata") or {}
    return {
        "task_description": task_metadata.get("task_description", row.get("task_description", "")),
        "current_observation": row.get("current_observation", ""),
        "admissible_actions": list(row.get("admissible_actions", []) or []),
        "done": False,
        "success": False,
        "final_env_score": 0.0,
        "step_id": 0,
        "invalid_action_count": 0,
    }


def _current_state(agent_data: Any, fallback_state: dict[str, Any]) -> dict[str, Any]:
    extra_fields = getattr(agent_data, "extra_fields", {}) or {}
    state = dict(fallback_state)
    for key in (
        "task_description",
        "current_observation",
        "admissible_actions",
        "done",
        "success",
        "final_env_score",
        "step_id",
        "invalid_action_count",
    ):
        if key in extra_fields:
            state[key] = extra_fields[key]
    state["admissible_actions"] = list(state.get("admissible_actions", []) or [])
    return state


def _build_user_prompt_text(state: dict[str, Any], override_text: str | None) -> str:
    template = override_text if override_text is not None else DEFAULT_USER_PROMPT_TEMPLATE
    return template.format(
        task_description=state.get("task_description", ""),
        current_observation=state.get("current_observation", ""),
        admissible_actions=render_actions_json(state.get("admissible_actions", [])),
    )


def _looks_like_malformed_tool_call(raw_output: str) -> bool:
    text = raw_output or ""
    if "<tool_call>" in text or "</tool_call>" in text:
        return True
    if '"name"' in text and "alfworld_step" in text:
        return True
    return False


def _format_tool_call_error_response(state: dict[str, Any] | None = None) -> str:
    admissible_actions = []
    current_observation = ""
    if state is not None:
        admissible_actions = state.get("admissible_actions", []) or []
        current_observation = state.get("current_observation", "")
    return (
        "Invalid tool call format.\n\n"
        "If you want to call a tool, you must output exactly one complete tool call wrapped in "
        "<tool_call> and </tool_call> tags.\n\n"
        "Correct format:\n"
        "<tool_call>\n"
        '{"name": "alfworld_step", "arguments": {"action": "<one admissible action>"}}\n'
        "</tool_call>\n\n"
        f"Current admissible actions are:\n{render_actions_json(admissible_actions)}\n\n"
        f"Current observation is: {current_observation}"
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
    del device
    model_inputs = tokenizer(prompt_text, return_tensors="pt")
    input_device = model.get_input_embeddings().weight.device
    model_inputs = {key: value.to(input_device) for key, value in model_inputs.items()}

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        outputs = model.generate(**model_inputs, **generate_kwargs)

    prompt_len = int(model_inputs["input_ids"].shape[1])
    new_token_ids = outputs[0, prompt_len:].tolist()
    raw_output = tokenizer.decode(new_token_ids, skip_special_tokens=False)
    return new_token_ids, raw_output


def _recent_visible_messages(round_history: list[tuple[dict[str, Any], dict[str, Any]]], memory_k: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for assistant_message, tool_message in round_history[-memory_k:]:
        messages.append(dict(assistant_message))
        messages.append(dict(tool_message))
    return messages


def _build_visible_messages(
    *,
    system_prompt_text: str,
    user_prompt_text: str,
    round_history: list[tuple[dict[str, Any], dict[str, Any]]],
    memory_k: int,
) -> list[dict[str, Any]]:
    messages = [{"role": "system", "content": system_prompt_text}]
    messages.extend(_recent_visible_messages(round_history, memory_k))
    messages.append({"role": "user", "content": user_prompt_text})
    return messages


def _serialize_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    serialized = []
    for tool_call in tool_calls:
        serialized.append({"name": tool_call.name, "arguments": tool_call.arguments})
    return serialized


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _print_state(label: str, state: dict[str, Any]) -> None:
    print(label)
    print(f"task_description: {state.get('task_description', '')}")
    print(f"current_observation: {state.get('current_observation', '')}")
    print(f"admissible_actions: {render_actions_json(state.get('admissible_actions', []))}")
    print(
        "done: {done} | success: {success} | score: {score} | step_id: {step_id} | invalid_action_count: {invalid}".format(
            done=state.get("done", False),
            success=state.get("success", False),
            score=state.get("final_env_score", 0.0),
            step_id=state.get("step_id", 0),
            invalid=state.get("invalid_action_count", 0),
        )
    )


def _print_full_trajectory(trajectory: list[dict[str, Any]]) -> None:
    print("=" * 80)
    print("Full trajectory replay:")
    for record in trajectory:
        turn = record["turn"]
        print("=" * 80)
        print(f"Turn {turn}")
        _print_state("State before turn:", record["state_before"])
        print("-" * 80)
        print("Assistant output:")
        print(record["assistant_output"])
        print("-" * 80)
        print(
            f"Contains <think>: {record['contains_think']} | Contains <tool_call>: {record['contains_tool_call']} | Parsed tool calls: {record['parsed_tool_call_count']}"
        )
        if record["parsed_content"]:
            print("Parsed content:")
            print(record["parsed_content"])
        if record["tool_call"] is not None:
            print("Tool call:")
            print(json.dumps(record["tool_call"], ensure_ascii=False))
            print(f"Tool reward: {record['tool_reward']}")
            print(f"Tool metrics: {json.dumps(record['tool_metrics'], ensure_ascii=False)}")
            print("Tool response:")
            print(record["tool_response"])
            _print_state("State after tool:", record["state_after"])
        else:
            print("No tool executed on this turn.")
        print(f"Termination reason after turn: {record['termination_reason']}")


def _task_type_from_row(row: dict[str, Any]) -> str:
    extra_info = row.get("extra_info") or {}
    task_metadata = extra_info.get("task_metadata") or {}
    task_type = task_metadata.get("task_type")
    return str(task_type) if task_type else "unknown"


def _init_fast_episode_state(row: dict[str, Any], runtime_configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    extra_info = row.get("extra_info") or {}
    tools_kwargs = extra_info.get("tools_kwargs") or {}
    session_init = ((tools_kwargs.get("alfworld_step") or {}).get("create_kwargs") or {}).get("session_init") or {}
    if not session_init:
        raise ValueError("extra_info.tools_kwargs.alfworld_step.create_kwargs.session_init is required")

    config_path = session_init["config_path"]
    runtime_config = runtime_configs.get(config_path)
    if runtime_config is None:
        runtime_config = apply_textworld_overrides(load_runtime_config(config_path))
        runtime_configs[config_path] = runtime_config

    env, current_observation, admissible_actions, task_description = build_single_game_env_fast(
        runtime_config,
        game_file=session_init["game_file"],
        train_eval=session_init.get("train_eval", "train"),
        task_description=session_init.get("task_description", ""),
    )
    return {
        "env": env,
        "task_description": task_description,
        "current_observation": current_observation,
        "admissible_actions": admissible_actions,
        "final_env_score": 0.0,
        "step_id": 0,
        "done": False,
        "success": False,
        "invalid_action_count": 0,
        "session_init": session_init,
    }


def _snapshot_fast_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_description": state["task_description"],
        "current_observation": state["current_observation"],
        "admissible_actions": list(state["admissible_actions"]),
        "final_env_score": float(state["final_env_score"]),
        "step_id": int(state["step_id"]),
        "done": bool(state["done"]),
        "success": bool(state["success"]),
        "invalid_action_count": int(state["invalid_action_count"]),
        "session_init": dict(state["session_init"]),
    }


def _execute_fast_tool_call(tool_args: dict[str, Any], episode_state: dict[str, Any]) -> tuple[str, float, dict[str, Any]]:
    raw_action = tool_args.get("action")
    display_action = "" if raw_action is None else str(raw_action)

    if episode_state["done"]:
        return (
            format_tool_response(
                action=display_action,
                current_observation=f"The episode is already finished.\n\n{episode_state['current_observation']}",
                admissible_actions=episode_state["admissible_actions"],
                done=episode_state["done"],
                success=episode_state["success"],
                score=episode_state["final_env_score"],
                step_id=episode_state["step_id"],
            ),
            0.0,
            _snapshot_fast_state(episode_state),
        )

    normalized_map = admissible_action_map(episode_state["admissible_actions"])
    invalid = False
    if raw_action is None or not isinstance(raw_action, str):
        invalid = True
    else:
        normalized_action = normalize_action(raw_action)
        if not normalized_action or normalized_action not in normalized_map:
            invalid = True

    if invalid:
        episode_state["invalid_action_count"] += 1
        return (
            format_tool_response(
                action=display_action,
                current_observation=(
                    "Invalid action. Choose exactly one action from the admissible actions list.\n\n"
                    f"{episode_state['current_observation']}"
                ),
                admissible_actions=episode_state["admissible_actions"],
                done=episode_state["done"],
                success=episode_state["success"],
                score=episode_state["final_env_score"],
                step_id=episode_state["step_id"],
            ),
            -0.1,
            _snapshot_fast_state(episode_state),
        )

    action = normalized_map[normalize_action(raw_action)]
    observations, scores, dones, infos = episode_state["env"].step([action])
    stepped_observation = str(first_batch_item(observations)).strip()
    _, episode_state["current_observation"] = extract_task_and_observation(stepped_observation)
    admissible_commands = (
        infos.get("admissible_commands", [episode_state["admissible_actions"]])
        if isinstance(infos, dict)
        else [episode_state["admissible_actions"]]
    )
    episode_state["admissible_actions"] = normalize_action_list(first_batch_item(admissible_commands))
    episode_state["final_env_score"] = float(first_batch_item(scores))
    episode_state["done"] = bool(first_batch_item(dones))
    won = infos.get("won", [False]) if isinstance(infos, dict) else [False]
    episode_state["success"] = bool(first_batch_item(won)) if episode_state["done"] else False
    episode_state["step_id"] += 1

    return (
        format_tool_response(
            action=action,
            current_observation=episode_state["current_observation"],
            admissible_actions=episode_state["admissible_actions"],
            done=episode_state["done"],
            success=episode_state["success"],
            score=episode_state["final_env_score"],
            step_id=episode_state["step_id"],
        ),
        0.0,
        _snapshot_fast_state(episode_state),
    )


def _run_single_episode(
    *,
    row: dict[str, Any],
    sample_index: int,
    args: argparse.Namespace,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    parser: Any,
    tool_schemas: list[dict[str, Any]],
    runtime_configs: dict[str, dict[str, Any]],
    system_prompt_text: str,
    user_prompt_override: str | None,
    print_live: bool,
) -> dict[str, Any]:
    extra_info = row.get("extra_info") or {}
    agent_data = SimpleNamespace(
        request_id=uuid4().hex,
        tools_kwargs=extra_info.get("tools_kwargs", {}),
        extra_fields={},
    )
    initial_state = _initial_state_from_row(row)
    round_history: list[tuple[dict[str, Any], dict[str, Any]]] = []
    full_round_history: list[tuple[dict[str, Any], dict[str, Any]]] = []
    trajectory: list[dict[str, Any]] = []
    episode_state = _init_fast_episode_state(row, runtime_configs)
    agent_data.extra_fields.update(_snapshot_fast_state(episode_state))

    if print_live:
        print("=" * 80)
        print(f"Sample index: {sample_index}")
        print(f"Data source: {row.get('data_source')}")
        print(f"Task type: {_task_type_from_row(row)}")
        print(f"Dual-memory window k: {args.memory_k}")
        print(f"Max turns: {args.max_turns}")
        print("=" * 80)
        print("System prompt:")
        print(system_prompt_text)

    termination_reason = "max_turns_reached"
    for turn in range(1, args.max_turns + 1):
        state_before = _current_state(agent_data, initial_state)
        user_prompt_text = _build_user_prompt_text(state_before, user_prompt_override)
        visible_messages = _build_visible_messages(
            system_prompt_text=system_prompt_text,
            user_prompt_text=user_prompt_text,
            round_history=round_history,
            memory_k=args.memory_k,
        )
        prompt_text = _build_prompt_text(tokenizer, visible_messages, tool_schemas)

        if print_live:
            print("=" * 80)
            print(f"Turn {turn} visible context:")
            print(
                f"visible_history_rounds: {min(len(round_history), args.memory_k)} / full_history_rounds: {len(full_round_history)}"
            )
            _print_state("Current task state:", state_before)
            print("-" * 80)
            print("Current user prompt:")
            print(user_prompt_text)
        if print_live and args.show_rendered_prompt:
            print(f"Rendered model prompt before turn {turn}:")
            print(prompt_text)

        new_token_ids, raw_output = _generate_one_turn(model, tokenizer, prompt_text, device, args)
        parsed_content, tool_calls = asyncio.run(parser.extract_tool_calls(new_token_ids, tool_schemas))

        if print_live:
            print("=" * 80)
            print(f"Turn {turn} raw model output:")
            print(raw_output)
            print("-" * 80)
            print(f"Contains <think>: {'<think>' in raw_output and '</think>' in raw_output}")
            print(f"Contains <tool_call>: {'<tool_call>' in raw_output and '</tool_call>' in raw_output}")
            print(f"Parsed tool calls: {len(tool_calls)}")
        assistant_message = {"role": "assistant", "content": raw_output}
        turn_record = {
            "turn": turn,
            "state_before": state_before,
            "visible_messages": visible_messages,
            "assistant_output": raw_output,
            "contains_think": "<think>" in raw_output and "</think>" in raw_output,
            "contains_tool_call": "<tool_call>" in raw_output and "</tool_call>" in raw_output,
            "parsed_tool_call_count": len(tool_calls),
            "parsed_tool_calls": _serialize_tool_calls(tool_calls),
            "parsed_content": parsed_content,
            "tool_call": None,
            "tool_reward": None,
            "tool_metrics": None,
            "tool_response": None,
            "state_after": None,
            "termination_reason": None,
        }

        if not tool_calls:
            if _looks_like_malformed_tool_call(raw_output):
                if print_live:
                    print("Malformed tool call detected. Returning a synthetic tool response and continuing.")
                tool_message = {"role": "tool", "content": _format_tool_call_error_response(state_before)}
                full_round_history.append((assistant_message, tool_message))
                round_history = full_round_history[-args.memory_k :]
                turn_record["tool_response"] = tool_message["content"]
                turn_record["tool_metrics"] = {"synthetic_tool_response": "invalid_tool_call_format"}
                turn_record["state_after"] = state_before
                turn_record["termination_reason"] = "tool_call_format_error_continue"
                trajectory.append(turn_record)
                continue

            if print_live:
                print("Parsed content without tool calls:")
                print(parsed_content)
            termination_reason = "no_tool_call"
            turn_record["termination_reason"] = termination_reason
            trajectory.append(turn_record)
            break

        first_tool_call = tool_calls[0]
        if print_live:
            print(f"[tool_call] name={first_tool_call.name} arguments={first_tool_call.arguments}")
        turn_record["tool_call"] = {"name": first_tool_call.name, "arguments": first_tool_call.arguments}
        try:
            tool_args = json.loads(first_tool_call.arguments)
        except json.JSONDecodeError:
            if print_live:
                print("Tool arguments are not valid JSON. Returning a synthetic tool response and continuing.")
            tool_message = {"role": "tool", "content": _format_tool_call_error_response(state_before)}
            full_round_history.append((assistant_message, tool_message))
            round_history = full_round_history[-args.memory_k :]
            turn_record["tool_response"] = tool_message["content"]
            turn_record["tool_metrics"] = {"synthetic_tool_response": "invalid_tool_arguments_json"}
            turn_record["state_after"] = state_before
            turn_record["termination_reason"] = "tool_call_argument_json_error_continue"
            trajectory.append(turn_record)
            continue

        tool_response, tool_reward, tool_metrics = _execute_fast_tool_call(tool_args, episode_state)
        agent_data.extra_fields.update(tool_metrics)
        if print_live:
            print("-" * 80)
            print(f"Tool reward: {tool_reward}")
            print(f"Tool metrics: {tool_metrics}")
            print("Tool response:")
            print(tool_response)
        tool_message = {"role": "tool", "content": tool_response}
        full_round_history.append((assistant_message, tool_message))
        round_history = full_round_history[-args.memory_k :]
        state_after = _current_state(agent_data, state_before)
        turn_record["tool_reward"] = tool_reward
        turn_record["tool_metrics"] = tool_metrics
        turn_record["tool_response"] = tool_response
        turn_record["state_after"] = state_after

        if agent_data.extra_fields.get("done"):
            if print_live:
                print("-" * 80)
                print("Episode marked done by tool session.")
            termination_reason = "episode_done"
            turn_record["termination_reason"] = termination_reason
            trajectory.append(turn_record)
            break

        termination_reason = "continue"
        turn_record["termination_reason"] = termination_reason
        trajectory.append(turn_record)

    if trajectory and trajectory[-1]["termination_reason"] == "continue":
        trajectory[-1]["termination_reason"] = "max_turns_reached"
        termination_reason = "max_turns_reached"

    if trajectory:
        final_state = trajectory[-1]["state_after"] or trajectory[-1]["state_before"]
    else:
        final_state = _current_state(agent_data, initial_state)
    episode_result = {
        "sample_index": sample_index,
        "data_source": row.get("data_source"),
        "task_type": _task_type_from_row(row),
        "termination_reason": termination_reason,
        "final_state": final_state,
        "success": bool(final_state.get("success", False)),
        "final_env_score": float(final_state.get("final_env_score", 0.0)),
        "invalid_action_count": int(final_state.get("invalid_action_count", 0)),
        "num_turns": len(trajectory),
        "trajectory": trajectory,
    }
    if print_live:
        print("=" * 80)
        print(f"Run finished with termination_reason={termination_reason}")
        _print_state("Final state:", final_state)
        _print_full_trajectory(trajectory)
    close_env_quietly(episode_state.get("env"))
    return _json_ready(episode_result)


def _empty_split_stats() -> dict[str, dict[str, int]]:
    return {task_type: {"success": 0, "total": 0} for task_type in TASK_TYPES.values()}


def _summarize_split(split_name: str, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    task_stats = _empty_split_stats()
    total_success = 0
    for episode in episodes:
        task_type = episode.get("task_type", "unknown")
        task_stats.setdefault(task_type, {"success": 0, "total": 0})
        task_stats[task_type]["total"] += 1
        if episode.get("success", False):
            task_stats[task_type]["success"] += 1
            total_success += 1

    per_task = {}
    for task_type, counts in task_stats.items():
        total = counts["total"]
        per_task[task_type] = {
            "success": counts["success"],
            "total": total,
            "success_rate": (counts["success"] / total) if total else 0.0,
        }

    total_episodes = len(episodes)
    return {
        "split": split_name,
        "total_success": total_success,
        "total_episodes": total_episodes,
        "overall_success_rate": (total_success / total_episodes) if total_episodes else 0.0,
        "per_task_type": per_task,
    }


def _print_split_summary(summary: dict[str, Any]) -> None:
    print("=" * 80)
    print(f"Split summary: {summary['split']}")
    print(
        "overall_success_rate: {rate:.4f} ({succ}/{total})".format(
            rate=summary["overall_success_rate"],
            succ=summary["total_success"],
            total=summary["total_episodes"],
        )
    )
    print("per_task_type:")
    for task_type, stats in summary["per_task_type"].items():
        print(
            "  {task}: {rate:.4f} ({succ}/{total})".format(
                task=task_type,
                rate=stats["success_rate"],
                succ=stats["success"],
                total=stats["total"],
            )
        )


def _evaluate_split(
    *,
    split_name: str,
    data_file: str,
    args: argparse.Namespace,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    parser: Any,
    tool_schemas: list[dict[str, Any]],
    runtime_configs: dict[str, dict[str, Any]],
    system_prompt_text: str,
    user_prompt_override: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _maybe_truncate_rows(_load_rows(data_file), args.max_samples_per_split)
    episodes = []
    print("=" * 80)
    print(f"Evaluating split={split_name} from {data_file}")
    print(f"num_samples={len(rows)}")
    for sample_index, row in enumerate(rows):
        print("-" * 80)
        print(f"[{split_name}] sample {sample_index + 1}/{len(rows)} | task_type={_task_type_from_row(row)}")
        episode = _run_single_episode(
            row=row,
            sample_index=sample_index,
            args=args,
            model=model,
            tokenizer=tokenizer,
            device=device,
            parser=parser,
            tool_schemas=tool_schemas,
            runtime_configs=runtime_configs,
            system_prompt_text=system_prompt_text,
            user_prompt_override=user_prompt_override,
            print_live=False,
        )
        episodes.append(episode)
        print(
            "[{split}] result success={success} termination={term} num_turns={turns} invalid_action_count={invalid}".format(
                split=split_name,
                success=episode["success"],
                term=episode["termination_reason"],
                turns=episode["num_turns"],
                invalid=episode["invalid_action_count"],
            )
        )
        if args.print_all_trajectories:
            _print_full_trajectory(episode["trajectory"])
    summary = _summarize_split(split_name, episodes)
    _print_split_summary(summary)
    return summary, episodes


def _split_indexed_rows(indexed_rows: list[tuple[int, dict[str, Any]]], num_shards: int) -> list[list[tuple[int, dict[str, Any]]]]:
    shards = [[] for _ in range(num_shards)]
    for i, item in enumerate(indexed_rows):
        shards[i % num_shards].append(item)
    return [shard for shard in shards if shard]


def _evaluate_rows_worker(
    worker_id: int,
    gpu_id: int,
    split_name: str,
    indexed_rows: list[tuple[int, dict[str, Any]]],
    args_dict: dict[str, Any],
    tool_schemas: list[dict[str, Any]],
    runtime_configs: dict[str, dict[str, Any]],
    system_prompt_text: str,
    user_prompt_override: str | None,
) -> list[dict[str, Any]]:
    args = _rebuild_args(args_dict)
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    model, tokenizer = _load_model_and_tokenizer(args, device)
    parser = ToolParser.get_tool_parser("hermes", tokenizer)
    episodes = []
    try:
        for sample_index, row in indexed_rows:
            episode = _run_single_episode(
                row=row,
                sample_index=sample_index,
                args=args,
                model=model,
                tokenizer=tokenizer,
                device=device,
                parser=parser,
                tool_schemas=tool_schemas,
                runtime_configs=runtime_configs,
                system_prompt_text=system_prompt_text,
                user_prompt_override=user_prompt_override,
                print_live=False,
            )
            episodes.append(episode)
            print(
                "[worker {worker} gpu {gpu}] [{split}] sample {idx} | success={success} termination={term} num_turns={turns}".format(
                    worker=worker_id,
                    gpu=gpu_id,
                    split=split_name,
                    idx=sample_index + 1,
                    success=episode["success"],
                    term=episode["termination_reason"],
                    turns=episode["num_turns"],
                ),
                flush=True,
            )
        return episodes
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _evaluate_split_multi_gpu(
    *,
    split_name: str,
    data_file: str,
    args: argparse.Namespace,
    tool_schemas: list[dict[str, Any]],
    runtime_configs: dict[str, dict[str, Any]],
    system_prompt_text: str,
    user_prompt_override: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _maybe_truncate_rows(_load_rows(data_file), args.max_samples_per_split)
    print("=" * 80)
    print(f"Evaluating split={split_name} from {data_file}")
    print(f"num_samples={len(rows)}")

    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    worker_count = min(args.num_gpus, visible_gpus if visible_gpus > 0 else 1, len(rows))
    if worker_count <= 1:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model, tokenizer = _load_model_and_tokenizer(args, device)
        parser = ToolParser.get_tool_parser("hermes", tokenizer)
        try:
            return _evaluate_split(
                split_name=split_name,
                data_file=data_file,
                args=args,
                model=model,
                tokenizer=tokenizer,
                device=device,
                parser=parser,
                tool_schemas=tool_schemas,
                runtime_configs=runtime_configs,
                system_prompt_text=system_prompt_text,
                user_prompt_override=user_prompt_override,
            )
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    indexed_rows = list(enumerate(rows))
    shards = _split_indexed_rows(indexed_rows, worker_count)
    args_dict = vars(args).copy()
    episodes: list[dict[str, Any]] = []
    mp_context = mp.get_context("spawn")

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(shards), mp_context=mp_context) as executor:
        futures = []
        for worker_id, shard in enumerate(shards):
            futures.append(
                executor.submit(
                    _evaluate_rows_worker,
                    worker_id,
                    worker_id,
                    split_name,
                    shard,
                    args_dict,
                    tool_schemas,
                    runtime_configs,
                    system_prompt_text,
                    user_prompt_override,
                )
            )

        for future in concurrent.futures.as_completed(futures):
            episodes.extend(future.result())

    episodes.sort(key=lambda episode: episode["sample_index"])
    if args.print_all_trajectories:
        for episode in episodes:
            _print_full_trajectory(episode["trajectory"])
    summary = _summarize_split(split_name, episodes)
    _print_split_summary(summary)
    return summary, episodes


def main() -> None:
    args = parse_args()

    tool_schemas, _ = _load_tools(args.tool_config_path)
    system_prompt_override = _read_text_if_needed(args.system_prompt_path)
    user_prompt_override = _read_text_if_needed(args.user_prompt_path)
    system_prompt_text = _build_system_prompt_text(tool_schemas, system_prompt_override)

    if args.valid_seen_file or args.valid_unseen_file:
        os.makedirs(args.output_dir, exist_ok=True)
        all_rows = []
        if args.valid_seen_file:
            all_rows.extend(_maybe_truncate_rows(_load_rows(args.valid_seen_file), args.max_samples_per_split))
        if args.valid_unseen_file:
            all_rows.extend(_maybe_truncate_rows(_load_rows(args.valid_unseen_file), args.max_samples_per_split))
        runtime_configs = _build_runtime_config_cache(all_rows)
        if args.valid_seen_file:
            seen_summary, seen_episodes = _evaluate_split_multi_gpu(
                split_name="valid_seen",
                data_file=args.valid_seen_file,
                args=args,
                tool_schemas=tool_schemas,
                runtime_configs=runtime_configs,
                system_prompt_text=system_prompt_text,
                user_prompt_override=user_prompt_override,
            )
            with open(os.path.join(args.output_dir, "valid_seen_summary.json"), "w", encoding="utf-8") as f:
                json.dump(seen_summary, f, ensure_ascii=False, indent=2)
            with open(os.path.join(args.output_dir, "valid_seen_trajectories.json"), "w", encoding="utf-8") as f:
                json.dump(seen_episodes, f, ensure_ascii=False, indent=2)

        if args.valid_unseen_file:
            unseen_summary, unseen_episodes = _evaluate_split_multi_gpu(
                split_name="valid_unseen",
                data_file=args.valid_unseen_file,
                args=args,
                tool_schemas=tool_schemas,
                runtime_configs=runtime_configs,
                system_prompt_text=system_prompt_text,
                user_prompt_override=user_prompt_override,
            )
            with open(os.path.join(args.output_dir, "valid_unseen_summary.json"), "w", encoding="utf-8") as f:
                json.dump(unseen_summary, f, ensure_ascii=False, indent=2)
            with open(os.path.join(args.output_dir, "valid_unseen_trajectories.json"), "w", encoding="utf-8") as f:
                json.dump(unseen_episodes, f, ensure_ascii=False, indent=2)
        print("=" * 80)
        print(f"Saved eval outputs to {os.path.abspath(args.output_dir)}")
        return

    if not args.data_file:
        raise ValueError("Single-sample mode requires --data_file, or use --valid_seen_file/--valid_unseen_file.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, tokenizer = _load_model_and_tokenizer(args, device)
    parser = ToolParser.get_tool_parser("hermes", tokenizer)
    sample_index, row = _pick_row(args.data_file, args.index, args.seed)
    runtime_configs = _build_runtime_config_cache([row])
    try:
        _run_single_episode(
            row=row,
            sample_index=sample_index,
            args=args,
            model=model,
            tokenizer=tokenizer,
            device=device,
            parser=parser,
            tool_schemas=tool_schemas,
            runtime_configs=runtime_configs,
            system_prompt_text=system_prompt_text,
            user_prompt_override=user_prompt_override,
            print_live=True,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
