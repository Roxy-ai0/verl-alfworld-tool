import json
import os
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}

TASK_PREFIX = "Your task is to: "


def ensure_alfworld_available() -> None:
    try:
        import alfworld  # noqa: F401
        import textworld  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ALFWorld TextWorld dependencies are required. Install `alfworld` and `textworld` first."
        ) from exc


def normalize_action(action: str) -> str:
    return " ".join(action.strip().split())


def normalize_action_list(actions: Iterable[Any]) -> list[str]:
    normalized = []
    for action in actions:
        text = str(action).strip()
        if text:
            normalized.append(normalize_action(text))
    return normalized


def admissible_action_map(actions: Iterable[Any]) -> dict[str, str]:
    mapping = {}
    for action in actions:
        text = str(action).strip()
        if not text:
            continue
        mapping.setdefault(normalize_action(text), text)
    return mapping


def render_actions_json(actions: Iterable[Any]) -> str:
    return json.dumps(list(actions), ensure_ascii=False)


def format_tool_response(
    *,
    action: str,
    current_observation: str,
    admissible_actions: Iterable[Any],
    done: bool,
    success: bool,
    score: float,
    step_id: int,
) -> str:
    return (
        f"Your previous action was: {action}\n\n"
        f"Your current observation is: {current_observation}\n\n"
        "Your admissible actions of the current situation are:\n"
        f"{render_actions_json(admissible_actions)}\n\n"
        f"done: {done}\n"
        f"success: {success}\n"
        f"score: {score}\n"
        f"step_id: {step_id}"
    )


def extract_task_and_observation(observation: str) -> tuple[str, str]:
    text = str(observation or "").strip()
    if TASK_PREFIX not in text:
        return "", text

    before, _, after = text.partition(TASK_PREFIX)
    task_description = after.strip()
    observation_without_task = before.strip()
    return task_description, observation_without_task


def first_batch_item(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0]
    try:
        return value[0]
    except Exception:
        return value


def load_runtime_config(config_path: str) -> dict[str, Any]:
    from omegaconf import OmegaConf

    if not config_path:
        raise ValueError("session_init.config_path is required")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"ALFWorld config file not found: {config_path}")

    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        raise TypeError(f"ALFWorld config must resolve to a dict, got {type(config)}")
    return config


def apply_textworld_overrides(config: dict[str, Any]) -> dict[str, Any]:
    runtime_config = deepcopy(config)
    runtime_config.setdefault("env", {})
    runtime_config.setdefault("general", {})
    runtime_config.setdefault("dagger", {}).setdefault("training", {})

    runtime_config["env"]["type"] = "AlfredTWEnv"
    runtime_config["env"]["domain_randomization"] = False
    runtime_config["env"]["goal_desc_human_anns_prob"] = 0.0
    runtime_config["general"]["training_method"] = "dagger"
    runtime_config["general"]["use_cuda"] = bool(runtime_config["general"].get("use_cuda", False))
    runtime_config["dagger"]["training"].setdefault("max_nb_steps_per_episode", 50)
    return runtime_config


def build_single_game_env(session_init: dict[str, Any]) -> tuple[Any, str, list[str], str]:
    ensure_alfworld_available()

    from alfworld.agents.environment import get_environment

    config_path = os.path.abspath(session_init["config_path"])
    game_file = os.path.abspath(session_init["game_file"])
    train_eval = session_init.get("train_eval", "train")

    runtime_config = apply_textworld_overrides(load_runtime_config(config_path))
    env_cls = get_environment(runtime_config["env"]["type"])
    env_builder = env_cls(runtime_config, train_eval=train_eval)
    env_builder.game_files = [game_file]
    env_builder.num_games = 1

    env = env_builder.init_env(batch_size=1)
    observations, infos = env.reset()
    observation = str(first_batch_item(observations)).strip()
    admissible = normalize_action_list(first_batch_item(infos["admissible_commands"]))
    task_description, current_observation = extract_task_and_observation(observation)
    if not task_description:
        task_description = str(session_init.get("task_description", "")).strip()
    return env, current_observation, admissible, task_description


def close_env_quietly(env: Any) -> None:
    if env is None:
        return
    for method_name in ("close",):
        method = getattr(env, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass


def split_to_data_path(config: dict[str, Any], train_eval: str) -> str:
    dataset = config["dataset"]
    if train_eval == "train":
        return os.path.expandvars(dataset["data_path"])
    if train_eval == "eval_in_distribution":
        return os.path.expandvars(dataset["eval_id_data_path"])
    if train_eval == "eval_out_of_distribution":
        return os.path.expandvars(dataset["eval_ood_data_path"])
    raise ValueError(f"Unsupported ALFWorld split: {train_eval}")


def resolve_task_types(config: dict[str, Any]) -> set[str]:
    task_type_ids = config.get("env", {}).get("task_types", [])
    return {TASK_TYPES[task_type_id] for task_type_id in task_type_ids if task_type_id in TASK_TYPES}


def discover_game_files(config: dict[str, Any], train_eval: str) -> list[dict[str, Any]]:
    data_path = split_to_data_path(config, train_eval)
    if not data_path or not os.path.isdir(data_path):
        raise FileNotFoundError(f"ALFWorld split path does not exist: {data_path}")

    allowed_task_types = resolve_task_types(config)
    rows = []
    for root, _, files in os.walk(data_path, topdown=False):
        if "traj_data.json" not in files:
            continue

        if "movable" in root or "Sliced" in root:
            continue

        traj_path = os.path.join(root, "traj_data.json")
        game_path = os.path.join(root, "game.tw-pddl")
        if not os.path.exists(game_path):
            continue

        with open(traj_path, encoding="utf-8") as f:
            traj_data = json.load(f)

        if allowed_task_types and traj_data.get("task_type") not in allowed_task_types:
            continue

        with open(game_path, encoding="utf-8") as f:
            game_data = json.load(f)
        if not game_data.get("solvable", False):
            continue

        rows.append(
            {
                "game_file": os.path.abspath(game_path),
                "traj_file": os.path.abspath(traj_path),
                "task_type": traj_data.get("task_type"),
                "task_id": traj_data.get("task_id"),
                "traj_data": traj_data,
            }
        )

    limit_key = "num_train_games" if train_eval == "train" else "num_eval_games"
    limit = int(config.get("dataset", {}).get(limit_key, -1))
    if limit > 0:
        rows = rows[:limit]
    return rows


def get_task_description_from_traj_data(traj_data: dict[str, Any]) -> str:
    template = traj_data.get("template", {})
    if template.get("task_desc"):
        return str(template["task_desc"]).strip()

    annotations = traj_data.get("turk_annotations", {}).get("anns", [])
    if annotations:
        task_desc = annotations[0].get("task_desc")
        if task_desc:
            return str(task_desc).strip()
    return ""
