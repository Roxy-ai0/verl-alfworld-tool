import argparse
import os
import shutil

import datasets

from verl_ext.alfworld.dataset import build_dataset_row
from verl_ext.alfworld.utils import (
    build_single_game_env,
    close_env_quietly,
    discover_game_files,
    get_task_description_from_traj_data,
    load_runtime_config,
)

try:
    from verl.utils.hdfs_io import copy, makedirs
except Exception:
    def makedirs(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def copy(src: str, dst: str) -> None:
        shutil.copytree(src, dst, dirs_exist_ok=True)


SPLIT_TO_FILENAME = {
    "train": "train.parquet",
    "eval_in_distribution": "valid_seen.parquet",
    "eval_out_of_distribution": "valid_unseen.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, help="Path to the ALFWorld base/eval config YAML.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "eval_in_distribution", "eval_out_of_distribution"],
        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
    )
    parser.add_argument("--max_samples_per_split", type=int, default=-1)
    parser.add_argument("--local_dir", default=None, help="Deprecated alias of --local_save_dir.")
    parser.add_argument("--local_save_dir", default="~/data/alfworld_multiturn")
    parser.add_argument("--hdfs_dir", default=None)
    return parser.parse_args()


def build_split_dataset(config_path: str, split: str, max_samples: int) -> datasets.Dataset:
    config = load_runtime_config(config_path)
    rows = discover_game_files(config, split)
    if max_samples > 0:
        rows = rows[:max_samples]

    dataset_rows = []
    for index, row in enumerate(rows):
        session_init = {
            "config_path": os.path.abspath(config_path),
            "train_eval": split,
            "game_file": row["game_file"],
            "task_description": get_task_description_from_traj_data(row["traj_data"]),
        }

        env = None
        try:
            env, current_observation, admissible_actions, task_description = build_single_game_env(session_init)
        finally:
            close_env_quietly(env)

        task_metadata = {
            "game_file": row["game_file"],
            "traj_file": row["traj_file"],
            "task_type": row["task_type"],
            "task_id": row["task_id"],
            "train_eval": split,
        }
        dataset_rows.append(
            build_dataset_row(
                split=split,
                index=index,
                task_description=task_description,
                current_observation=current_observation,
                admissible_actions=admissible_actions,
                session_init=session_init,
                task_metadata=task_metadata,
            )
        )

    return datasets.Dataset.from_list(dataset_rows)


if __name__ == "__main__":
    args = parse_args()
    local_save_dir = args.local_dir or args.local_save_dir
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    for split in args.splits:
        dataset = build_split_dataset(args.config_path, split, args.max_samples_per_split)
        output_path = os.path.join(local_save_dir, SPLIT_TO_FILENAME[split])
        dataset.to_parquet(output_path)

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
