import argparse
import concurrent.futures
import os
import shutil

import datasets
from tqdm import tqdm

from verl_ext.alfworld.dataset import build_dataset_row
from verl_ext.alfworld.utils import (
    apply_textworld_overrides,
    build_single_game_env_fast,
    close_env_quietly,
    discover_game_files,
    get_task_description_from_traj_data,
    load_runtime_config,
)


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
    parser.add_argument("--num_workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--local_dir", default=None, help="Deprecated alias of --local_save_dir.")
    parser.add_argument("--local_save_dir", default="~/data/alfworld_multiturn")
    parser.add_argument("--hdfs_dir", default=None)
    return parser.parse_args()


def get_copy_utils():
    """Import HDFS helpers lazily so local preprocess does not pull the whole verl package at startup."""
    if os.name == "nt":
        def makedirs(path: str) -> None:
            os.makedirs(path, exist_ok=True)

        def copy(src: str, dst: str) -> None:
            shutil.copytree(src, dst, dirs_exist_ok=True)

        return makedirs, copy

    try:
        from verl.utils.hdfs_io import copy, makedirs

        return makedirs, copy
    except Exception:
        def makedirs(path: str) -> None:
            os.makedirs(path, exist_ok=True)

        def copy(src: str, dst: str) -> None:
            shutil.copytree(src, dst, dirs_exist_ok=True)

        return makedirs, copy


def _build_single_row(index: int, row: dict, config_path: str, split: str, runtime_config: dict) -> dict:
    session_init = {
        "config_path": os.path.abspath(config_path),
        "train_eval": split,
        "game_file": row["game_file"],
        "task_description": get_task_description_from_traj_data(row["traj_data"]),
    }

    env = None
    try:
        env, current_observation, admissible_actions, task_description = build_single_game_env_fast(
            runtime_config,
            game_file=row["game_file"],
            train_eval=split,
            task_description=session_init["task_description"],
        )
    finally:
        close_env_quietly(env)

    task_metadata = {
        "game_file": row["game_file"],
        "traj_file": row["traj_file"],
        "task_type": row["task_type"],
        "task_id": row["task_id"],
        "train_eval": split,
    }
    return build_dataset_row(
        split=split,
        index=index,
        task_description=task_description,
        current_observation=current_observation,
        admissible_actions=admissible_actions,
        session_init=session_init,
        task_metadata=task_metadata,
    )


def build_split_dataset(config_path: str, split: str, max_samples: int, num_workers: int) -> datasets.Dataset:
    print(f"[alfworld preprocess] loading config for split={split}", flush=True)
    config = load_runtime_config(config_path)
    runtime_config = apply_textworld_overrides(config)
    print(f"[alfworld preprocess] discovering games for split={split}", flush=True)
    rows = discover_game_files(config, split)
    print(f"[alfworld preprocess] discovered {len(rows)} games for split={split}", flush=True)
    if max_samples > 0:
        rows = rows[:max_samples]
        print(f"[alfworld preprocess] truncated to {len(rows)} samples for split={split}", flush=True)

    print(
        f"[alfworld preprocess] building {len(rows)} rows for split={split} with num_workers={num_workers}",
        flush=True,
    )

    indexed_rows = list(enumerate(rows))
    dataset_rows = [None] * len(indexed_rows)

    if num_workers <= 1:
        for index, row in tqdm(indexed_rows, desc=f"build {split} parquet", dynamic_ncols=True):
            dataset_rows[index] = _build_single_row(index, row, config_path, split, runtime_config)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {
                executor.submit(_build_single_row, index, row, config_path, split, runtime_config): index
                for index, row in indexed_rows
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_index),
                total=len(future_to_index),
                desc=f"build {split} parquet",
                dynamic_ncols=True,
            ):
                index = future_to_index[future]
                dataset_rows[index] = future.result()

    return datasets.Dataset.from_list(dataset_rows)


if __name__ == "__main__":
    args = parse_args()
    local_save_dir = args.local_dir or args.local_save_dir
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    for split in args.splits:
        print(f"[alfworld preprocess] start split={split}", flush=True)
        dataset = build_split_dataset(args.config_path, split, args.max_samples_per_split, args.num_workers)
        output_path = os.path.join(local_save_dir, SPLIT_TO_FILENAME[split])
        print(f"[alfworld preprocess] writing {len(dataset)} rows to {output_path}", flush=True)
        dataset.to_parquet(output_path)
        print(f"[alfworld preprocess] finished split={split}", flush=True)

    if args.hdfs_dir is not None:
        makedirs, copy = get_copy_utils()
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
