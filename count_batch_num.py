#!/usr/bin/env python
"""
Count WOMD scenarios and batches for a given preprocessing configuration.

This script iterates through the Waymax dataloader once (without multiprocessing)
to report:
  * total number of batches yielded
  * total number of individual scenarios processed
  * batch-size statistics and histogram
  * number of unique scenario ids encountered

Usage example:
    python count_batch_num.py \
        --dataset-path "${WOMD_VAL_PATH}" \
        --batch-dims "[1,500]" \
        --config configs/simulate.yaml
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import List, Set

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
from omegaconf import OmegaConf

from waymax import dataloader
from waymax.config import DataFormat, DatasetConfig

from src.utils.utils import update_waymax_config


def parse_batch_dims(text: str) -> List[int]:
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1]
    if not stripped:
        return []
    return [int(token.strip()) for token in stripped.split(",") if token.strip()]


def build_dataset_config(cfg) -> DatasetConfig:
    cfg = update_waymax_config(cfg)
    cfg.waymax_conf.batch_dims = tuple(cfg.batch_dims)

    waymax_conf = OmegaConf.to_container(cfg.waymax_conf, resolve=True)
    waymax_conf.pop("env_conf", None)
    waymax_conf.pop("customized", None)

    if isinstance(waymax_conf.get("data_format"), str):
        waymax_conf["data_format"] = DataFormat(waymax_conf["data_format"])

    batch_dims = waymax_conf.get("batch_dims")
    if batch_dims is not None:
        waymax_conf["batch_dims"] = tuple(batch_dims)

    return DatasetConfig(**waymax_conf)


def count_batches(dataset_cfg: DatasetConfig):
    iterator = dataloader.simulator_state_generator(config=dataset_cfg)

    total_scenarios = 0
    batch_sizes: List[int] = []
    unique_ids: Set[str] = set()

    for batch in iterator:
        scenario_ids = np.array(batch._scenario_id).reshape(-1)
        batch_size = int(scenario_ids.size)
        batch_sizes.append(batch_size)
        total_scenarios += batch_size

        for sid in scenario_ids:
            value = np.array(sid).item()
            if isinstance(value, bytes):
                unique_ids.add(value.decode("utf-8"))
            else:
                unique_ids.add(str(value))

    return len(batch_sizes), total_scenarios, batch_sizes, unique_ids


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count WOMD batches.")
    parser.add_argument(
        "--config",
        default="configs/simulate.yaml",
        help="Hydra config used for preprocessing (defaults to simulate.yaml).",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Dataset path or sharded pattern (e.g., validation_tfexample.tfrecord@150).",
    )
    parser.add_argument(
        "--batch-dims",
        required=True,
        help="Batch dimensions in the form \"[devices, per_device_batch]\".",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the dataset (matches preprocessing default).",
    )
    parser.add_argument(
        "--drop-remainder",
        type=str,
        default="false",
        choices=("true", "false"),
        help="Whether to drop the final partial batch.",
    )
    parser.add_argument(
        "--distributed",
        type=str,
        default="true",
        choices=("true", "false"),
        help="Mirror ++waymax_conf.distributed flag.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        help="Override ++waymax_conf.num_shards (useful to lower parallelism when counting).",
    )
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)

    batch_dims = parse_batch_dims(args.batch_dims)
    cfg.batch_dims = batch_dims

    cfg.waymax_conf.path = args.dataset_path
    cfg.waymax_conf.repeat = args.repeat
    cfg.waymax_conf.drop_remainder = args.drop_remainder.lower() == "true"
    cfg.waymax_conf.distributed = args.distributed.lower() == "true"
    cfg.waymax_conf.customized = False
    if args.num_shards is not None:
        cfg.waymax_conf.num_shards = args.num_shards

    dataset_cfg = build_dataset_config(cfg)
    total_batches, total_scenarios, batch_sizes, unique_ids = count_batches(dataset_cfg)

    min_batch = min(batch_sizes) if batch_sizes else 0
    max_batch = max(batch_sizes) if batch_sizes else 0
    mean_batch = sum(batch_sizes) / total_batches if total_batches else 0.0

    print(f"Dataset path    : {args.dataset_path}")
    print(f"Batch dims      : {batch_dims}")
    print(f"Total batches   : {total_batches}")
    print(f"Total scenarios : {total_scenarios}")
    print(f"Unique scenarios: {len(unique_ids)}")
    print(f"Min batch size  : {min_batch}")
    print(f"Max batch size  : {max_batch}")
    print(f"Mean batch size : {mean_batch:.2f}")

    size_counter = Counter(batch_sizes)
    print("Batch size histogram:")
    for size, count in sorted(size_counter.items()):
        print(f"  {size:4d}: {count}")


if __name__ == "__main__":
    main()
