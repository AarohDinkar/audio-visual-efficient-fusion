"""
Create deterministic train/test splits from a precomputed retrieval feature cache.

Splitting is done by video_id so no video appears in both train and test.
"""

import argparse
import json
import random
from pathlib import Path

import torch


def split_cache(
    cache_path: Path,
    train_output: Path,
    test_output: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict:
    payload = torch.load(cache_path, map_location="cpu")
    records = payload["records"]
    by_video = {}
    for record in records:
        by_video.setdefault(record["video_id"], []).append(record)

    video_ids = sorted(by_video)
    rng = random.Random(seed)
    rng.shuffle(video_ids)
    train_count = max(1, min(len(video_ids) - 1, round(len(video_ids) * train_ratio)))
    train_ids = set(video_ids[:train_count])
    test_ids = set(video_ids[train_count:])

    train_records = [record for record in records if record["video_id"] in train_ids]
    test_records = [record for record in records if record["video_id"] in test_ids]

    base_config = payload.get("config", {})
    split_config = {
        **base_config,
        "source_cache": str(cache_path),
        "train_ratio": train_ratio,
        "seed": seed,
        "split_by": "video_id",
        "source_records": len(records),
        "source_unique_videos": len(video_ids),
    }
    train_payload = {
        "config": {
            **split_config,
            "split": "train",
            "num_samples": len(train_records),
            "num_unique_videos": len(train_ids),
        },
        "records": train_records,
    }
    test_payload = {
        "config": {
            **split_config,
            "split": "test",
            "num_samples": len(test_records),
            "num_unique_videos": len(test_ids),
        },
        "records": test_records,
    }

    train_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_payload, train_output)
    torch.save(test_payload, test_output)

    summary = {
        "source": str(cache_path),
        "train_output": str(train_output),
        "test_output": str(test_output),
        "train_records": len(train_records),
        "test_records": len(test_records),
        "train_unique_videos": len(train_ids),
        "test_unique_videos": len(test_ids),
        "seed": seed,
        "train_ratio": train_ratio,
    }
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_cache(
        cache_path=args.cache,
        train_output=args.train_output,
        test_output=args.test_output,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
