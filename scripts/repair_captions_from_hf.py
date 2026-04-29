"""
Repair data/captions.json from the Hugging Face MSR-VTT metadata.

This keeps existing local videos, frame tensors, and audio tensors. It only
rebuilds the caption index by matching HF video_id values such as "video0" to:
  data/frames/video0.pt
  data/audio/video0.pt
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AUDIO_DIR, CAPTIONS_FILE, FRAMES_DIR
from scripts.validate_dataset import validate_captions


def choose_captions(captions: list[str], strategy: str, max_captions: int) -> list[str]:
    cleaned = [str(c).strip() for c in captions if str(c).strip()]
    if not cleaned:
        return []
    if strategy == "first":
        return [cleaned[0]]
    if strategy == "all":
        return cleaned[:max_captions] if max_captions > 0 else cleaned
    raise ValueError(f"Unsupported caption strategy: {strategy}")


def backup_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    shutil.copy2(path, backup)
    return backup


def repair_from_hf(
    dataset_name: str = "friedrichor/MSR-VTT",
    config_name: str = "train_7k",
    split: str = "train",
    output: Path = CAPTIONS_FILE,
    caption_strategy: str = "all",
    max_captions_per_video: int = 20,
    min_unique_captions: int = 10,
) -> dict:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, config_name, split=split)
    frame_ids = {path.stem for path in FRAMES_DIR.glob("*.pt")}
    audio_ids = {path.stem for path in AUDIO_DIR.glob("*.pt")}
    available_ids = frame_ids & audio_ids
    if not available_ids:
        raise ValueError("No matching frame/audio tensor ids found.")

    samples = []
    missing_local = []
    for row in dataset:
        video_id = str(row["video_id"])
        if video_id not in available_ids:
            missing_local.append(video_id)
            continue
        captions = row.get("caption", [])
        if isinstance(captions, str):
            captions = [captions]
        selected = choose_captions(
            captions,
            strategy=caption_strategy,
            max_captions=max_captions_per_video,
        )
        for idx, caption in enumerate(selected):
            samples.append({
                "video_id": video_id,
                "caption_id": f"{video_id}_cap{idx}",
                "frames_path": str((FRAMES_DIR / f"{video_id}.pt").resolve()),
                "audio_path": str((AUDIO_DIR / f"{video_id}.pt").resolve()),
                "caption": caption,
                "source": dataset_name,
                "split": config_name,
            })

    if not samples:
        raise ValueError(
            "HF metadata loaded, but no video_id matched local frame/audio tensors."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(output)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    summary = validate_captions(output, min_unique_captions=min_unique_captions)
    summary.update({
        "dataset_name": dataset_name,
        "config_name": config_name,
        "split": split,
        "caption_strategy": caption_strategy,
        "max_captions_per_video": max_captions_per_video,
        "matched_local_videos": len({sample["video_id"] for sample in samples}),
        "missing_local_videos": len(missing_local),
        "output": str(output),
        "backup": str(backup) if backup else None,
    })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="friedrichor/MSR-VTT")
    parser.add_argument("--config", default="train_7k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=CAPTIONS_FILE)
    parser.add_argument("--caption-strategy", choices=["first", "all"], default="all")
    parser.add_argument("--max-captions-per-video", type=int, default=20)
    parser.add_argument("--min-unique-captions", type=int, default=10)
    args = parser.parse_args()

    summary = repair_from_hf(
        dataset_name=args.dataset,
        config_name=args.config,
        split=args.split,
        output=args.output,
        caption_strategy=args.caption_strategy,
        max_captions_per_video=args.max_captions_per_video,
        min_unique_captions=args.min_unique_captions,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
