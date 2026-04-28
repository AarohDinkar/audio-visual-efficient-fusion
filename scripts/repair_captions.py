"""
Repair data/captions.json by mapping real MSR-VTT annotations to preprocessed
frame/audio tensors.

Expected output format:
[
  {"video_id": "...", "frames_path": "...", "audio_path": "...", "caption": "..."}
]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AUDIO_DIR, CAPTIONS_FILE, DATA_DIR, FRAMES_DIR
from scripts.preprocess_video import load_captions
from scripts.validate_dataset import validate_captions


def find_annotation_json(data_dir: Path) -> Path | None:
    preferred = [
        "train_val_videodatainfo.json",
        "MSRVTT_data.json",
        "videodatainfo.json",
    ]
    for name in preferred:
        path = data_dir / name
        if path.exists():
            return path
    for path in data_dir.rglob("*.json"):
        if path.name != CAPTIONS_FILE.name:
            return path
    return None


def repair_captions(
    annotations: Path | None = None,
    output: Path = CAPTIONS_FILE,
    frames_dir: Path = FRAMES_DIR,
    audio_dir: Path = AUDIO_DIR,
    min_unique_captions: int = 10,
) -> dict:
    annotations = annotations or find_annotation_json(DATA_DIR)
    if annotations is None or not annotations.exists():
        raise FileNotFoundError(
            "No annotation JSON found. Run scripts/download_dataset.py or pass --annotations."
        )

    caption_map = load_captions(annotations)
    if not caption_map:
        raise ValueError(f"No captions could be parsed from {annotations}")

    frame_files = sorted(frames_dir.glob("*.pt"))
    samples = []
    unmatched = []
    for frame_path in frame_files:
        video_id = frame_path.stem
        audio_path = audio_dir / f"{video_id}.pt"
        caption = caption_map.get(video_id)
        if caption is None:
            unmatched.append(video_id)
            continue
        samples.append({
            "video_id": video_id,
            "frames_path": str(frame_path.resolve()),
            "audio_path": str(audio_path.resolve()),
            "caption": caption,
        })

    if not samples:
        raise ValueError(
            f"No preprocessed frame/audio ids matched annotations in {annotations}."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(samples, f, indent=2)

    summary = validate_captions(output, min_unique_captions=min_unique_captions)
    summary.update({
        "annotations": str(annotations),
        "output": str(output),
        "unmatched_preprocessed": len(unmatched),
    })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=CAPTIONS_FILE)
    parser.add_argument("--min-unique-captions", type=int, default=10)
    args = parser.parse_args()
    summary = repair_captions(
        annotations=args.annotations,
        output=args.output,
        min_unique_captions=args.min_unique_captions,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
