"""
Validate preprocessed audio-visual caption metadata.

Fails fast when captions are placeholders or paths are missing, because those
conditions make training/evaluation scientifically invalid.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CAPTIONS_FILE


PLACEHOLDER_CAPTIONS = {"a video.", "a video", "", "sample video"}


def is_placeholder(caption: str) -> bool:
    text = caption.strip().lower()
    return text in PLACEHOLDER_CAPTIONS or text.startswith("sample video")


def validate_captions(captions_path: Path = CAPTIONS_FILE, min_unique_captions: int = 10) -> dict:
    if not captions_path.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_path}")

    with open(captions_path) as f:
        samples = json.load(f)
    if not isinstance(samples, list) or not samples:
        raise ValueError("Captions file must contain a non-empty list of samples.")

    missing_frames = []
    missing_audio = []
    placeholders = []
    captions = []
    for sample in samples:
        caption = str(sample.get("caption", ""))
        captions.append(caption)
        if is_placeholder(caption):
            placeholders.append(sample.get("video_id", "unknown"))
        if not Path(sample.get("frames_path", "")).exists():
            missing_frames.append(sample.get("video_id", "unknown"))
        if not Path(sample.get("audio_path", "")).exists():
            missing_audio.append(sample.get("video_id", "unknown"))

    unique_captions = set(captions)
    summary = {
        "samples": len(samples),
        "unique_captions": len(unique_captions),
        "placeholder_count": len(placeholders),
        "missing_frames": len(missing_frames),
        "missing_audio": len(missing_audio),
        "top_captions": Counter(captions).most_common(10),
    }

    errors = []
    if len(unique_captions) < min_unique_captions:
        errors.append(f"Only {len(unique_captions)} unique captions; need at least {min_unique_captions}.")
    if placeholders:
        errors.append(f"{len(placeholders)} placeholder captions found.")
    if missing_frames:
        errors.append(f"{len(missing_frames)} frame tensor paths are missing.")
    if missing_audio:
        errors.append(f"{len(missing_audio)} audio tensor paths are missing.")

    if errors:
        raise ValueError(json.dumps({"summary": summary, "errors": errors}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=Path, default=CAPTIONS_FILE)
    parser.add_argument("--min-unique-captions", type=int, default=10)
    args = parser.parse_args()
    summary = validate_captions(args.captions, args.min_unique_captions)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
