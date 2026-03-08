"""
Create minimal sample data for testing without real videos.

Generates synthetic frames and audio, plus captions.json.
Useful for quick pipeline validation.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FRAMES_DIR, AUDIO_DIR, CAPTIONS_FILE, DATA_DIR, NUM_FRAMES, AUDIO_SAMPLE_RATE, AUDIO_MAX_DURATION


def create_sample_data(num_samples: int = 5):
    """Create synthetic frames, audio, and captions for testing."""
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    samples = []
    for i in range(num_samples):
        video_id = f"sample_{i}"
        # Synthetic frames [T, 3, 224, 224] in [0, 1]
        frames = torch.rand(NUM_FRAMES, 3, 224, 224)
        # Synthetic audio [samples]
        audio = torch.randn(int(AUDIO_MAX_DURATION * AUDIO_SAMPLE_RATE)) * 0.5
        frames_path = FRAMES_DIR / f"{video_id}.pt"
        audio_path = AUDIO_DIR / f"{video_id}.pt"
        torch.save(frames, frames_path)
        torch.save(audio, audio_path)
        caption = f"Sample video {i} with synthetic content."
        samples.append({
            "video_id": video_id,
            "frames_path": str(frames_path),
            "audio_path": str(audio_path),
            "caption": caption,
        })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CAPTIONS_FILE, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Created {num_samples} sample entries in {CAPTIONS_FILE}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=5)
    args = parser.parse_args()
    create_sample_data(args.num)
