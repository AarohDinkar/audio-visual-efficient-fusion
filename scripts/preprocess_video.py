"""
Preprocess videos: extract frames and audio.

Steps:
  - Sample 8 frames per video
  - Resize frames to 224x224
  - Extract audio waveform using torchaudio
  - Normalize audio
  - Store in data/frames/, data/audio/, captions.json
"""

import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_DIR,
    VIDEOS_DIR,
    FRAMES_DIR,
    AUDIO_DIR,
    CAPTIONS_FILE,
    PROCESSED_DIR,
    NUM_FRAMES,
    FRAME_SIZE,
    AUDIO_SAMPLE_RATE,
    AUDIO_MAX_DURATION,
)


def extract_frames(video_path: Path, num_frames: int = NUM_FRAMES, size: int = FRAME_SIZE) -> torch.Tensor:
    """
    Extract uniformly sampled frames from video.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        size: Frame resize dimension (H, W)

    Returns:
        frames: [num_frames, 3, H, W] tensor, values in [0, 1]
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None

    # Uniform sampling
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # BGR -> RGB, resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frames.append(frame)
    cap.release()

    if len(frames) < num_frames // 2:
        return None
    # Pad if needed
    while len(frames) < num_frames:
        frames.append(frames[-1].clone())
    frames = torch.stack(frames[:num_frames])
    return frames


def extract_audio(video_path: Path, sample_rate: int = AUDIO_SAMPLE_RATE, max_duration: float = AUDIO_MAX_DURATION) -> torch.Tensor:
    """
    Extract audio waveform from video.

    Args:
        video_path: Path to video file
        sample_rate: Target sample rate (16kHz for Whisper)
        max_duration: Max duration in seconds

    Returns:
        waveform: [samples] tensor, normalized
    """
    try:
        waveform, sr = torchaudio.load(str(video_path))
        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        waveform = waveform.squeeze(0)
        # Trim to max duration
        max_samples = int(max_duration * sample_rate)
        if waveform.shape[0] > max_samples:
            waveform = waveform[:max_samples]
        # Pad if too short
        if waveform.shape[0] < max_samples:
            waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.shape[0]))
        # Normalize
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()
        return waveform
    except Exception:
        max_samples = int(max_duration * sample_rate)
        return torch.zeros(max_samples)  # Silent if extraction fails


def load_captions(captions_path: Path) -> dict:
    """Load captions from MSR-VTT format or custom JSON."""
    if not captions_path.exists():
        return {}
    with open(captions_path) as f:
        data = json.load(f)
    # MSR-VTT format: {"sentences": [{"video_id": "vid", "caption": "..."}]}
    if "sentences" in data:
        captions = {}
        for s in data["sentences"]:
            vid = s.get("video_id", "")
            cap = s.get("caption", s.get("sentence", s.get("sentences", "")))
            if vid not in captions:
                captions[vid] = []
            captions[vid].append(cap)
        return {k: v[0] if v else "" for k, v in captions.items()}
    # Format: {"videos": [{"video_id": "vid", "captions": [...]}]}
    if "videos" in data:
        return {v["video_id"]: v["captions"][0] if v.get("captions") else "" for v in data["videos"]}
    # Simple format: {"video_id": "caption"}
    return data


def find_videos(videos_dir: Path) -> list[Path]:
    """Find all video files."""
    exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    videos = []
    for p in videos_dir.rglob("*"):
        if p.suffix.lower() in exts:
            videos.append(p)
    return sorted(videos)


def preprocess_all(
    videos_dir: Path = VIDEOS_DIR,
    frames_dir: Path = FRAMES_DIR,
    audio_dir: Path = AUDIO_DIR,
    captions_path: Path = DATA_DIR / "train_val_videodatainfo.json",
    output_captions: Path = CAPTIONS_FILE,
    num_frames: int = NUM_FRAMES,
    limit: Optional[int] = None,
):
    """
    Preprocess all videos and save to disk.

    Output format per sample:
    {
        "video_id": str,
        "frames_path": str,
        "audio_path": str,
        "caption": str,
    }
    """
    videos_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(videos_dir)
    if not videos:
        print(f"No videos found in {videos_dir}")
        print("Place .mp4/.avi files in data/videos/ and run again.")
        return

    if limit:
        videos = videos[:limit]
    print(f"Processing {len(videos)} videos...")

    captions = load_captions(captions_path)
    samples = []

    for video_path in tqdm(videos, desc="Preprocessing"):
        video_id = video_path.stem
        caption = captions.get(video_id, "A video.")

        # Extract
        frames = extract_frames(video_path, num_frames=num_frames)
        audio = extract_audio(video_path)
        if frames is None:
            continue

        # Save
        frames_path = frames_dir / f"{video_id}.pt"
        audio_path = audio_dir / f"{video_id}.pt"
        torch.save(frames, frames_path)
        torch.save(audio, audio_path)

        samples.append({
            "video_id": video_id,
            "frames_path": str(frames_path),
            "audio_path": str(audio_path),
            "caption": caption,
        })

    # Save captions index
    with open(output_captions, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} samples to {output_captions}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos")
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--captions", type=Path, default=DATA_DIR / "train_val_videodatainfo.json")
    args = parser.parse_args()
    preprocess_all(limit=args.limit, captions_path=args.captions)


if __name__ == "__main__":
    main()
