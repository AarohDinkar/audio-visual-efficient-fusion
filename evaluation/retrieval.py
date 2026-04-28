"""
Audio-visual retrieval evaluation.

Computes video-to-caption Recall@K using CLIP visual embeddings for sampled
video frames and CLIP text embeddings for captions. This is a lightweight,
repeatable retrieval proxy for the research ablation table.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import CAPTIONS_FILE, VISION_ENCODER
from evaluation.metrics import recall_at_k
from scripts.validate_dataset import validate_captions


def load_samples(captions_path: Path, limit: Optional[int] = None) -> list[dict]:
    with open(captions_path) as f:
        samples = json.load(f)
    return samples[:limit] if limit else samples


def encode_video_frames(model: CLIPModel, frames: torch.Tensor, device: torch.device) -> torch.Tensor:
    frames = frames.to(device)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    with torch.no_grad():
        features = model.get_image_features(pixel_values=frames)
    return features.mean(dim=0)


def run_retrieval(
    captions_path: Path = CAPTIONS_FILE,
    limit: Optional[int] = 100,
    results_dir: Path = Path("results"),
    run_name: Optional[str] = None,
    validate_dataset: bool = True,
) -> dict:
    if validate_dataset:
        validate_captions(captions_path)
    samples = load_samples(captions_path, limit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(VISION_ENCODER).to(device)
    processor = CLIPProcessor.from_pretrained(VISION_ENCODER)
    model.eval()

    video_embeddings = []
    text_embeddings = []
    labels = []
    for sample in tqdm(samples, desc="Retrieval embeddings"):
        frames = torch.load(sample["frames_path"])
        video_embeddings.append(encode_video_frames(model, frames, device).cpu().numpy())

        text_inputs = processor(text=[sample["caption"]], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
        text_embeddings.append(text_features.squeeze(0).cpu().numpy())
        labels.append(sample.get("video_id", str(len(labels))))

    video_embeddings = np.stack(video_embeddings)
    text_embeddings = np.stack(text_embeddings)
    metrics = recall_at_k(
        query_embeddings=video_embeddings,
        gallery_embeddings=text_embeddings,
        query_labels=labels,
        gallery_labels=labels,
        k_values=[1, 5, 10],
    )
    result = {
        "run_name": run_name or f"retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "config": {
            "captions_path": str(captions_path),
            "limit": limit,
            "num_samples": len(samples),
            "encoder": VISION_ENCODER,
        },
        "metrics": metrics,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{result['run_name']}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=Path, default=CAPTIONS_FILE)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--skip-dataset-validation", action="store_true")
    args = parser.parse_args()
    run_retrieval(
        args.captions,
        args.limit,
        args.results_dir,
        args.run_name,
        validate_dataset=not args.skip_dataset_validation,
    )


if __name__ == "__main__":
    main()
