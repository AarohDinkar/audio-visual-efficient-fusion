"""
Evaluate non-LLM fused audio-visual retrieval.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import recall_at_k
from models.retrieval_fusion import AudioVisualRetrievalFusion
from training.train_retrieval_fusion import RetrievalFeatureDataset


def evaluate(
    cache_path: Path,
    checkpoint_path: Optional[Path] = None,
    results_dir: Path = Path("results"),
    fusion_type: str = "gated",
    modality: str = "audio_visual",
    run_name: Optional[str] = None,
) -> dict:
    dataset = RetrievalFeatureDataset(cache_path)
    sample = dataset[0]
    model = AudioVisualRetrievalFusion(
        vision_dim=sample["vision_embedding"].numel(),
        audio_dim=sample["audio_embedding"].numel(),
        text_dim=sample["text_embedding"].numel(),
        fusion_type=fusion_type,
    )
    if checkpoint_path:
        payload = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(payload["model_state"])
    model.eval()

    fused_embeddings = []
    text_embeddings = []
    labels = []
    captions = []
    with torch.no_grad():
        for record in dataset:
            fused = model(
                record["vision_embedding"].unsqueeze(0),
                record["audio_embedding"].unsqueeze(0),
                modality=modality,
            ).squeeze(0)
            fused_embeddings.append(fused.numpy())
            text_embeddings.append(record["text_embedding"].numpy())
            labels.append(record["video_id"])
            captions.append(record["caption"])

    metrics = recall_at_k(
        query_embeddings=np.stack(fused_embeddings),
        gallery_embeddings=np.stack(text_embeddings),
        query_labels=labels,
        gallery_labels=labels,
        k_values=[1, 5, 10],
    )
    result = {
        "run_name": run_name or f"fusion_retrieval_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "config": {
            "cache_path": str(cache_path),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "fusion_type": fusion_type,
            "modality": modality,
            "num_samples": len(dataset),
            "feature_config": dataset.config,
        },
        "metrics": metrics,
        "examples": [
            {"video_id": labels[i], "caption": captions[i]}
            for i in range(min(5, len(labels)))
        ],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{result['run_name']}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--fusion-type", choices=["gated", "additive"], default="gated")
    parser.add_argument("--modality", choices=["audio_visual", "vision_only", "audio_only"], default="audio_visual")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    evaluate(
        cache_path=args.cache,
        checkpoint_path=args.checkpoint,
        results_dir=args.results_dir,
        fusion_type=args.fusion_type,
        modality=args.modality,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
