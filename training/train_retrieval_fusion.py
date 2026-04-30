"""
Train non-LLM audio-visual fusion for caption retrieval.

Input is a precomputed feature cache from scripts/precompute_retrieval_features.py.
Only the lightweight fusion/projection head is trained.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.retrieval_fusion import AudioVisualRetrievalFusion, contrastive_retrieval_loss


class RetrievalFeatureDataset(Dataset):
    def __init__(self, cache_path: Path):
        payload = torch.load(cache_path, map_location="cpu")
        self.config = payload.get("config", {})
        self.records = payload["records"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        return {
            "video_id": record["video_id"],
            "caption_id": record["caption_id"],
            "caption": record["caption"],
            "vision_embedding": record["vision_embedding"].float(),
            "audio_embedding": record["audio_embedding"].float(),
            "text_embedding": record["text_embedding"].float(),
        }


def collate_fn(batch):
    return {
        "video_ids": [item["video_id"] for item in batch],
        "caption_ids": [item["caption_id"] for item in batch],
        "captions": [item["caption"] for item in batch],
        "vision_embedding": torch.stack([item["vision_embedding"] for item in batch]),
        "audio_embedding": torch.stack([item["audio_embedding"] for item in batch]),
        "text_embedding": torch.stack([item["text_embedding"] for item in batch]),
    }


def train(
    cache_path: Path,
    output_dir: Path = Path("checkpoints"),
    results_dir: Path = Path("results"),
    fusion_type: str = "gated",
    modality: str = "audio_visual",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    temperature: float = 0.07,
    run_name: Optional[str] = None,
) -> dict:
    dataset = RetrievalFeatureDataset(cache_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    sample = dataset[0]
    model = AudioVisualRetrievalFusion(
        vision_dim=sample["vision_embedding"].numel(),
        audio_dim=sample["audio_embedding"].numel(),
        text_dim=sample["text_embedding"].numel(),
        fusion_type=fusion_type,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    run_name = run_name or f"retrieval_{fusion_type}_{modality}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "run_name": run_name,
        "config": {
            "cache_path": str(cache_path),
            "fusion_type": fusion_type,
            "modality": modality,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "temperature": temperature,
            "dataset_size": len(dataset),
            "feature_config": dataset.config,
        },
        "epochs": [],
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Retrieval epoch {epoch + 1}"):
            fused = model(
                batch["vision_embedding"],
                batch["audio_embedding"],
                modality=modality,
            )
            loss = contrastive_retrieval_loss(fused, batch["text_embedding"], temperature=temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(loader))
        metrics["epochs"].append({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")

    checkpoint_path = output_dir / f"{run_name}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": metrics["config"],
        },
        checkpoint_path,
    )
    metrics["checkpoint"] = str(checkpoint_path)
    results_path = results_dir / f"{run_name}_train.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {results_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--fusion-type", choices=["gated", "additive"], default="gated")
    parser.add_argument("--modality", choices=["audio_visual", "vision_only", "audio_only"], default="audio_visual")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    train(
        cache_path=args.cache,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        fusion_type=args.fusion_type,
        modality=args.modality,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
