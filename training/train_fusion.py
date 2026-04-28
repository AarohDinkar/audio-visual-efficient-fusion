"""
Training script for Audio-Visual Fusion model.

Trains the fusion layer + LLM projection while keeping encoders frozen.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CAPTIONS_FILE, LLM_MODEL
from scripts.validate_dataset import validate_captions


def contrastive_alignment_loss(
    vision_emb: torch.Tensor,
    audio_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE loss for paired audio/video embeddings."""
    vision = F.normalize(vision_emb.float(), dim=-1)
    audio = F.normalize(audio_emb.float(), dim=-1)
    logits = vision @ audio.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )


def apply_input_ablations(
    frames: torch.Tensor,
    audio: torch.Tensor,
    noise_std: float = 0.0,
    frame_dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply robustness ablations without changing tensor shapes."""
    if noise_std > 0:
        audio = audio + torch.randn_like(audio) * noise_std
    if frame_dropout > 0:
        keep = torch.rand(frames.shape[:2], device=frames.device) > frame_dropout
        keep = keep[:, :, None, None, None].to(frames.dtype)
        frames = frames * keep
    return frames, audio


class MSRVTTDataset(Dataset):
    """Dataset from preprocessed frames, audio, and captions."""

    def __init__(self, captions_path: Path, max_samples: Optional[int] = None):
        with open(captions_path) as f:
            self.samples = json.load(f)
        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frames = torch.load(s["frames_path"])
        audio = torch.load(s["audio_path"])
        caption = s["caption"]
        return {"frames": frames, "audio": audio, "caption": caption}


def collate_fn(batch):
    frames = torch.stack([b["frames"] for b in batch])
    audio = torch.stack([b["audio"] for b in batch])
    captions = [b["caption"] for b in batch]
    return {"frames": frames, "audio": audio, "captions": captions}


def train(
    captions_path: Path = CAPTIONS_FILE,
    output_dir: Path = Path("checkpoints"),
    results_dir: Path = Path("results"),
    fusion_type: str = "gated",
    modality: str = "audio_visual",
    alignment_loss_weight: float = 0.0,
    alignment_temperature: float = 0.07,
    noise_std: float = 0.0,
    frame_dropout: float = 0.0,
    run_name: Optional[str] = None,
    validate_dataset: bool = True,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    max_samples: Optional[int] = None,
):
    """Train the fusion model."""
    if not captions_path.exists():
        print(f"Captions not found: {captions_path}")
        print("Run preprocess_video.py first.")
        return
    if validate_dataset:
        validate_captions(captions_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from models.multimodal_model import AudioVisualCaptioner
    model = AudioVisualCaptioner(fusion_type=fusion_type, llm_name=LLM_MODEL)
    model = model.to(device)

    dataset = MSRVTTDataset(captions_path, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Only train fusion + projection
    optimizer = torch.optim.AdamW(
        list(model.fusion.parameters()) + list(model.audio_enc.projection.parameters()),
        lr=lr,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_name or (
        f"{fusion_type}_{modality}_align{alignment_loss_weight:g}_"
        f"noise{noise_std:g}_drop{frame_dropout:g}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_metrics = {
        "run_name": run_name,
        "config": {
            "captions_path": str(captions_path),
            "fusion_type": fusion_type,
            "modality": modality,
            "alignment_loss_weight": alignment_loss_weight,
            "alignment_temperature": alignment_temperature,
            "noise_std": noise_std,
            "frame_dropout": frame_dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "max_samples": max_samples,
            "dataset_size": len(dataset),
        },
        "epochs": [],
        "checkpoints": [],
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            frames = batch["frames"].to(device)
            audio = batch["audio"].to(device)
            captions = batch["captions"]
            frames, audio = apply_input_ablations(
                frames,
                audio,
                noise_std=noise_std,
                frame_dropout=frame_dropout,
            )

            out = model(
                frames,
                audio,
                captions=captions,
                modality=modality,
                return_embeddings=alignment_loss_weight > 0,
            )
            caption_loss = out["loss"]
            loss = caption_loss
            alignment_loss = None
            if alignment_loss_weight > 0:
                alignment_loss = contrastive_alignment_loss(
                    out["vision_emb"],
                    out["audio_emb"],
                    temperature=alignment_temperature,
                )
                loss = loss + alignment_loss_weight * alignment_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")
        checkpoint_path = output_dir / f"{run_name}_epoch{epoch+1}.pt"
        torch.save(model.fusion.state_dict(), checkpoint_path)
        run_metrics["epochs"].append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "checkpoint": str(checkpoint_path),
        })
        run_metrics["checkpoints"].append(str(checkpoint_path))
        with open(results_dir / f"{run_name}_train.json", "w") as f:
            json.dump(run_metrics, f, indent=2)

    print(f"Saved checkpoints to {output_dir}")
    print(f"Saved run metrics to {results_dir / f'{run_name}_train.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=Path, default=CAPTIONS_FILE)
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--fusion-type", choices=["gated", "additive"], default="gated")
    parser.add_argument("--modality", choices=["audio_visual", "vision_only", "audio_only"], default="audio_visual")
    parser.add_argument("--alignment-loss-weight", type=float, default=0.0)
    parser.add_argument("--alignment-temperature", type=float, default=0.07)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--frame-dropout", type=float, default=0.0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--skip-dataset-validation", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    train(
        captions_path=args.captions,
        output_dir=args.output,
        results_dir=args.results_dir,
        fusion_type=args.fusion_type,
        modality=args.modality,
        alignment_loss_weight=args.alignment_loss_weight,
        alignment_temperature=args.alignment_temperature,
        noise_std=args.noise_std,
        frame_dropout=args.frame_dropout,
        run_name=args.run_name,
        validate_dataset=not args.skip_dataset_validation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
    )
