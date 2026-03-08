"""
Training script for Audio-Visual Fusion model.

Trains the fusion layer + LLM projection while keeping encoders frozen.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CAPTIONS_FILE, LLM_MODEL
from models.multimodal_model import AudioVisualCaptioner


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
    fusion_type: str = "gated",
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            frames = batch["frames"].to(device)
            audio = batch["audio"].to(device)
            captions = batch["captions"]

            out = model(frames, audio, captions=captions)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")
        torch.save(model.fusion.state_dict(), output_dir / f"fusion_{fusion_type}_epoch{epoch+1}.pt")

    print(f"Saved checkpoints to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=Path, default=CAPTIONS_FILE)
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--fusion-type", choices=["gated", "additive"], default="gated")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    train(
        captions_path=args.captions,
        output_dir=args.output,
        fusion_type=args.fusion_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
    )
