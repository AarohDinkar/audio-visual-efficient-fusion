"""
Evaluation script: compare BLIP (vision-only) vs Audio-Visual model.

Metrics: BLEU, ROUGE-L, Recall@K
"""

import json
import sys
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import CAPTIONS_FILE, LLM_MODEL
from evaluation.metrics import compute_caption_metrics, bleu, rouge_l


def load_blip():
    """Load BLIP image captioning model."""
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


def caption_with_blip(frames_path: str, processor, model, device) -> str:
    """Extract middle frame and caption with BLIP."""
    import torch
    frames = torch.load(frames_path)
    # Middle frame
    mid = frames.shape[0] // 2
    frame = frames[mid]  # [3, H, W]
    # BLIP expects PIL or numpy in [0, 255]
    img = (frame.permute(1, 2, 0).numpy() * 255).astype("uint8")
    from PIL import Image
    img = Image.fromarray(img)
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)
    return processor.decode(out[0], skip_special_tokens=True)


def caption_with_multimodal(frames_path: str, audio_path: str, model, device) -> str:
    """Caption with our audio-visual model."""
    frames = torch.load(frames_path).unsqueeze(0).to(device)
    audio = torch.load(audio_path).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(frames, audio, captions=None, max_new_tokens=50)
    ids = out["generated_ids"][0]
    return model.tokenizer.decode(ids, skip_special_tokens=True)


def run_evaluation(
    captions_path: Path = CAPTIONS_FILE,
    limit: Optional[int] = None,
    multimodal_checkpoint: Optional[Path] = None,
) -> dict:
    """
    Run full evaluation: BLIP vs Multimodal.

    Returns:
        {
            "blip": {"BLEU": ..., "ROUGE-L": ...},
            "multimodal": {"BLEU": ..., "ROUGE-L": ...},
        }
    """
    if not captions_path.exists():
        print(f"Captions not found: {captions_path}")
        return {}

    with open(captions_path) as f:
        samples = json.load(f)
    if limit:
        samples = samples[:limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # BLIP baseline
    print("Loading BLIP...")
    blip_processor, blip_model = load_blip()
    blip_model = blip_model.to(device)

    blip_refs, blip_hyps = [], []
    for s in tqdm(samples, desc="BLIP"):
        ref = s["caption"]
        hyp = caption_with_blip(s["frames_path"], blip_processor, blip_model, device)
        blip_refs.append(ref)
        blip_hyps.append(hyp)

    blip_metrics = compute_caption_metrics(blip_refs, blip_hyps)
    print(f"BLIP: BLEU={blip_metrics['BLEU']:.4f}, ROUGE-L={blip_metrics['ROUGE-L']:.4f}")

    # Multimodal (if checkpoint available)
    multimodal_metrics = None
    if multimodal_checkpoint and multimodal_checkpoint.exists():
        print("Loading Multimodal model...")
        from models.multimodal_model import AudioVisualCaptioner
        model = AudioVisualCaptioner(fusion_type="gated", llm_name=LLM_MODEL)
        model.fusion.load_state_dict(torch.load(multimodal_checkpoint, map_location="cpu"))
        model = model.to(device)

        mm_refs, mm_hyps = [], []
        for s in tqdm(samples, desc="Multimodal"):
            ref = s["caption"]
            hyp = caption_with_multimodal(s["frames_path"], s["audio_path"], model, device)
            mm_refs.append(ref)
            mm_hyps.append(hyp)
        multimodal_metrics = compute_caption_metrics(mm_refs, mm_hyps)
        print(f"Multimodal: BLEU={multimodal_metrics['BLEU']:.4f}, ROUGE-L={multimodal_metrics['ROUGE-L']:.4f}")
    else:
        print("No multimodal checkpoint - skipping. Train first with train_fusion.py")

    return {
        "blip": blip_metrics,
        "multimodal": multimodal_metrics,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=Path, default=CAPTIONS_FILE)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/fusion_gated_epoch3.pt"))
    args = parser.parse_args()
    run_evaluation(
        captions_path=args.captions,
        limit=args.limit,
        multimodal_checkpoint=args.checkpoint,
    )
