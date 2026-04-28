"""
Evaluation script: compare BLIP (vision-only) vs Audio-Visual model.

Metrics: BLEU, ROUGE-L, Recall@K
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import CAPTIONS_FILE, LLM_MODEL
from evaluation.metrics import compute_caption_metrics, bleu, rouge_l
from scripts.validate_dataset import validate_captions


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


def caption_with_multimodal(frames_path: str, audio_path: str, model, device, modality: str = "audio_visual") -> str:
    """Caption with our audio-visual model."""
    frames = torch.load(frames_path).unsqueeze(0).to(device)
    audio = torch.load(audio_path).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(frames, audio, captions=None, max_new_tokens=50, modality=modality)
    ids = out["generated_ids"][0]
    return model.tokenizer.decode(ids, skip_special_tokens=True)


def run_evaluation(
    captions_path: Path = CAPTIONS_FILE,
    limit: Optional[int] = None,
    multimodal_checkpoint: Optional[Path] = None,
    fusion_type: str = "gated",
    modality: str = "audio_visual",
    results_dir: Path = Path("results"),
    run_name: Optional[str] = None,
    validate_dataset: bool = True,
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
    if validate_dataset:
        validate_captions(captions_path)

    with open(captions_path) as f:
        samples = json.load(f)
    if limit:
        samples = samples[:limit]
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_name or f"eval_{fusion_type}_{modality}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # BLIP baseline
    print("Loading BLIP...")
    blip_processor, blip_model = load_blip()
    blip_model = blip_model.to(device)

    blip_refs, blip_hyps = [], []
    qualitative = []
    for s in tqdm(samples, desc="BLIP"):
        ref = s["caption"]
        hyp = caption_with_blip(s["frames_path"], blip_processor, blip_model, device)
        blip_refs.append(ref)
        blip_hyps.append(hyp)
        if len(qualitative) < 10:
            qualitative.append({
                "video_id": s.get("video_id", ""),
                "reference": ref,
                "blip": hyp,
            })

    blip_metrics = compute_caption_metrics(blip_refs, blip_hyps)
    print(f"BLIP: BLEU={blip_metrics['BLEU']:.4f}, ROUGE-L={blip_metrics['ROUGE-L']:.4f}")

    # Multimodal (if checkpoint available)
    multimodal_metrics = None
    if multimodal_checkpoint and multimodal_checkpoint.exists():
        print("Loading Multimodal model...")
        from models.multimodal_model import AudioVisualCaptioner
        model = AudioVisualCaptioner(fusion_type=fusion_type, llm_name=LLM_MODEL)
        model.fusion.load_state_dict(torch.load(multimodal_checkpoint, map_location="cpu"))
        model = model.to(device)

        mm_refs, mm_hyps = [], []
        for s in tqdm(samples, desc="Multimodal"):
            ref = s["caption"]
            hyp = caption_with_multimodal(s["frames_path"], s["audio_path"], model, device, modality=modality)
            mm_refs.append(ref)
            mm_hyps.append(hyp)
            if len(mm_hyps) <= len(qualitative):
                qualitative[len(mm_hyps) - 1]["multimodal"] = hyp
        multimodal_metrics = compute_caption_metrics(mm_refs, mm_hyps)
        print(f"Multimodal: BLEU={multimodal_metrics['BLEU']:.4f}, ROUGE-L={multimodal_metrics['ROUGE-L']:.4f}")
    else:
        print("No multimodal checkpoint - skipping. Train first with train_fusion.py")

    result = {
        "run_name": run_name,
        "config": {
            "captions_path": str(captions_path),
            "limit": limit,
            "checkpoint": str(multimodal_checkpoint) if multimodal_checkpoint else None,
            "fusion_type": fusion_type,
            "modality": modality,
            "num_samples": len(samples),
        },
        "blip": blip_metrics,
        "multimodal": multimodal_metrics,
        "qualitative": qualitative,
    }
    with open(results_dir / f"{run_name}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved evaluation results to {results_dir / f'{run_name}.json'}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=Path, default=CAPTIONS_FILE)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/fusion_gated_epoch3.pt"))
    parser.add_argument("--fusion-type", choices=["gated", "additive"], default="gated")
    parser.add_argument("--modality", choices=["audio_visual", "vision_only", "audio_only"], default="audio_visual")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--skip-dataset-validation", action="store_true")
    args = parser.parse_args()
    run_evaluation(
        captions_path=args.captions,
        limit=args.limit,
        multimodal_checkpoint=args.checkpoint,
        fusion_type=args.fusion_type,
        modality=args.modality,
        results_dir=args.results_dir,
        run_name=args.run_name,
        validate_dataset=not args.skip_dataset_validation,
    )
