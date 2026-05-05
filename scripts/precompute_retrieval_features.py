"""
Precompute frozen audio, video, and text embeddings for non-LLM retrieval.

This is intentionally separated from training so CPU machines only pay the
encoder cost once. The resulting .pt cache is consumed by
training/train_retrieval_fusion.py and evaluation/evaluate_retrieval_fusion.py.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CAPTIONS_FILE, VISION_ENCODER
from models.audio_encoder import WhisperAudioEncoder
from models.vision_encoder import CLIPVisionEncoder
from scripts.validate_dataset import validate_captions


AUDIO_KEYWORDS = {
    "audio",
    "sound",
    "sounds",
    "music",
    "song",
    "sing",
    "singing",
    "sings",
    "talk",
    "talking",
    "speaking",
    "speech",
    "voice",
    "voices",
    "laugh",
    "laughing",
    "clap",
    "clapping",
    "cheer",
    "cheering",
    "shout",
    "shouting",
    "cry",
    "crying",
    "bark",
    "barking",
    "noise",
    "guitar",
    "piano",
    "drum",
    "drums",
}


def _extract_tensor(output, field: str) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, field):
        return getattr(output, field)
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    raise TypeError(f"Unsupported output type: {type(output)}")


def load_samples(
    captions_path: Path,
    limit: Optional[int],
    one_caption_per_video: bool = False,
    audio_keywords_only: bool = False,
) -> list[dict]:
    with open(captions_path, encoding="utf-8") as f:
        samples = json.load(f)
    if audio_keywords_only:
        samples = [
            sample for sample in samples
            if any(keyword in sample["caption"].lower() for keyword in AUDIO_KEYWORDS)
        ]
    if one_caption_per_video:
        seen = set()
        unique_samples = []
        for sample in samples:
            video_id = sample["video_id"]
            if video_id in seen:
                continue
            seen.add(video_id)
            unique_samples.append(sample)
        samples = unique_samples
    return samples[:limit] if limit else samples


def precompute_features(
    captions_path: Path = CAPTIONS_FILE,
    output: Path = Path("data/processed/retrieval_features.pt"),
    limit: Optional[int] = None,
    one_caption_per_video: bool = False,
    audio_keywords_only: bool = False,
    device: Optional[str] = None,
    validate_dataset: bool = True,
) -> dict:
    if validate_dataset:
        validate_captions(captions_path)
    samples = load_samples(
        captions_path,
        limit,
        one_caption_per_video=one_caption_per_video,
        audio_keywords_only=audio_keywords_only,
    )
    if not samples:
        raise ValueError("No samples loaded.")

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vision_encoder = CLIPVisionEncoder(VISION_ENCODER).to(device_obj).eval()
    audio_encoder = WhisperAudioEncoder(project_to_512=True).to(device_obj).eval()
    text_model = CLIPModel.from_pretrained(VISION_ENCODER).to(device_obj).eval()
    text_processor = CLIPProcessor.from_pretrained(VISION_ENCODER)

    av_cache = {}
    records = []
    for sample in tqdm(samples, desc="Precomputing retrieval features"):
        video_id = sample["video_id"]
        if video_id not in av_cache:
            frames = torch.load(sample["frames_path"]).unsqueeze(0).to(device_obj)
            audio = torch.load(sample["audio_path"]).unsqueeze(0).to(device_obj)
            with torch.no_grad():
                vision_embedding = vision_encoder(frames).squeeze(0).cpu()
                audio_embedding = audio_encoder(audio).squeeze(0).cpu()
            av_cache[video_id] = {
                "vision_embedding": vision_embedding,
                "audio_embedding": audio_embedding,
            }

        text_inputs = text_processor(
            text=[sample["caption"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device_obj)
        with torch.no_grad():
            text_output = text_model.get_text_features(**text_inputs)
            text_embedding = _extract_tensor(text_output, "text_embeds").squeeze(0).cpu()

        record = {
            "video_id": video_id,
            "caption_id": sample.get("caption_id", video_id),
            "caption": sample["caption"],
            "vision_embedding": av_cache[video_id]["vision_embedding"],
            "audio_embedding": av_cache[video_id]["audio_embedding"],
            "text_embedding": text_embedding,
        }
        records.append(record)

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "captions_path": str(captions_path),
            "limit": limit,
            "one_caption_per_video": one_caption_per_video,
            "audio_keywords_only": audio_keywords_only,
            "num_samples": len(records),
            "num_unique_videos": len(av_cache),
            "vision_encoder": VISION_ENCODER,
            "audio_encoder": "openai/whisper-tiny",
        },
        "records": records,
    }
    torch.save(payload, output)
    print(json.dumps({"output": str(output), **payload["config"]}, indent=2))
    return payload["config"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=Path, default=CAPTIONS_FILE)
    parser.add_argument("--output", type=Path, default=Path("data/processed/retrieval_features.pt"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--one-caption-per-video", action="store_true")
    parser.add_argument("--audio-keywords-only", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-dataset-validation", action="store_true")
    args = parser.parse_args()
    precompute_features(
        captions_path=args.captions,
        output=args.output,
        limit=args.limit,
        one_caption_per_video=args.one_caption_per_video,
        audio_keywords_only=args.audio_keywords_only,
        device=args.device,
        validate_dataset=not args.skip_dataset_validation,
    )


if __name__ == "__main__":
    main()
