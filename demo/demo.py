"""
Demo script: Generate captions from a video file.

Compares Vision-Only (BLIP) vs Audio-Visual (our model) outputs.
"""

import sys
from pathlib import Path
from typing import Optional, Union

import torch

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    NUM_FRAMES,
    FRAME_SIZE,
    AUDIO_SAMPLE_RATE,
    AUDIO_MAX_DURATION,
    LLM_BACKEND,
    LLAMA_CPP_REPO_ID,
    LLAMA_CPP_FILENAME,
)
from scripts.preprocess_video import extract_frames, extract_audio


def load_blip():
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


def load_multimodal(checkpoint_path: Optional[Path] = None):
    """Load multimodal model - uses llama-cpp when LLM_BACKEND='llama_cpp'."""
    if LLM_BACKEND == "llama_cpp":
        from models.multimodal_model_llamacpp import AudioVisualCaptionerLlamaCpp
        return AudioVisualCaptionerLlamaCpp(
            llama_repo_id=LLAMA_CPP_REPO_ID,
            llama_filename=LLAMA_CPP_FILENAME,
        )
    else:
        from models.multimodal_model import AudioVisualCaptioner
        model = AudioVisualCaptioner(fusion_type="gated")
        if checkpoint_path and checkpoint_path.exists():
            model.fusion.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        return model


def caption_vision_only(frames: torch.Tensor, processor, model, device) -> str:
    """BLIP caption from middle frame."""
    mid = frames.shape[0] // 2
    frame = frames[mid]
    img = (frame.permute(1, 2, 0).numpy() * 255).astype("uint8")
    from PIL import Image
    img = Image.fromarray(img)
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)
    return processor.decode(out[0], skip_special_tokens=True)


def caption_multimodal(frames: torch.Tensor, audio: torch.Tensor, model, device, max_tokens: int = 150) -> str:
    """Audio-visual caption."""
    frames = frames.unsqueeze(0).to(device)
    audio = audio.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(frames, audio, captions=None, max_new_tokens=max_tokens)
    # Llama-cpp returns "generated_text", transformers returns "generated_ids"
    if "generated_text" in out:
        text = out["generated_text"]
        return text if isinstance(text, str) else text[0]
    ids = out.get("generated_ids", out.get("full_ids", out["generated_ids"]))[0]
    return model.tokenizer.decode(ids, skip_special_tokens=True)


def run_demo(
    video_path: Union[str, Path],
    checkpoint_path: Optional[Path] = None,
    max_tokens: int = 150,
) -> None:
    """
    Full demo pipeline: extract -> encode -> fuse -> generate.

    Prints:
        Vision Only Caption: "..."
        Multimodal Caption: "..."
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Extract frames and audio
    print("Extracting frames and audio...")
    frames = extract_frames(video_path, num_frames=NUM_FRAMES, size=FRAME_SIZE)
    audio = extract_audio(video_path, sample_rate=AUDIO_SAMPLE_RATE, max_duration=AUDIO_MAX_DURATION)
    if frames is None:
        print("Could not extract frames from video.")
        return

    # 2. Vision-only (BLIP)
    print("Loading BLIP (vision-only)...")
    blip_processor, blip_model = load_blip()
    blip_model = blip_model.to(device)
    vision_caption = caption_vision_only(frames, blip_processor, blip_model, device)

    # 3. Multimodal (our model)
    print("Loading Audio-Visual model...")
    mm_model = load_multimodal(checkpoint_path)
    if hasattr(mm_model, "to") and hasattr(mm_model, "parameters"):
        mm_model = mm_model.to(device)
    multimodal_caption = caption_multimodal(frames, audio, mm_model, device, max_tokens=max_tokens)

    # 4. Output
    print("\n" + "=" * 50)
    print("Vision Only Caption")
    print("-" * 50)
    print(vision_caption)
    print("\nMultimodal Caption")
    print("-" * 50)
    print(multimodal_caption)
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audio-Visual Captioning Demo")
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Fusion checkpoint (optional)")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max tokens for caption (default: 150)")
    args = parser.parse_args()
    run_demo(args.video, args.checkpoint, max_tokens=args.max_tokens)
