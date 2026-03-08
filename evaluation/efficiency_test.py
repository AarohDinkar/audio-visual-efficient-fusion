"""
Efficiency test: measure inference latency and GPU memory.

Simulates edge device by optionally quantizing the model.
"""

import sys
import time
from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def measure_latency(model, frames: torch.Tensor, audio: torch.Tensor, num_runs: int = 10) -> float:
    """Average inference latency in seconds."""
    model.eval()
    device = next(model.parameters()).device
    frames = frames.to(device)
    audio = audio.to(device)

    # Warmup
    with torch.no_grad():
        _ = model(frames, audio, captions=None, max_new_tokens=32)

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(frames, audio, captions=None, max_new_tokens=32)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def measure_memory(model, frames: torch.Tensor, audio: torch.Tensor) -> float:
    """Peak GPU memory in MB."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    device = next(model.parameters()).device
    with torch.no_grad():
        _ = model(frames.to(device), audio.to(device), captions=None, max_new_tokens=32)
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def run_efficiency_test(
    checkpoint_path: Optional[Path] = None,
    quantize: bool = False,
    batch_size: int = 1,
) -> dict:
    """
    Run efficiency benchmarks.

    Returns:
        {"latency_ms": float, "memory_mb": float, "quantized": bool}
    """
    from models.multimodal_model import AudioVisualCaptioner

    # Dummy inputs
    frames = torch.rand(batch_size, 8, 3, 224, 224)
    audio = torch.rand(batch_size, 16000 * 10)  # 10 sec at 16kHz

    model = AudioVisualCaptioner(fusion_type="gated")
    if checkpoint_path and checkpoint_path.exists():
        model.fusion.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    if quantize and torch.cuda.is_available():
        try:
            # Dynamic quantization for inference (CPU-friendly)
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except Exception as e:
            print(f"Quantization skipped: {e}")

    latency = measure_latency(model, frames, audio)
    memory = measure_memory(model, frames, audio)

    return {
        "latency_ms": latency * 1000,
        "memory_mb": memory,
        "quantized": quantize,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()
    results = run_efficiency_test(args.checkpoint, quantize=args.quantize)
    print("Efficiency Results:")
    print(f"  Latency: {results['latency_ms']:.2f} ms")
    print(f"  GPU Memory: {results['memory_mb']:.2f} MB")
    print(f"  Quantized: {results['quantized']}")
