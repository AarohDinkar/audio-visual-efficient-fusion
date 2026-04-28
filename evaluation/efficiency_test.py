"""
Efficiency test: measure inference latency and GPU memory.

Simulates edge device by optionally quantizing the model.
"""

import sys
import time
import json
from datetime import datetime
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
    results_dir: Path = Path("results"),
    run_name: Optional[str] = None,
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
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if quantize:
        try:
            # Dynamic quantization for inference (CPU-friendly)
            model = model.to("cpu")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except Exception as e:
            print(f"Quantization skipped: {e}")

    latency = measure_latency(model, frames, audio)
    memory = measure_memory(model, frames, audio)

    results = {
        "run_name": run_name or f"efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "config": {
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "batch_size": batch_size,
        },
        "latency_ms": latency * 1000,
        "memory_mb": memory,
        "quantized": quantize,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "device": "cuda" if torch.cuda.is_available() and not quantize else "cpu",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{results['run_name']}.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    results = run_efficiency_test(
        args.checkpoint,
        quantize=args.quantize,
        batch_size=args.batch_size,
        results_dir=args.results_dir,
        run_name=args.run_name,
    )
    print("Efficiency Results:")
    print(f"  Latency: {results['latency_ms']:.2f} ms")
    print(f"  GPU Memory: {results['memory_mb']:.2f} MB")
    print(f"  Quantized: {results['quantized']}")
    print(f"  Trainable params: {results['trainable_params']}")
