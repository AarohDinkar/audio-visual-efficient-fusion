"""
Run the research experiment matrix.

Default behavior is --dry-run so the full paper matrix can be inspected before
launching expensive training/evaluation jobs.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def build_matrix(stage: str, samples: int, epochs: int) -> list[dict]:
    base = {"epochs": epochs, "max_samples": samples, "batch_size": 4}
    return [
        {"name": f"{stage}_gated_av", "fusion_type": "gated", "modality": "audio_visual", **base},
        {"name": f"{stage}_additive_av", "fusion_type": "additive", "modality": "audio_visual", **base},
        {"name": f"{stage}_gated_vision_only", "fusion_type": "gated", "modality": "vision_only", **base},
        {"name": f"{stage}_gated_audio_only", "fusion_type": "gated", "modality": "audio_only", **base},
        {
            "name": f"{stage}_gated_align",
            "fusion_type": "gated",
            "modality": "audio_visual",
            "alignment_loss_weight": 0.1,
            **base,
        },
        {
            "name": f"{stage}_gated_noisy_audio",
            "fusion_type": "gated",
            "modality": "audio_visual",
            "noise_std": 0.05,
            **base,
        },
        {
            "name": f"{stage}_gated_frame_dropout",
            "fusion_type": "gated",
            "modality": "audio_visual",
            "frame_dropout": 0.25,
            **base,
        },
    ]


def train_command(config: dict) -> list[str]:
    cmd = [
        sys.executable,
        "training/train_fusion.py",
        "--fusion-type", config["fusion_type"],
        "--modality", config["modality"],
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--max-samples", str(config["max_samples"]),
        "--run-name", config["name"],
        "--results-dir", str(RESULTS_DIR),
    ]
    if config.get("alignment_loss_weight", 0.0):
        cmd += ["--alignment-loss-weight", str(config["alignment_loss_weight"])]
    if config.get("noise_std", 0.0):
        cmd += ["--noise-std", str(config["noise_std"])]
    if config.get("frame_dropout", 0.0):
        cmd += ["--frame-dropout", str(config["frame_dropout"])]
    return cmd


def eval_command(config: dict, checkpoint: Path) -> list[str]:
    return [
        sys.executable,
        "evaluation/evaluate.py",
        "--checkpoint", str(checkpoint),
        "--fusion-type", config["fusion_type"],
        "--modality", config["modality"],
        "--limit", str(config["max_samples"]),
        "--run-name", f"{config['name']}_eval",
        "--results-dir", str(RESULTS_DIR),
    ]


def run_command(cmd: list[str], dry_run: bool):
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["local", "expanded"], default="local")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true", help="Actually run commands; overrides --dry-run.")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    samples = args.samples if args.samples is not None else (100 if args.stage == "local" else 1000)
    epochs = args.epochs if args.epochs is not None else (1 if args.stage == "local" else 3)
    dry_run = args.dry_run and not args.execute
    matrix = build_matrix(args.stage, samples=samples, epochs=epochs)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": args.stage,
        "samples": samples,
        "epochs": epochs,
        "dry_run": dry_run,
        "matrix": matrix,
    }
    manifest_path = RESULTS_DIR / f"{args.stage}_research_suite_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")

    run_command([sys.executable, "scripts/validate_dataset.py"], dry_run=dry_run)
    for config in matrix:
        run_command(train_command(config), dry_run=dry_run)
        checkpoint = PROJECT_ROOT / "checkpoints" / f"{config['name']}_epoch{config['epochs']}.pt"
        if not args.skip_eval:
            run_command(eval_command(config, checkpoint), dry_run=dry_run)

    run_command([
        sys.executable,
        "evaluation/retrieval.py",
        "--limit", str(samples),
        "--run-name", f"{args.stage}_retrieval",
        "--results-dir", str(RESULTS_DIR),
    ], dry_run=dry_run)

    best_checkpoint = PROJECT_ROOT / "checkpoints" / f"{args.stage}_gated_av_epoch{epochs}.pt"
    run_command([
        sys.executable,
        "evaluation/efficiency_test.py",
        "--checkpoint", str(best_checkpoint),
        "--batch-size", "1",
        "--run-name", f"{args.stage}_efficiency_unquantized",
        "--results-dir", str(RESULTS_DIR),
    ], dry_run=dry_run)
    run_command([
        sys.executable,
        "evaluation/efficiency_test.py",
        "--checkpoint", str(best_checkpoint),
        "--batch-size", "1",
        "--quantize",
        "--run-name", f"{args.stage}_efficiency_quantized",
        "--results-dir", str(RESULTS_DIR),
    ], dry_run=dry_run)


if __name__ == "__main__":
    main()
