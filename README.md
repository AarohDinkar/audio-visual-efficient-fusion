# Lightweight Audio-Visual Fusion for Physical AI

A **research demo project** demonstrating that combining **audio + vision** produces better scene understanding than **vision-only models**, while keeping the architecture lightweight and efficient for physical AI systems (e.g., robots).

## Project Overview

This system takes a short video with audio as input and generates a natural language description using both visual and audio information. It compares against a vision-only baseline (BLIP) to show the benefit of multimodal fusion.

**Example:**
- **Input:** Video of dog running and barking
- **Vision-only (BLIP):** "A dog running in a grassy field."
- **Audio + Vision:** "A dog running in a grassy field while barking."

## Architecture

```
VIDEO INPUT                    AUDIO INPUT
     │                              │
     ▼                              ▼
extract frames              extract waveform
     │                              │
     ▼                              ▼
CLIP ViT-B/32               Whisper tiny
(Vision Encoder)            (Audio Encoder)
     │                              │
     ▼                              ▼
VISION EMBEDDING (512D)     AUDIO EMBEDDING (512D)
     │                              │
     └──────────┬───────────────────┘
                ▼
        Efficient Fusion Layer
        (gated: fusion = sigmoid(W1*v + W2*a)
                output = fusion*v + (1-fusion)*a)
                ▼
        Project to LLM dim
                ▼
        TinyLlama / Gemma (prefix or text)
                ▼
        Generated Caption
```

### Design Constraints
- **Lightweight fusion** – no cross-attention layers
- **Gated fusion formula:** `fusion = sigmoid(W1*vision + W2*audio)`, `output = fusion * vision + (1 - fusion) * audio`
- **Optimized for real-time inference** on edge devices

## Tech Stack

| Component | Model |
|-----------|-------|
| Vision Encoder | OpenAI CLIP ViT-B/32 |
| Audio Encoder | Whisper tiny (or HuBERT base) |
| Language Model | TinyLlama (training + llama-cpp inference) |
| Baseline | BLIP image captioning |

**Note:** Dataset (videos, frames, audio) and model checkpoints are **not** included in the repo. Download separately—see below.

## Project Structure

```
audio_visual_llm/
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
├── models/
│   ├── fusion.py             # EfficientFusionLayer, AdditiveFusionLayer
│   ├── vision_encoder.py     # CLIP encoder
│   ├── audio_encoder.py     # Whisper/HuBERT encoder
│   ├── multimodal_model.py  # Full pipeline (transformers)
│   └── multimodal_model_llamacpp.py  # Llama-cpp inference
├── scripts/
│   ├── download_dataset.py   # MSR-VTT from Kaggle
│   ├── preprocess_video.py   # Extract frames + audio
│   └── create_sample_data.py # Synthetic data for testing
├── training/
│   └── train_fusion.py       # Train fusion layer
├── evaluation/
│   ├── metrics.py            # BLEU, ROUGE-L, Recall@K
│   ├── evaluate.py           # Compare BLIP vs Multimodal
│   └── efficiency_test.py   # Latency & memory
├── demo/
│   └── demo.py               # Interactive demo
├── data/                     # (empty - add videos after clone)
│   └── videos/               # Place .mp4/.avi here
└── README.md
```

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download dataset (Kaggle - requires kaggle.json)
python scripts/download_dataset.py

# 3. Preprocess
python scripts/preprocess_video.py --limit 100

# 4. Train
python training/train_fusion.py --epochs 3 --batch-size 4 --max-samples 100

# 5. Demo
python demo/demo.py path/to/video.mp4 --checkpoint checkpoints/fusion_gated_epoch3.pt
```

## Dataset: MSR-VTT

MSR-VTT contains ~10k videos with captions. **Not included in repo**—download separately.

### Download from Kaggle
```bash
pip install kagglehub
# Set up ~/.kaggle/kaggle.json with your API key
python scripts/download_dataset.py
```

### Preprocessing
```bash
# After videos are in data/videos/:
python scripts/preprocess_video.py --limit 100

# Or create synthetic data (no download):
python scripts/create_sample_data.py -n 10
```

## Training

```bash
# Train gated fusion (uses TinyLlama - no HF login)
python training/train_fusion.py --epochs 3 --batch-size 4 --max-samples 100

# Additive fusion (baseline)
python training/train_fusion.py --fusion-type additive --epochs 3
```

## Demo

```bash
# Llama-cpp backend (default) - uses TinyLlama GGUF, no checkpoint needed
python demo/demo.py path/to/video.mp4

# With transformers checkpoint (trained fusion)
python demo/demo.py path/to/video.mp4 --checkpoint checkpoints/fusion_gated_epoch3.pt

# Longer captions
python demo/demo.py path/to/video.mp4 --max-tokens 256
```

## Evaluation

```bash
python evaluation/evaluate.py --limit 50 --checkpoint checkpoints/fusion_gated_epoch3.pt
```

**Metrics:** BLEU, ROUGE-L

## Research-Paper Experiment Workflow

Before training, validate that the preprocessed dataset has real captions:

```bash
python scripts/validate_dataset.py
```

If validation reports placeholder captions such as `"A video."`, repair the
metadata from the MSR-VTT annotation JSON:

```bash
python scripts/repair_captions.py --annotations data/train_val_videodatainfo.json
```

Dry-run the full ablation matrix:

```bash
python experiments/run_research_suite.py --stage local
```

Execute the local proof run:

```bash
python experiments/run_research_suite.py --stage local --execute
```

The suite covers:
- BLIP vision-only baseline
- audio-only and video-only modality ablations
- gated vs additive audio-visual fusion
- contrastive audio-visual alignment loss
- noisy audio and frame-dropout robustness ablations
- Recall@K retrieval
- quantized and unquantized latency/memory benchmarks

Paper tables and limitations are scaffolded in `reports/research_paper.md`.
Raw metrics are written to `results/*.json`.

## Configuration

Edit `config.py` to change:
- `LLM_BACKEND`: `"llama_cpp"` (demo) or `"transformers"` (training)
- `LLM_MODEL`: TinyLlama (default, no login) or Gemma (gated)
- `LLAMA_CPP_REPO_ID` / `LLAMA_CPP_FILENAME`: GGUF model for inference

## Installation

```bash
cd audio_visual_llm
pip install -r requirements.txt
```

- **Python 3.9+** (3.10 recommended)
- **GPU** recommended for training
- **Metal** (Mac) or **CUDA** for faster llama-cpp: `CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python`

## What's Not in the Repo

The following are excluded via `.gitignore` (too large for GitHub):
- `data/videos/` – raw video files
- `data/frames/` – extracted frames
- `data/audio/` – extracted audio
- `data/captions.json` – processed captions
- `checkpoints/` – trained model weights
- `venv/` – virtual environment

Clone the repo, then run `download_dataset.py` and `preprocess_video.py` to generate data locally.

## License

Research demo – use for educational and research purposes.
