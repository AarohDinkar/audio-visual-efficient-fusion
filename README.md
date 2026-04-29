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

## Quick Start (OS-Agnostic)

### Step 1: Clone & Setup Environment
```bash
git clone <repo>
cd audio-visual-efficient-fusion

# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Note on GPU Support:**
- **Automatic Detection:** The project automatically detects and uses CUDA (NVIDIA GPU) if available
- Uses `float16` precision on GPU for faster training, `float32` on CPU
- For faster llama-cpp inference on Mac, enable Metal: `CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python`

### Step 3: Download Dataset (Kaggle - requires API key)
```bash
pip install kagglehub
# Set up ~/.kaggle/kaggle.json with your Kaggle API key
python scripts/download_dataset.py
```

Or **create synthetic data (no download):**
```bash
python scripts/create_sample_data.py -n 100
```

### Step 4: Preprocess Videos
```bash
python scripts/preprocess_video.py --limit 100
```

Validates dataset:
```bash
python scripts/validate_dataset.py
```

### Step 5: Train Fusion Model
```bash
# Gated fusion (recommended)
python training/train_fusion.py --epochs 3 --batch-size 4 --max-samples 100

# Additive fusion (baseline)
python training/train_fusion.py --fusion-type additive --epochs 3
```

**GPU/CPU:** Training automatically uses available GPU; falls back to CPU if not available.

### Step 6: Evaluate & Demo
```bash
# Evaluate on 100 samples
python evaluation/evaluate.py --limit 100 --checkpoint checkpoints/fusion_gated_epoch1.pt

# Interactive demo (no checkpoint needed, uses llama-cpp backend)
python demo/demo.py path/to/video.mp4

# Demo with trained checkpoint
python demo/demo.py path/to/video.mp4 --checkpoint checkpoints/fusion_gated_epoch3.pt --max-tokens 128
```

### Step 7: Full Research Experiment Suite
```bash
# Dry-run (no execution)
python experiments/run_research_suite.py --stage local

# Execute full suite (may take hours on CPU)
python experiments/run_research_suite.py --stage local --execute
```

## Troubleshooting

### No GPU Detected
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- If False, install CUDA Toolkit 11.8+ and PyTorch with CUDA support:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

### Dataset Download Issues
- Kaggle API key not found: Ensure `~/.kaggle/kaggle.json` exists with valid credentials
- Rate limiting: Wait a few hours or use synthetic data: `python scripts/create_sample_data.py -n 100`

### Training Memory Issues (CPU)
- Reduce batch size: `--batch-size 2`
- Reduce samples: `--max-samples 50`
- On GPU with limited VRAM: Try same adjustments

## Research-Paper Experiment Workflow (Optional)

For the full ablation study:

```bash
# Validate dataset before running
python scripts/validate_dataset.py

# Repair captions if needed
python scripts/repair_captions.py --annotations data/train_val_videodatainfo.json

# Dry-run experiment matrix
python experiments/run_research_suite.py --stage local

# Execute full suite (very slow on CPU, recommended on GPU)
HF_HUB_DISABLE_SYMLINKS_WARNING=1 python experiments/run_research_suite.py --stage local --execute
```

The suite covers:
- BLIP vision-only baseline
- Audio-only, video-only modality ablations
- Gated vs additive fusion
- Contrastive alignment loss
- Robustness: noisy audio, frame dropout
- Recall@K retrieval benchmarks
- Latency & memory profiling (GPU vs CPU quantized)

Results written to `results/*.json`. Paper scaffold in `reports/research_paper.md`.

## Configuration

Edit `config.py` to change:
- `LLM_BACKEND`: `"llama_cpp"` (demo) or `"transformers"` (training)
- `LLM_MODEL`: TinyLlama (default, no login) or Gemma (gated)
- `LLAMA_CPP_REPO_ID` / `LLAMA_CPP_FILENAME`: GGUF model for inference

## Installation & System Requirements

### Prerequisites
- **Python 3.9+** (3.10+ recommended)
- **Virtual environment** (venv, conda, or poetry)

### GPU Support (Recommended)
This project **automatically detects and uses GPU if available**:
- **NVIDIA CUDA:** Automatically detected and used for training/inference
  - Requires CUDA Toolkit 11.8+ and cuDNN installed
  - Training time: ~1 hour per epoch (vs 50+ minutes on CPU)
- **Mac Metal:** Enable for faster llama-cpp inference:
  ```bash
  CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
  ```
- **CPU Fallback:** All operations work on CPU (slower but works everywhere)

### Install Dependencies
```bash
pip install -r requirements.txt
```

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
