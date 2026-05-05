# Lightweight Audio-Visual Fusion for Physical AI

A **research demo project** demonstrating that combining **audio + vision** produces better scene understanding than **vision-only models**, while keeping the architecture lightweight and efficient for physical AI systems (e.g., robots).

## Project Overview

The primary research path trains lightweight audio-visual fusion layers for retrieval/alignment, without requiring an LLM. The LLM captioner remains available as an optional language-rendering demo, but the core benchmark is whether fused audio+video embeddings retrieve the correct captions better than single-modality baselines.

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
audio-visual-efficient-fusion/
├── config.py                            # Configuration
├── requirements.txt                     # Dependencies
├── models/
│   ├── fusion.py                        # EfficientFusionLayer, AdditiveFusionLayer
│   ├── retrieval_fusion.py              # Lightweight fusion for retrieval (NEW)
│   ├── vision_encoder.py                # CLIP encoder
│   ├── audio_encoder.py                 # Whisper/HuBERT encoder
│   ├── multimodal_model.py              # Full pipeline (transformers) [optional LLM]
│   └── multimodal_model_llamacpp.py     # Llama-cpp inference
├── scripts/
│   ├── download_dataset.py              # MSR-VTT from Kaggle
│   ├── preprocess_video.py              # Extract frames + audio
│   ├── precompute_retrieval_features.py # Cache CLIP/Whisper/text embeddings (NEW)
│   ├── create_sample_data.py            # Synthetic data for testing
│   ├── validate_dataset.py              # Check captions & data integrity
│   └── repair_captions.py               # Fix placeholder captions
├── training/
│   ├── train_retrieval_fusion.py        # Train fusion head for retrieval (NEW)
│   └── train_fusion.py                  # Train LLM captioning [optional]
├── evaluation/
│   ├── evaluate_retrieval_fusion.py     # Eval Recall@K for retrieval (NEW)
│   ├── retrieval.py                     # Legacy retrieval with encoders
│   ├── evaluate.py                      # LLM captioning eval [optional]
│   ├── metrics.py                       # BLEU, ROUGE-L, Recall@K
│   └── efficiency_test.py               # Latency & memory profiling
├── demo/
│   └── demo.py                          # Interactive LLM demo [optional]
├── docs/
│   └── non_llm_retrieval_workflow.md    # Primary workflow (NEW)
├── experiments/
│   └── run_research_suite.py            # Full ablation study
├── reports/
│   └── research_paper.md                # Results & analysis
├── data/
│   ├── videos/                          # Raw video files (after download)
│   ├── audio/                           # Extracted audio features
│   ├── captions.json                    # Processed captions
│   └── processed/                       # Precomputed embeddings cache
├── checkpoints/                         # Model weights (gated, additive, retrieval)
└── results/                             # Training metrics & evaluation results
```

## Quick Start: Retrieval Workflow (Recommended)

This is the primary path: train a lightweight fusion layer for **audio-visual retrieval** using contrastive loss.

### Prerequisites
```bash
git clone <repo>
cd audio-visual-efficient-fusion

# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 1: Download & Preprocess Videos
```bash
# Download MSR-VTT (requires Kaggle API key)
python scripts/download_dataset.py

# Or create synthetic data (no download)
python scripts/create_sample_data.py -n 100

# Extract frames & audio
python scripts/preprocess_video.py --limit 100

# Validate
python scripts/validate_dataset.py
```

### Step 2: Precompute Embeddings (CPU-Friendly Cache)
```bash
# Cache CLIP vision, Whisper audio, and CLIP text embeddings
python scripts/precompute_retrieval_features.py --limit 50 \
  --one-caption-per-video \
  --output data/processed/retrieval_features_unique50.pt
```

This caches embeddings, so training only needs to learn the fusion head (fast!).

### Step 3: Train Fusion Head
```bash
# Gated fusion (recommended)
python training/train_retrieval_fusion.py \
  --cache data/processed/retrieval_features_unique50.pt \
  --fusion-type gated \
  --epochs 20 \
  --batch-size 16 \
  --run-name gated_retrieval_50

# Additive fusion (baseline)
python training/train_retrieval_fusion.py \
  --cache data/processed/retrieval_features_unique50.pt \
  --fusion-type additive \
  --epochs 20 \
  --run-name additive_retrieval_50
```

### Step 4: Evaluate
```bash
# Evaluate audio+video vs vision-only vs audio-only
python evaluation/evaluate_retrieval_fusion.py \
  --cache data/processed/retrieval_features_unique50.pt \
  --checkpoint checkpoints/gated_retrieval_50.pt \
  --fusion-type gated \
  --modality audio_visual \
  --run-name gated_retrieval_50_av_eval
```

**Metrics:** Recall@1, Recall@5, Recall@10 (how often correct caption is in top-K retrieved)

**Example Results (50 unique videos, 20 epochs):**
```json
{
  "gated_retrieval_50": {
    "audio_visual": { "R@1": 0.86, "R@5": 0.96, "R@10": 1.0 },
    "vision_only": { "R@1": 0.80, "R@5": 0.94, "R@10": 1.0 },
    "audio_only": { "R@1": 0.12, "R@5": 0.36, "R@10": 0.68 }
  }
}
```

---

## Optional: LLM Captioning (Legacy Path)

The LLM approach (TinyLlama captioning) is available but **not recommended** for small datasets (underfits). Use only if:
- You have 1000+ unique videos with varied captions
- You can GPU-train for 10+ epochs
- You want natural language generation, not retrieval

```bash
### Step 5: Train LLM (Optional, Large Data Only)
python training/train_fusion.py --epochs 3 --batch-size 4 --max-samples 1000

### Step 6: Evaluate LLM (Optional)
python evaluation/evaluate.py --limit 100 --checkpoint checkpoints/fusion_gated_epoch3.pt

### Step 7: Demo (Optional)
python demo/demo.py path/to/video.mp4 --checkpoint checkpoints/fusion_gated_epoch3.pt
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

### Training Memory Issues (CPU/GPU)
- Reduce batch size: `--batch-size 2`
- Reduce samples: `--max-samples 50`
- For retrieval: embeddings are cached, so memory is minimal
- For LLM: requires more VRAM; use GPU or reduce to CPU-friendly sizes

## Retrieval vs Captioning: Which Path?

| Aspect | Retrieval (Recommended) | LLM Captioning |
|---|---|---|
| **Task** | Match video to caption | Generate caption text |
| **Metric** | Recall@K | BLEU, ROUGE-L |
| **Training Speed** | 5-30 min/epoch (CPU) | 1h+ per epoch (GPU needed) |
| **Data Needed** | 50-500 unique videos | 1000+ diverse captions |
| **When to Use** | Measure audio-vision fusion | Qualitative demos, language generation |
| **Output** | Ranking of captions | Natural language description |

**Recommendation:** Start with **retrieval**. It's CPU-friendly (embeddings cached) and gives clear metrics.

## Research-Paper Experiment Workflow

For full ablation study comparing architectures and modalities:

```bash
# Validate dataset
python scripts/validate_dataset.py

# Precompute for retrieval (primary path)
python scripts/precompute_retrieval_features.py --limit 100 --one-caption-per-video

# Train and evaluate both fusion types
python training/train_retrieval_fusion.py --cache data/processed/retrieval_features_100.pt --fusion-type gated --epochs 20
python training/train_retrieval_fusion.py --cache data/processed/retrieval_features_100.pt --fusion-type additive --epochs 20
python evaluation/evaluate_retrieval_fusion.py --cache data/processed/retrieval_features_100.pt --checkpoint checkpoints/gated_*.pt --modality audio_visual
```

Full suite with LLM ablations:

```bash
# Optional: run legacy LLM experiment suite (slow, GPU recommended)
python experiments/run_research_suite.py --stage local --execute
```

Results saved to `results/*.json`. Analysis scaffold in `reports/research_paper.md`.

## Configuration

Edit `config.py` to customize:
- `LLM_BACKEND`: `"llama_cpp"` (inference) or `"transformers"` (training)
- `LLM_MODEL`: Model name for captioning (TinyLlama, Gemma)
- CLIP, Whisper model versions

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
