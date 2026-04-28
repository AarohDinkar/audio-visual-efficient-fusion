# Multi-Modal Integration for Audio-Visual Language Understanding in Physical AI

## Abstract

This study evaluates whether a low-parameter local language model equipped with lightweight audio-visual fusion can produce richer scene descriptions than a vision-only baseline while preserving inference efficiency for physical-AI settings.

## Method

The system uses CLIP ViT-B/32 for visual frame embeddings, Whisper tiny for audio embeddings, and TinyLlama as the local language model. Audio and visual features are combined with either additive fusion or gated fusion. The gated layer avoids heavy cross-attention by projecting both modalities to a shared space and learning a sigmoid gate over the two signals.

## Experiment Matrix

| Experiment | Fusion | Modality | Alignment Loss | Robustness Setting |
| --- | --- | --- | --- | --- |
| Vision-only baseline | BLIP | vision | none | clean |
| Audio-only | gated | audio | none | clean |
| Video-only | gated | vision | none | clean |
| Audio+video additive | additive | audio+video | none | clean |
| Audio+video gated | gated | audio+video | none | clean |
| Audio+video gated + alignment | gated | audio+video | contrastive | clean |
| Noisy audio | gated | audio+video | none | Gaussian audio noise |
| Frame dropout | gated | audio+video | none | random frame masking |
| Quantized inference | gated | audio+video | best checkpoint | dynamic int8 |

## Metrics

| Metric | Purpose |
| --- | --- |
| BLEU | Caption n-gram precision with smoothing |
| ROUGE-L | Caption sequence overlap |
| Recall@1/5/10 | Video-caption retrieval relevance |
| Latency ms/clip | Real-time suitability |
| Peak memory MB | Device feasibility |
| Trainable parameters | Efficiency of adaptation |

## Results

Populate these tables from `results/*.json` after running:

```powershell
.\.venv\Scripts\python.exe experiments\run_research_suite.py --stage local --execute
.\.venv\Scripts\python.exe experiments\run_research_suite.py --stage expanded --samples 1000 --epochs 3 --execute
```

### Captioning

| Run | BLEU | ROUGE-L | Notes |
| --- | ---: | ---: | --- |
| BLIP vision-only | TBD | TBD |  |
| Gated audio+video | TBD | TBD |  |
| Additive audio+video | TBD | TBD |  |
| Alignment ablation | TBD | TBD |  |
| Audio-only | TBD | TBD |  |
| Video-only | TBD | TBD |  |

### Retrieval

| Run | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| CLIP video-caption retrieval | TBD | TBD | TBD |

### Efficiency

| Run | Latency ms | Memory MB | Trainable Params | Quantized |
| --- | ---: | ---: | ---: | --- |
| Gated audio+video | TBD | TBD | TBD | no |
| Gated audio+video quantized | TBD | TBD | TBD | yes |

## Qualitative Examples

Copy representative examples from the `qualitative` entries in evaluation result JSON files. Prioritize examples where audio changes the caption meaning, such as visible motion plus sound event.

## Limitations

- MSR-VTT captions may not always explicitly describe audio, so AudioSet or synthetic audio-visual captions should be used for a stronger audio-specific claim.
- Whisper is speech-oriented and may underrepresent non-speech physical-AI sounds; HuBERT or an audio event classifier can be evaluated as an extension.
- Local subset results are for debugging and should not be overclaimed until the expanded run completes.

## Reproducibility

1. Validate or repair captions:

```powershell
.\.venv\Scripts\python.exe scripts\repair_captions.py --annotations data\train_val_videodatainfo.json
.\.venv\Scripts\python.exe scripts\validate_dataset.py
```

2. Dry-run the experiment matrix:

```powershell
.\.venv\Scripts\python.exe experiments\run_research_suite.py --stage local
```

3. Execute local and expanded experiments:

```powershell
.\.venv\Scripts\python.exe experiments\run_research_suite.py --stage local --execute
.\.venv\Scripts\python.exe experiments\run_research_suite.py --stage expanded --samples 1000 --epochs 3 --execute
```
