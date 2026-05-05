# Non-LLM Fusion Retrieval Workflow

Use this path as the main research benchmark. It avoids fragile free-form LLM
captioning and trains only a small fusion head against frozen caption
embeddings.

## Commands

```powershell
# Precompute frozen audio/video/text embeddings.
.\.venv\Scripts\python.exe scripts\precompute_retrieval_features.py --limit 100 --output data\processed\retrieval_features_100.pt

# Train gated fusion.
.\.venv\Scripts\python.exe training\train_retrieval_fusion.py --cache data\processed\retrieval_features_100.pt --fusion-type gated --epochs 20 --run-name gated_retrieval_100

# Evaluate fused retrieval.
.\.venv\Scripts\python.exe evaluation\evaluate_retrieval_fusion.py --cache data\processed\retrieval_features_100.pt --checkpoint checkpoints\gated_retrieval_100.pt --fusion-type gated --run-name gated_retrieval_100_eval

# Modality ablations using the same trained checkpoint.
.\.venv\Scripts\python.exe evaluation\evaluate_retrieval_fusion.py --cache data\processed\retrieval_features_100.pt --checkpoint checkpoints\gated_retrieval_100.pt --fusion-type gated --modality vision_only --run-name gated_retrieval_100_vision_only
.\.venv\Scripts\python.exe evaluation\evaluate_retrieval_fusion.py --cache data\processed\retrieval_features_100.pt --checkpoint checkpoints\gated_retrieval_100.pt --fusion-type gated --modality audio_only --run-name gated_retrieval_100_audio_only
```

## Reporting

Use `results/*retrieval*.json` for Recall@1, Recall@5, and Recall@10. Treat
LLM captions as qualitative examples only unless the LLM path is retrained and
validated separately.
