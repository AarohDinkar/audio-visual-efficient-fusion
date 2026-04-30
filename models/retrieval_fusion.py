"""
Non-LLM audio-visual fusion model for retrieval/alignment experiments.

The model maps frozen audio/video embeddings into the same space as frozen text
caption embeddings. This makes the core research claim measurable without
depending on free-form LLM generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import AdditiveFusionLayer, EfficientFusionLayer


class AudioVisualRetrievalFusion(nn.Module):
    """Trainable fusion head for video/audio-to-caption retrieval."""

    def __init__(
        self,
        vision_dim: int = 512,
        audio_dim: int = 512,
        text_dim: int = 512,
        hidden_dim: int = 512,
        fusion_type: str = "gated",
    ):
        super().__init__()
        if fusion_type == "gated":
            self.fusion = EfficientFusionLayer(
                vision_dim=vision_dim,
                audio_dim=audio_dim,
                output_dim=text_dim,
                hidden_dim=hidden_dim,
            )
        elif fusion_type == "additive":
            self.fusion = AdditiveFusionLayer(
                vision_dim=vision_dim,
                audio_dim=audio_dim,
                output_dim=text_dim,
            )
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")
        self.fusion_type = fusion_type

    def forward(
        self,
        vision_embedding: torch.Tensor,
        audio_embedding: torch.Tensor,
        modality: str = "audio_visual",
    ) -> torch.Tensor:
        if modality == "vision_only":
            audio_embedding = torch.zeros_like(audio_embedding)
        elif modality == "audio_only":
            vision_embedding = torch.zeros_like(vision_embedding)
        elif modality != "audio_visual":
            raise ValueError(f"Unsupported modality: {modality}")
        fused = self.fusion(vision_embedding, audio_embedding)
        return F.normalize(fused.float(), dim=-1)


def contrastive_retrieval_loss(
    fused_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE loss between fused AV embeddings and captions."""
    fused = F.normalize(fused_embedding.float(), dim=-1)
    text = F.normalize(text_embedding.float(), dim=-1)
    logits = fused @ text.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
