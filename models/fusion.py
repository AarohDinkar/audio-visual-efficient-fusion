"""
Efficient Fusion Layer for Audio-Visual Multimodal Understanding.

Implements lightweight gated fusion instead of heavy cross-attention.
Formula: fusion = sigmoid(W1*vision + W2*audio)
         output = fusion * vision + (1 - fusion) * audio
"""

import torch
import torch.nn as nn


class EfficientFusionLayer(nn.Module):
    """
    Lightweight gated fusion layer for combining vision and audio embeddings.
    
    Avoids cross-attention for efficiency - suitable for real-time inference
    on edge devices and physical AI systems.
    
    Inputs:
        vision_embedding: [batch, vision_dim] - CLIP visual features
        audio_embedding: [batch, audio_dim] - Whisper/HuBERT audio features
    
    Output:
        fused_embedding: [batch, output_dim] - Combined representation for LLM
    """

    def __init__(
        self,
        vision_dim: int = 512,
        audio_dim: int = 512,
        output_dim: int = 2560,  # Gemma-2B embedding dim
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.audio_dim = audio_dim
        self.output_dim = output_dim

        # Step 1: Linear projection for each modality to common dimension
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Step 2: Gated fusion - learnable weights for combining modalities
        # fusion_gate = sigmoid(W1*vision + W2*audio)
        self.gate_vision = nn.Linear(hidden_dim, hidden_dim)
        self.gate_audio = nn.Linear(hidden_dim, hidden_dim)

        # Step 3: Project fused representation to LLM embedding dimension
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        vision_embedding: torch.Tensor,
        audio_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse vision and audio embeddings using gated fusion.

        Args:
            vision_embedding: [batch, vision_dim] - averaged CLIP frame features
            audio_embedding: [batch, audio_dim] - mean-pooled audio features

        Returns:
            fused_embedding: [batch, output_dim] - ready for LLM prefix tokens
        """
        # Step 1: Project both modalities to common hidden dimension
        vision_hidden = self.vision_proj(vision_embedding)  # [B, hidden_dim]
        audio_hidden = self.audio_proj(audio_embedding)  # [B, hidden_dim]

        # Step 2: Gated fusion
        # fusion = sigmoid(W1*vision + W2*audio)
        # output = fusion * vision + (1 - fusion) * audio
        gate_logits = self.gate_vision(vision_hidden) + self.gate_audio(audio_hidden)
        fusion_gate = torch.sigmoid(gate_logits)

        # Weighted combination - gate controls how much of each modality to use
        fused_hidden = fusion_gate * vision_hidden + (1 - fusion_gate) * audio_hidden

        # Step 3: Project to LLM embedding dimension
        fused_embedding = self.output_proj(fused_hidden)

        return fused_embedding


class AdditiveFusionLayer(nn.Module):
    """
    Simple additive fusion baseline for comparison.
    output = W * concat([vision, audio])
    """

    def __init__(
        self,
        vision_dim: int = 512,
        audio_dim: int = 512,
        output_dim: int = 2560,
    ):
        super().__init__()
        self.fusion = nn.Linear(vision_dim + audio_dim, output_dim)

    def forward(
        self,
        vision_embedding: torch.Tensor,
        audio_embedding: torch.Tensor,
    ) -> torch.Tensor:
        concatenated = torch.cat([vision_embedding, audio_embedding], dim=-1)
        return self.fusion(concatenated)
