"""
Vision Encoder using OpenAI CLIP ViT-B/32.

Extracts visual features from video frames for multimodal fusion.
"""

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel


class CLIPVisionEncoder(nn.Module):
    """
    CLIP ViT-B/32 vision encoder for video frame feature extraction.

    For each frame: vision_features = CLIP(image)
    Average pool across frames for video-level representation.
    Output: 512-dimensional embedding per video.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        # get_image_features returns projected embeddings (projection_dim), not hidden_size
        self.embed_dim = self.model.config.projection_dim  # 512 for CLIP

        # Freeze CLIP - we only use it for feature extraction
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self.embed_dim

    def preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Preprocess frames for CLIP. Expects [B, T, C, H, W] or [B, C, H, W].

        Returns normalized tensor for CLIP.
        """
        if frames.dim() == 5:
            # [B, T, C, H, W] -> flatten to [B*T, C, H, W]
            b, t, c, h, w = frames.shape
            frames = frames.view(b * t, c, h, w)
        return frames

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract vision features from video frames.

        Args:
            frames: [batch, num_frames, 3, 224, 224] - preprocessed RGB frames

        Returns:
            vision_embedding: [batch, 512] - averaged frame features
        """
        batch_size = frames.shape[0]
        num_frames = frames.shape[1]

        # Reshape: [B, T, C, H, W] -> [B*T, C, H, W]
        frames_flat = frames.view(-1, *frames.shape[2:])

        # CLIP ImageNet normalization (frames assumed in [0, 1])
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=frames.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=frames.device).view(1, 3, 1, 1)
        frames_flat = (frames_flat - mean) / std

        with torch.no_grad():
            output = self.model.get_image_features(pixel_values=frames_flat)
            # Handle both tensor and output object cases
            if isinstance(output, torch.Tensor):
                vision_outputs = output
            else:
                # Extract pooled image embeddings from output object
                vision_outputs = output.pooler_output if hasattr(output, 'pooler_output') else output

        # Reshape back: [B*T, 512] -> [B, T, 512]
        vision_outputs = vision_outputs.view(batch_size, num_frames, -1)

        # Average pool across frames
        vision_embedding = vision_outputs.mean(dim=1)  # [B, 512]

        return vision_embedding

    def encode_from_pil(self, images) -> torch.Tensor:
        """
        Encode from list of PIL images (for inference without preprocessing).
        """
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features.mean(dim=0, keepdim=True)
