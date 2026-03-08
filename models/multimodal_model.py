"""
Full Audio-Visual Multimodal Model for video captioning.

Pipeline: Video -> [Vision Encoder, Audio Encoder] -> Fusion -> LLM -> Caption
"""

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import LLM_MODEL

from .fusion import EfficientFusionLayer, AdditiveFusionLayer
from .vision_encoder import CLIPVisionEncoder
from .audio_encoder import WhisperAudioEncoder


class AudioVisualCaptioner(nn.Module):
    """
    End-to-end audio-visual captioning model.

    Uses prefix tuning: fused embedding is prepended as soft prompts to LLM.
    """

    def __init__(
        self,
        vision_encoder: str = "openai/clip-vit-base-patch32",
        audio_encoder: str = "openai/whisper-tiny",
        llm_name: Optional[str] = None,
        fusion_type: str = "gated",  # "gated" or "additive"
        num_prefix_tokens: int = 8,
    ):
        super().__init__()
        llm_name = llm_name or LLM_MODEL
        self.vision_enc = CLIPVisionEncoder(vision_encoder)
        self.audio_enc = WhisperAudioEncoder(audio_encoder, project_to_512=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.num_prefix_tokens = num_prefix_tokens

        # LLM embedding dimension
        llm_embed_dim = self.llm.config.hidden_size
        vision_dim = self.vision_enc.output_dim  # 512
        audio_dim = self.audio_enc.output_dim   # 512

        if fusion_type == "gated":
            self.fusion = EfficientFusionLayer(
                vision_dim=vision_dim,
                audio_dim=audio_dim,
                output_dim=llm_embed_dim * num_prefix_tokens,
            )
        else:
            self.fusion = AdditiveFusionLayer(
                vision_dim=vision_dim,
                audio_dim=audio_dim,
                output_dim=llm_embed_dim * num_prefix_tokens,
            )

        self.llm_embed_dim = llm_embed_dim

    def get_device(self):
        return next(self.parameters()).device

    def forward(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
        captions: Optional[List[str]] = None,
        max_new_tokens: int = 64,
    ) -> dict:
        """
        Forward pass - generate captions from video.

        Args:
            frames: [B, T, 3, 224, 224]
            audio: [B, samples]
            captions: optional, for training (teacher forcing)
            max_new_tokens: for generation

        Returns:
            dict with "generated_ids", "loss" (if captions provided)
        """
        device = self.get_device()
        frames = frames.to(device)
        audio = audio.to(device)

        # Encode
        vision_emb = self.vision_enc(frames)
        audio_emb = self.audio_enc(audio)

        # Fuse
        fused = self.fusion(vision_emb, audio_emb)  # [B, llm_dim * num_prefix]
        B = fused.shape[0]
        prefix_emb = fused.view(B, self.num_prefix_tokens, self.llm_embed_dim)

        if captions is not None:
            # Training: teacher forcing
            inputs = self.tokenizer(
                captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)
            caption_ids = inputs["input_ids"]
            caption_emb = self.llm.get_input_embeddings()(caption_ids)
            # Prepend prefix
            full_emb = torch.cat([prefix_emb, caption_emb], dim=1)
            # Create labels: -100 for prefix, actual ids for caption
            prefix_labels = torch.full((B, self.num_prefix_tokens), -100, device=device, dtype=caption_ids.dtype)
            labels = torch.cat([prefix_labels, caption_ids], dim=1)
            outputs = self.llm(inputs_embeds=full_emb, labels=labels)
            return {"loss": outputs.loss, "logits": outputs.logits}
        else:
            # Inference: generate
            return self._generate(prefix_emb, max_new_tokens, device)

    def _generate(self, prefix_emb: torch.Tensor, max_new_tokens: int, device: torch.device):
        """Generate caption from prefix embedding."""
        B = prefix_emb.shape[0]
        # Use a short text prompt for better caption quality (e.g. "Caption: ")
        prompt = "Describe this video: "
        start_ids = self.tokenizer(
            [prompt] * B,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)["input_ids"]
        start_emb = self.llm.get_input_embeddings()(start_ids)
        full_emb = torch.cat([prefix_emb, start_emb], dim=1)
        generated = self.llm.generate(
            inputs_embeds=full_emb,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        # Decode only the newly generated part (after the prompt)
        gen_only = generated[:, full_emb.shape[1] :]
        return {"generated_ids": gen_only, "full_ids": generated}
