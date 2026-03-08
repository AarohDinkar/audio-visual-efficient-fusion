"""
Audio-Visual Captioner using llama-cpp-python for LLM inference.

Uses BLIP (vision) + Whisper (audio) -> text prompt -> Llama/Gemma GGUF.
No gradient flow - inference only. For training, use multimodal_model.py (transformers).
"""

from typing import Optional

import torch
import torch.nn as nn

from .vision_encoder import CLIPVisionEncoder
from .audio_encoder import WhisperAudioEncoder


def load_llama_cpp(repo_id: str, filename: str):
    """Load Llama from Hugging Face via llama-cpp-python."""
    from llama_cpp import Llama
    return Llama.from_pretrained(repo_id=repo_id, filename=filename)


class AudioVisualCaptionerLlamaCpp(nn.Module):
    """
    Audio-visual captioning using llama-cpp for inference.

    Pipeline: Vision (BLIP) + Audio (Whisper) -> text prompt -> Llama/Gemma.
    Uses text-based prompting since llama-cpp does not support input embeddings.
    """

    def __init__(
        self,
        blip_model: str = "Salesforce/blip-image-captioning-base",
        whisper_model: str = "openai/whisper-tiny",
        llama_repo_id: str = "google/gemma-2b-it-GGUF",
        llama_filename: str = "gemma-2b-it-Q4_K_M.gguf",
    ):
        super().__init__()
        from transformers import BlipProcessor, BlipForConditionalGeneration, WhisperProcessor, WhisperForConditionalGeneration

        # BLIP for vision caption
        self.blip_processor = BlipProcessor.from_pretrained(blip_model)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model)

        # Whisper for audio transcription
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model)

        # Llama-cpp for text generation (lazy load - loaded on first inference)
        self._llm = None
        self.llama_repo_id = llama_repo_id
        self.llama_filename = llama_filename

        # Move BLIP/Whisper to CPU by default (llama-cpp runs separately)
        self.blip_model.eval()
        self.whisper_model.eval()

    @property
    def llm(self):
        if self._llm is None:
            self._llm = load_llama_cpp(self.llama_repo_id, self.llama_filename)
        return self._llm

    def _vision_to_text(self, frames: torch.Tensor, device: torch.device) -> str:
        """Generate caption from middle frame using BLIP."""
        # Handle [B, T, 3, 224, 224] or [T, 3, 224, 224]
        if frames.dim() == 5:
            mid = frames.shape[1] // 2
            frame = frames[0, mid]  # [3, 224, 224]
        else:
            mid = frames.shape[0] // 2
            frame = frames[mid]  # [3, 224, 224]
        img = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        from PIL import Image
        img = Image.fromarray(img)
        inputs = self.blip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_length=50)
        return self.blip_processor.decode(out[0], skip_special_tokens=True)

    def _audio_to_text(self, audio: torch.Tensor, device: torch.device) -> str:
        """Transcribe audio using Whisper."""
        if audio.dim() == 2:
            audio = audio.squeeze(0)
        wav = audio.cpu().numpy()
        inputs = self.whisper_processor(wav, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(input_features)
        transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip() if transcription else "audio present"

    def forward(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
        captions: Optional[list] = None,
        max_new_tokens: int = 150,
    ) -> dict:
        """
        Generate caption from video using text-based prompt + Llama.

        Args:
            frames: [B, T, 3, 224, 224]
            audio: [B, samples]
            captions: ignored (no training with this model)
            max_new_tokens: for generation

        Returns:
            dict with "generated_ids" (as decoded text in "generated_text")
        """
        device = next(self.blip_model.parameters()).device
        frames = frames.to(device)
        audio = audio.to(device)

        # Get text from vision and audio
        vision_text = self._vision_to_text(frames, device)
        audio_text = self._audio_to_text(audio, device)

        # Handle batch: process each sample
        if frames.dim() == 5:
            batch_size = frames.shape[0]
            texts = []
            for i in range(batch_size):
                v_text = self._vision_to_text(frames[i], device)
                a_text = self._audio_to_text(audio[i], device)
                prompt = f"Video shows: {v_text}. Audio: {a_text}. Describe this video in one sentence:"
                out = self.llm(prompt, max_tokens=max_new_tokens, echo=False, stop=None)
                t = out["choices"][0]["text"].strip() if out.get("choices") else ""
                texts.append(t)
            return {"generated_ids": None, "generated_text": texts[0] if len(texts) == 1 else texts}
        else:
            prompt = f"Video shows: {vision_text}. Audio: {audio_text}. Describe this video in one sentence:"
            out = self.llm(prompt, max_tokens=max_new_tokens, echo=False, stop=None)
            text = out["choices"][0]["text"].strip() if out.get("choices") else ""
            return {"generated_ids": None, "generated_text": text}
