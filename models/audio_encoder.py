"""
Audio Encoder using Whisper tiny or HuBERT base.

Extracts audio features from waveform for multimodal fusion.
"""

import torch
import torch.nn as nn
import torchaudio
from transformers import WhisperModel, WhisperProcessor, HubertModel, Wav2Vec2Processor


class WhisperAudioEncoder(nn.Module):
    """
    Whisper tiny encoder for audio feature extraction.

    audio_features = WhisperEncoder(waveform)
    Mean pool across time for audio-level representation.
    Output: 384-dimensional embedding (Whisper tiny), or 512 with projection.
    """

    def __init__(self, model_name: str = "openai/whisper-tiny", project_to_512: bool = True):
        super().__init__()
        self.model = WhisperModel.from_pretrained(model_name)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.raw_embed_dim = self.model.config.d_model  # 384 for whisper-tiny
        self.embed_dim = 512 if project_to_512 else self.raw_embed_dim
        self.project_to_512 = project_to_512
        if project_to_512:
            self.projection = nn.Linear(self.raw_embed_dim, 512)
        else:
            self.projection = nn.Identity()

        # Freeze encoder - feature extraction only
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self.embed_dim

    def forward(self, waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract audio features from waveform.

        Args:
            waveform: [batch, samples] or [samples] - raw audio waveform
            sample_rate: 16000 for Whisper

        Returns:
            audio_embedding: [batch, 384] - mean-pooled audio features
        """
        # Ensure batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Whisper expects 16kHz - resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(waveform.device)
            waveform = resampler(waveform)

        # Process each waveform - WhisperProcessor expects list of numpy arrays
        # Whisper encoder expects mel features of length 3000 (30 sec); pad if shorter
        WHISPER_MEL_LENGTH = 3000
        batch_embeddings = []
        for i in range(waveform.shape[0]):
            wav = waveform[i].cpu().numpy()
            inputs = self.processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
            input_features = inputs.input_features.to(waveform.device)
            # Pad or truncate to 3000 mel frames (Whisper's expected length)
            if input_features.shape[-1] < WHISPER_MEL_LENGTH:
                pad_len = WHISPER_MEL_LENGTH - input_features.shape[-1]
                input_features = torch.nn.functional.pad(input_features, (0, pad_len), value=0)
            elif input_features.shape[-1] > WHISPER_MEL_LENGTH:
                input_features = input_features[:, :, :WHISPER_MEL_LENGTH]
            with torch.no_grad():
                encoder_outputs = self.model.encoder(input_features)
            emb = encoder_outputs.last_hidden_state.mean(dim=1)  # [1, D]
            batch_embeddings.append(emb)
        audio_embedding = torch.cat(batch_embeddings, dim=0)
        return self.projection(audio_embedding)


class HubertAudioEncoder(nn.Module):
    """
    HuBERT base encoder for audio feature extraction.

    Alternative to Whisper - often better for non-speech sounds.
    Output: 768-dimensional embedding.
    """

    def __init__(self, model_name: str = "facebook/hubert-base-ls960"):
        super().__init__()
        self.model = HubertModel.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.embed_dim = self.model.config.hidden_size  # 768

        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self.embed_dim

    def forward(self, waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract audio features from waveform.

        Args:
            waveform: [batch, samples] or [samples] - raw audio
            sample_rate: 16000 for HuBERT

        Returns:
            audio_embedding: [batch, 768] - mean-pooled features
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(waveform.device)
            waveform = resampler(waveform)

        batch_embeddings = []
        for i in range(waveform.shape[0]):
            wav = waveform[i].cpu().numpy()
            inputs = self.processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(waveform.device)
            with torch.no_grad():
                outputs = self.model(input_values)
            emb = outputs.last_hidden_state.mean(dim=1)
            batch_embeddings.append(emb)
        return torch.cat(batch_embeddings, dim=0)
