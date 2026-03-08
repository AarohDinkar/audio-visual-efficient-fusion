"""
Configuration for Audio-Visual Multimodal Understanding.
"""

from pathlib import Path

# Paths (resolved to absolute for robustness)
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = (PROJECT_ROOT / "data").resolve()
VIDEOS_DIR = (DATA_DIR / "videos").resolve()
FRAMES_DIR = (DATA_DIR / "frames").resolve()
AUDIO_DIR = (DATA_DIR / "audio").resolve()
CAPTIONS_FILE = (DATA_DIR / "captions.json").resolve()
PROCESSED_DIR = (DATA_DIR / "processed").resolve()

# Model configs
VISION_ENCODER = "openai/clip-vit-base-patch32"
AUDIO_ENCODER = "openai/whisper-tiny"  # or "facebook/hubert-base-ls960"
BLIP_MODEL = "Salesforce/blip-image-captioning-base"

# LLM: "transformers" (for training) or "llama_cpp" (for inference with llama-cpp-python)
LLM_BACKEND = "llama_cpp"

# Transformers backend (for training - requires gradients)
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Llama-cpp backend (for inference - uses GGUF, no gradients)
# Requires: pip install llama-cpp-python, huggingface-hub
# TinyLlama GGUF - non-gated, no login required
LLAMA_CPP_REPO_ID = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
LLAMA_CPP_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Processing
NUM_FRAMES = 8
FRAME_SIZE = 224
AUDIO_SAMPLE_RATE = 16000
AUDIO_MAX_DURATION = 10  # seconds

# LLM dimensions (Gemma-2B)
GEMMA_EMBED_DIM = 2048  # gemma-2b uses 2048
PHI_EMBED_DIM = 3072   # Phi-3-mini uses 3072

# MSR-VTT
MSR_VTT_URL = "https://www.rocq.inria.fr/cluster-willow/amiech/msr-vtt.zip"
MSR_VTT_SPLIT_URL = "https://github.com/ArrowLuo/CLIP4Clip/raw/main/dataloaders/msrvtt_data.zip"
