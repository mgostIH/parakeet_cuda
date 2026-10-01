"""Compatibility entry point: download only the small diarization GGUF."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from parakeet_cuda.download import MODELS,download
from parakeet_cuda.paths import MODEL_DIR

if __name__=='__main__':download(MODELS['diarization'],MODEL_DIR)
