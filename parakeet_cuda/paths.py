# SPDX-License-Identifier: Apache-2.0
"""Checkout and installed-package paths with explicit user overrides."""
import os
from pathlib import Path

PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parent
CHECKOUT=(ROOT/'pyproject.toml').is_file() and (ROOT/'transcribe_cuda.sh').is_file()
DATA_HOME=ROOT if CHECKOUT else Path(os.environ.get('XDG_DATA_HOME',Path.home()/'.local/share'))/'parakeet-cuda'
MODEL_DIR=Path(os.environ.get('PARAKEET_MODEL_DIR',DATA_HOME/'models')).expanduser()
OUTPUT_DIR=DATA_HOME/'outputs'
CACHE_DIR=Path(os.environ.get('PARAKEET_CACHE_DIR',
    ROOT/'artifacts/compile' if CHECKOUT else Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'parakeet-cuda/kernels')).expanduser()
KERNEL_SOURCE=PACKAGE/'kernels/runtime.cu'
