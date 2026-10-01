#!/usr/bin/env python3
"""Reject models, private media/output and build products in release archives."""
import argparse
from pathlib import Path,PurePosixPath
import tarfile
import zipfile

FORBIDDEN_SUFFIXES={'.gguf','.nemo','.safetensors','.pt','.pth','.onnx','.wav','.mp3','.m4a','.flac','.ogg','.opus','.aac','.mp4','.webm','.mkv','.html','.npy','.npz','.cubin','.ptx','.so','.a','.log'}
FORBIDDEN_DIRS={'models','runtime','nemo-speech-src','artifacts','outputs','.venv','.cache','.jj','.git'}
FORBIDDEN_FILES={'REIMPLEMENTATION_PLAN.md'}


def check(path):
    if path.suffix=='.whl':
        with zipfile.ZipFile(path) as archive:names=archive.namelist()
    else:
        with tarfile.open(path) as archive:names=archive.getnames()
    for name in names:
        item=PurePosixPath(name)
        if item.name in FORBIDDEN_FILES or item.suffix.lower() in FORBIDDEN_SUFFIXES or any(part in FORBIDDEN_DIRS or part.startswith('.env') for part in item.parts):
            raise ValueError(f'Excluded content in {path}: {name}')
    if not any(name.endswith('parakeet_cuda/kernels/runtime.cu') for name in names):
        raise ValueError(f'CUDA source missing from {path}')
    print(f'Clean release: {path} ({len(names)} entries, {path.stat().st_size} bytes)')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archives',type=Path,nargs='+')
    args=parser.parse_args()
    for path in args.archives:check(path)
