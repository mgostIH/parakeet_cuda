#!/usr/bin/env python3
"""Fetch pinned small source archives for the optional C++ oracle, no weights."""
from pathlib import Path
import tarfile
import tempfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM='4c101bc7113f49101a3e11d2c994c519f41939f6'
GGML='c03b4e2bcece5134827881af90242086daf75be5'


def fetch(repo,revision,target):
    if (target/'CMakeLists.txt').exists():print(f'Already present: {target}');return
    target.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        archive=Path(directory)/'source.tar.gz'
        urllib.request.urlretrieve(f'https://codeload.github.com/{repo}/tar.gz/{revision}',archive)
        print(f'{repo}: {archive.stat().st_size} compressed bytes')
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                parts=member.name.split('/',1)
                if len(parts)==2 and parts[1]:
                    member.name=parts[1];tar.extract(member,target,filter='data')


if __name__=='__main__':
    fetch('NVIDIA/NeMo-Speech.cpp',UPSTREAM,ROOT/'nemo-speech-src')
    fetch('ggml-org/ggml',GGML,ROOT/'nemo-speech-src/ggml')
