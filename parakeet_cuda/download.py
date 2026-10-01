# SPDX-License-Identifier: Apache-2.0
"""Download exactly the supported GGUF files and verify pinned SHA-256 hashes."""
import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request
from .paths import MODEL_DIR

MODELS={
 'asr':{'repo':'nvidia/parakeet-tdt-0.6b-v3','revision':'541d1f99c6b0c3cd0b11a95167540bb8edefd82b',
        'filename':'parakeet-tdt-0.6b-v3.q8_0.gguf','size':713975456,
        'sha256':'e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e','license':'CC-BY-4.0'},
 'diarization':{'repo':'nvidia/Nemotron-3-Diarization','revision':'f667ed73aee57d40cc39428eb768b4fd87a0a29e',
        'filename':'Nemotron-3-Diarization.q8_0.gguf','size':107012128,
        'sha256':'08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1','license':'OpenMDW-1.1'},
}


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as file:
        while chunk:=file.read(1024**2):result.update(chunk)
    return result.hexdigest()


def verified(path,model):
    return path.is_file() and path.stat().st_size==model['size'] and digest(path)==model['sha256']


def download(model,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    target=directory/model['filename']
    if verified(target,model):print(f'Already verified: {target}');return target
    url=f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{model['filename']}"
    print(f"Downloading {model['filename']} ({model['size']/1024**2:.2f} MiB); license: {model['license']}",flush=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=directory,prefix=model['filename']+'.',suffix='.partial',delete=False) as output:
            temporary=Path(output.name)
            with urllib.request.urlopen(url) as source:
                while chunk:=source.read(1024**2):output.write(chunk)
        if not verified(temporary,model):raise ValueError(f'Integrity verification failed for {model["filename"]}')
        temporary.replace(target)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    print(f'Verified: {target}');return target


def main():
    parser=argparse.ArgumentParser(description='Download only pinned GGUF weights; no framework or full checkpoint downloads')
    parser.add_argument('--diarize',action='store_true',help='Also download the 102 MiB diarization GGUF')
    parser.add_argument('--diarization-only',action='store_true',help='Download only the diarization GGUF')
    parser.add_argument('--model-dir',type=Path,default=MODEL_DIR)
    parser.add_argument('--verify-only',action='store_true',help='Check local size/SHA-256 without network access')
    args=parser.parse_args()
    selected=[] if args.diarization_only else ['asr']
    if args.diarize or args.diarization_only:selected.append('diarization')
    for key in selected:
        model=MODELS[key]
        if args.verify_only:
            path=args.model_dir/model['filename']
            if not verified(path,model):parser.exit(1,f'Missing or invalid model: {path}\n')
            print(f'Verified: {path}')
        else:download(model,args.model_dir)


if __name__=='__main__':main()
