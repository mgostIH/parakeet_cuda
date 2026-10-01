# SPDX-License-Identifier: Apache-2.0
"""Compile ordinary CUDA source with nvcc; cache by source/toolchain/target."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from .paths import CACHE_DIR,KERNEL_SOURCE


def find_nvcc(arch):
    explicit=os.environ.get('PARAKEET_NVCC') or os.environ.get('TXIR_NVCC')
    installed=sorted(Path('/usr/local').glob('cuda-12*/bin/nvcc'),
                     key=lambda p:tuple(int(n) for n in re.findall(r'\d+',p.parts[-3])),reverse=True)
    candidates=[explicit] if explicit else [*installed,shutil.which('nvcc'),'/usr/local/cuda/bin/nvcc','/usr/bin/nvcc']
    errors=[]
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():continue
        compiler=Path(candidate).resolve()
        version=subprocess.run([str(compiler),'--version'],capture_output=True,text=True,check=True).stdout
        match=re.search(r'release (\d+)\.',version)
        if not match or int(match.group(1))<12:
            errors.append(f'{compiler}: CUDA 12 or later is required');continue
        supported=subprocess.run([str(compiler),'--list-gpu-code'],capture_output=True,text=True,check=True).stdout.split()
        if arch not in supported:
            errors.append(f'{compiler} does not support {arch}');continue
        return compiler,version
    details='; '.join(errors)
    raise RuntimeError(f'nvcc supporting {arch} was not found. Install CUDA 12.9 for Maxwell or set PARAKEET_NVCC to a compatible nvcc. {details}')


def compile_runtime(arch,cache_dir=None):
    if not re.fullmatch(r'sm_\d+',arch):raise ValueError('Architecture must look like sm_50 or sm_86')
    compiler,version=find_nvcc(arch)
    source=KERNEL_SOURCE.read_bytes()
    identity={'source_sha256':hashlib.sha256(source).hexdigest(),'arch':arch,
              'nvcc':str(compiler),'nvcc_version':version.strip(),'flags':['-O3','-std=c++17','-lineinfo'],
              'cache_version':1}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    cache=Path(cache_dir or CACHE_DIR);cache.mkdir(parents=True,exist_ok=True)
    target=cache/key
    # Build atomically; concurrent processes cannot see partial cubins.
    with (cache/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (target/'compile.json').is_file() and (target/'kernel.cubin').is_file():return target/'kernel.cubin'
        with tempfile.TemporaryDirectory(prefix=key+'-',dir=cache) as temp:
            staging=Path(temp);(staging/'kernel.cu').write_bytes(source)
            commands=[];logs=[]
            for kind in ('ptx','cubin'):
                command=[str(compiler),'--'+kind,*identity['flags'],'-arch='+arch,str(staging/'kernel.cu'),'-o',str(staging/('kernel.'+kind))]
                run=subprocess.run(command,capture_output=True,text=True)
                commands.append(command);logs.append(run.stdout+run.stderr)
                if run.returncode:
                    failure=cache/(key+'.failure.log');failure.write_text('\n'.join(logs))
                    raise RuntimeError(f'nvcc failed compiling {arch}; see {failure}')
            (staging/'compile.log').write_text('\n'.join(logs))
            (staging/'compile.json').write_text(json.dumps({**identity,'commands':commands},indent=2)+'\n')
            if target.exists():shutil.rmtree(target)
            staging.rename(target)
    return target/'kernel.cubin'


def main():
    parser=argparse.ArgumentParser(description='Precompile Parakeet CUDA kernels; no model weights or GPU needed')
    parser.add_argument('--arch',default='sm_50',help='GPU architecture (default: sm_50 for GTX 750 Ti)')
    parser.add_argument('--cache-dir',type=Path)
    args=parser.parse_args()
    try:print(compile_runtime(args.arch,args.cache_dir))
    except (RuntimeError,ValueError) as error:parser.exit(1,f'{error}\n')


if __name__=='__main__':main()
