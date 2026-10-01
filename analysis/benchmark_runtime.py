"""Alternating process latency, with exact transcript comparison per run."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pairs',type=int,default=3)
    p.add_argument('--audio',required=True,help='Audio file for both runtimes; not bundled')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/process-benchmark')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    records=[];reference=None
    for pair in range(args.pairs):
        for kind in ('cpu','cuda') if pair%2==0 else ('cuda','cpu'):
            output=args.output/f'{pair}-{kind}.json'
            command=[str(ROOT/'transcribe.sh'),args.audio,'--no-warmup','--format','json','--output',str(output)] if kind=='cpu' else [str(ROOT/'transcribe_cuda.sh'),args.audio,'--output',str(output)]
            start=time.perf_counter()
            run=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            wall=time.perf_counter()-start
            (args.output/f'{pair}-{kind}.stderr').write_text(run.stderr)
            if run.returncode:raise RuntimeError(f'{kind}: {run.stderr}')
            data=json.loads(output.read_text())
            if reference is None:reference=data['text']
            equal=data['text']==reference
            records.append({'pair':pair,'runtime':kind,'wall_seconds':wall,'text_equal':equal,
                            'stages':data.get('seconds'),'memory':data.get('memory')})
            (args.output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
            print(json.dumps(records[-1]),flush=True)
            if not equal:raise AssertionError('Transcript differs')


if __name__=='__main__':main()
