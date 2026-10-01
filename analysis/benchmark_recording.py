"""One CUDA recording run with process-tree VRAM/RSS and progress samples."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import wave

ROOT=Path(__file__).resolve().parents[1]


def descendants(pid):
    found={pid}
    try:
        # uv can spawn Python from a worker thread rather than the thread leader.
        children=set()
        for path in Path(f'/proc/{pid}/task').glob('*/children'):
            try:children.update(path.read_text().split())
            except OSError:pass
    except (OSError,ValueError):children=[]
    for child in children:found.update(descendants(int(child)))
    return found


def rss_kib(pids):
    total=0
    for pid in pids:
        try:
            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                if line.startswith('VmRSS:'):total+=int(line.split()[1]);break
        except OSError:pass
    return total


def main():
    p=argparse.ArgumentParser();p.add_argument('audio',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--diarize',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if args.audio.suffix.lower()=='.wav':
        with wave.open(str(args.audio),'rb') as f:duration=f.getnframes()/f.getframerate()
    else:
        probe=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of',
                              'default=noprint_wrappers=1:nokey=1',str(args.audio)],capture_output=True,text=True,check=True)
        duration=float(probe.stdout.strip())
    command=[str(ROOT/'transcribe_cuda.sh'),str(args.audio.resolve()),'--progress','--output',str(args.output.resolve()/'transcript.json')]
    if args.diarize:command.append('--diarize')
    samples=[];offset=0;start=time.perf_counter()
    with (args.output/'stdout.txt').open('w') as out,(args.output/'stderr.txt').open('w') as err:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err)
        while proc.poll() is None:
            pids=descendants(proc.pid)
            query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
            gpu=[]
            for line in query.stdout.splitlines():
                fields=[x.strip() for x in line.split(',')]
                if len(fields)==2 and fields[0].isdigit() and int(fields[0]) in pids and fields[1].isdigit():gpu.append(int(fields[1]))
            sample={'elapsed_seconds':time.perf_counter()-start,'process_pids':sorted(pids),
                    'gpu_mib':sum(gpu) if gpu else None,'rss_kib':rss_kib(pids)}
            samples.append(sample)
            log=(args.output/'stderr.txt').read_text()
            for line in log[offset:].splitlines():
                if line.startswith(('segment ','diarization chunk ')):print(f'{sample["elapsed_seconds"]:.1f}s: {line}',flush=True)
            offset=len(log)
            time.sleep(.5)
        wall=time.perf_counter()-start
        if proc.returncode:raise RuntimeError((args.output/'stderr.txt').read_text())
    result=json.loads((args.output/'transcript.json').read_text())
    (args.output/'transcript.txt').write_text(result['text']+'\n')
    measured={'audio_seconds':duration,'wall_seconds':wall,'runtime_seconds':result['seconds'],
              'audio_seconds_per_wall_second':duration/wall,'segments':len(result.get('segments',[])),
              'peak_process_gpu_mib':max((x['gpu_mib'] for x in samples if x['gpu_mib'] is not None),default=None),
              'peak_process_tree_rss_mib':max((x['rss_kib'] for x in samples),default=0)/1024,
              'allocation':result['memory'],'samples':samples,'command':command}
    (args.output/'measurement.json').write_text(json.dumps(measured,indent=2)+'\n')
    print(json.dumps({k:v for k,v in measured.items() if k not in ('samples','command')},indent=2),flush=True)


if __name__=='__main__':main()
