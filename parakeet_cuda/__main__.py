# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from .gguf import Model
from .frontend import pcm_file,PCMSource,mel_features,pcm_identity
from .gpu import GPU,gpu_lock,ROOT
from .encoder import Encoder
from .decoder import CPUDecoder,CUDADecoder,decode
from .recording import wav_info,segments,words_from_tokens,merge_owned_words
from .paths import MODEL_DIR,OUTPUT_DIR


def main():
    p=argparse.ArgumentParser(prog='parakeet',description='Transcribe audio with Parakeet on CUDA; optionally label speakers and create a listening report.',
                             formatter_class=argparse.RawDescriptionHelpFormatter,
                             epilog='''Examples:
  parakeet recording.m4a
  parakeet recording.mp3 --diarize
  parakeet recording.wav --diarize --output speakers.json
  parakeet long-recording.wav --progress --output transcript.json

Relative input/output paths use your current directory.
With --diarize, HTML is saved beside --output, or in the installation's
outputs/<audio-name>.html when --output is omitted. Small audio is embedded;
large audio is linked to its original file. FFmpeg converts non-WAV input
once and the temporary WAV is removed automatically.''')
    p.add_argument('audio',help='Audio file: WAV, MP3, M4A, or another FFmpeg-supported format')
    p.add_argument('--model',default=str(MODEL_DIR/'parakeet-tdt-0.6b-v3.q8_0.gguf'),help='Parakeet GGUF path (default: downloaded Q8 model)')
    p.add_argument('--device',type=int,default=0,help='CUDA device index (default: 0)')
    p.add_argument('--vram-budget-mib',type=int,default=384,help='GPU arena allocation limit in MiB, excluding driver overhead (default: 384)')
    p.add_argument('--activation-policy',choices=['auto','spill'],default='auto',help='Automatic activation storage or force CPU spilling (default: auto)')
    p.add_argument('--decoder',choices=['cpu','cuda'],default='cuda',help='Transducer decoder backend (default: cuda)')
    p.add_argument('--mode',choices=['auto','full','segmented'],default='auto',
                   help='Auto segments recordings longer than --segment-seconds; full preserves utterance context')
    p.add_argument('--segment-seconds',type=float,default=45,help='Maximum ASR window in seconds (default: 45)')
    p.add_argument('--overlap-seconds',type=float,default=2,help='ASR overlap in seconds (default: 2)')
    p.add_argument('--tile',type=int,default=256,help='Advanced: projection tile size (default: 256)')
    p.add_argument('--stem-tile',type=int,default=64,help='Advanced: convolution stem tile size (default: 64)')
    p.add_argument('--query-tile',type=int,default=64,help='Advanced: spilled attention query tile size (default: 64)')
    p.add_argument('--key-tile',type=int,default=256,help='Advanced: spilled attention key tile size (default: 256)')
    p.add_argument('--layers',type=int,default=24,help='Diagnostic partial encoder; decoding requires 24')
    p.add_argument('--dump-encoder',type=Path,help='Diagnostic: save full-mode encoder activations as NumPy .npy')
    p.add_argument('--output',type=Path,help='Save transcript, timestamps and speaker labels as JSON; also creates HTML with --diarize')
    p.add_argument('--json',action='store_true',help='Print JSON to stdout instead of readable text')
    p.add_argument('--verbose',action='store_true',help='Show encoder and recording progress on stderr')
    p.add_argument('--progress',action='store_true',help='Report completed recording segments')
    p.add_argument('--diarize',action='store_true',help='Add speaker labels and automatically save an HTML listening report')
    p.add_argument('--diar-model',default=str(MODEL_DIR/'Nemotron-3-Diarization.q8_0.gguf'),help='Diarization GGUF path (default: downloaded Nemotron 3 Q8 model)')
    p.add_argument('--diar-probabilities',type=Path,help='Diagnostic: save native 10 ms speaker probabilities as .npy')
    if len(sys.argv)==1:
        p.print_help();return
    args=p.parse_args()
    if not 0<=args.layers<=24 or min(args.tile,args.stem_tile,args.query_tile,args.key_tile)<1:
        p.error('Invalid layer count or tile sizes')
    if not 5<=args.segment_seconds<=300 or not 0<=args.overlap_seconds<args.segment_seconds/3:
        p.error('Segment duration must be 5..300 seconds; overlap must be less than one third')
    if args.device<0:p.error('--device must be non-negative')
    if not Path(args.model).is_file():p.error(f'Model not found: {args.model}. Run parakeet-download first, or specify --model.')
    started=time.perf_counter();model=Model(args.model)
    diar_model=None
    if args.diarize:
        from .diarization import DiarModel,diarize,speaker_turns
        if not Path(args.diar_model).is_file():p.error(f'Diarization model not found: {args.diar_model}. Run parakeet-download --diarize first.')
        diar_model=DiarModel(args.diar_model)
        if args.layers!=24:p.error('Diarization requires a complete transcription')
    elif args.diar_probabilities:p.error('--diar-probabilities requires --diarize')
    with pcm_file(args.audio) as path,PCMSource(path) as source:
        samples=len(source)
        segmented=args.mode=='segmented' or (args.mode=='auto' and samples>args.segment_seconds*16000)
        if segmented and (args.dump_encoder or args.layers!=24):
            p.error('Partial encoder and encoder dumps require full mode')
        if not samples:p.error('Audio is empty')
        if not segmented and (samples//160+1+7)//8>5000:
            p.error('Full mode exceeds the 5000-frame positional limit; use segmented mode')
        with gpu_lock(args.device),GPU(args.vram_budget_mib,compact_cache_mib=104 if diar_model else 0,device=args.device) as gpu:
            if diar_model:gpu.cache_model(diar_model)
            engine=Encoder(model,gpu,args.tile,args.stem_tile,args.query_tile,args.key_tile)
            def infer(audio):
                front=time.perf_counter();features=mel_features(audio,model);frontend_seconds=time.perf_counter()-front
                enc_start=time.perf_counter()
                encoded=engine.run(features,spill=args.activation_policy=='spill',layers=args.layers,progress=args.verbose)
                encoder_seconds=time.perf_counter()-enc_start
                if args.dump_encoder:
                    args.dump_encoder.parent.mkdir(parents=True,exist_ok=True);np.save(args.dump_encoder,encoded)
                item={'text':'','pcm':pcm_identity(audio),'mel_frames':len(features),'encoder_frames':len(encoded)}
                if args.layers==24:
                    start=time.perf_counter();projected=engine.project(encoded)
                    projection_seconds=time.perf_counter()-start
                    start=time.perf_counter();decoder=CPUDecoder(model) if args.decoder=='cpu' else CUDADecoder(model,gpu)
                    item.update(decode(projected,decoder,model));decoder_seconds=time.perf_counter()-start
                    item['words']=words_from_tokens(item['tokens'],item['token_times'],model)
                else:projection_seconds=decoder_seconds=0
                item['seconds']={'frontend':frontend_seconds,'encoder':encoder_seconds,
                                 'projection':projection_seconds,'decoder':decoder_seconds}
                item['profile']=engine.profile
                return item
            if segmented:
                words=[];details=[];times={n:0. for n in ('frontend','encoder','projection','decoder')}
                for index,segment in enumerate(segments(path,args.segment_seconds,args.overlap_seconds,source=source)):
                    item=infer(segment.audio)
                    emitted=words_from_tokens(item['tokens'],item['token_times'],model,segment.offset)
                    owned=[w for w in emitted if segment.keep_start<=(w['start']+w['end'])/2<segment.keep_end]
                    merge_owned_words(words,owned)
                    details.append({'offset':segment.offset,'duration':len(segment.audio)/16000,
                                    'keep_start':segment.keep_start,'keep_end':segment.keep_end,
                                    'words':len(owned),'seconds':item['seconds'],'memory':gpu.stats()})
                    for name in times:times[name]+=item['seconds'][name]
                    if args.verbose or args.progress:
                        print(f'segment {index+1}: {segment.offset:.2f}s..{segment.offset+len(segment.audio)/16000:.2f}s',file=sys.stderr,flush=True)
                result={'text':' '.join(w['word'] for w in words),'words':words,'segments':details,
                        'pcm':{'samples':samples,'duration':samples/16000},'seconds':times}
            else:result=infer(source.read())
            if diar_model:
                result['diarization']=diarize(source,diar_model,gpu,result['words'],progress=args.progress or args.verbose,
                                             probability_output=args.diar_probabilities)
                result['speaker_turns']=speaker_turns(result['words'])
                result['seconds']['diarization']=result['diarization']['seconds']['total']
            result.update({'memory':gpu.stats(),'decoder':args.decoder,'mode':'segmented' if segmented else 'full'})
            result['seconds']['total']=time.perf_counter()-started
    if args.diarize:
        from .report import write_report
        report_path=args.output.with_suffix('.html') if args.output else OUTPUT_DIR/f'{Path(args.audio).stem}.html'
        if args.output and report_path.resolve()==args.output.resolve():
            report_path=args.output.with_name(args.output.stem+'.report.html')
        result['report_html']=str(report_path.resolve())
        write_report(result,args.audio,report_path)
        print(f'HTML listening report: {report_path.resolve()}',file=sys.stderr)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    if args.json:print(json.dumps(result,ensure_ascii=False))
    elif args.diarize:
        for turn in result['speaker_turns']:
            print(f"[{turn['start']:.2f}–{turn['end']:.2f}] {turn['speaker']}: {turn['text']}")
    else:print(result['text'])
    if not args.json:print(json.dumps({'seconds':result['seconds'],'memory':result['memory']}),file=sys.stderr)


if __name__=='__main__':main()
