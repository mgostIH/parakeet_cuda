# SPDX-License-Identifier: Apache-2.0
"""Nemotron 3 Q8/FP32 CUDA inference with bounded arrival-order speaker state.

Architecture and AOSC semantics follow NVIDIA/NeMo-Speech.cpp (Apache-2.0).
GPU storage is shared with Parakeet; waveform and predictions are mapped on host.
"""
from dataclasses import dataclass, asdict
import math
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
from .gguf import Model


class DiarModel(Model):
    def __init__(self,path):
        super().__init__(path,validate=False)
        required={'general.architecture':'sortformer','sortformer.version':'v3',
                  'sortformer.encoder.d_model':512,'sortformer.encoder.n_layers':31,
                  'sortformer.encoder.n_heads':8,'sortformer.encoder.d_ff':2048,
                  'sortformer.encoder.subsampling_factor':8,'sortformer.encoder.feat_in':128,
                  'sortformer.encoder.qkv_bias':False,'sortformer.encoder.qk_norm':False,
                  'sortformer.encoder.pre_block_norm':True,'sortformer.encoder.rope_base':10000.,
                  'sortformer.encoder.rotary_fraction':1.,'sortformer.encoder.xscaling':False,
                  'sortformer.preprocessor.sample_rate':16000,'sortformer.preprocessor.n_fft':512,
                  'sortformer.preprocessor.features':128,'sortformer.preprocessor.normalize':'NA',
                  'sortformer.high_resolution':True,'sortformer.output_subsampling_factor':1,
                  'sortformer.upsample_factor':8,'sortformer.learnable_silence':True,
                  'sortformer.transformer.hidden_size':192,'sortformer.num_speakers':8}
        for key,value in required.items():
            if self.metadata.get(key)!=value:raise ValueError(f'Unsupported diarizer metadata: {key}')
        def need(name,shape,kind):
            t=self.tensors.get(name)
            if t is None or t.shape!=tuple(shape) or t.kind!=kind:
                raise ValueError(f'Unsupported diarizer tensor: {name}')
        need('preprocessor.fb',(128,257),0);need('learnable_sil_emb',(512,),0)
        need('encoder.pre_encode.proj.weight',(512,1024),8)
        for norm in ('encoder.embed_norm','encoder.final_norm'):
            for suffix in ('weight','bias'):need(norm+'.'+suffix,(512,),0)
        for i in range(31):
            b=f'encoder.layers.{i}.'
            for norm in ('norm1','norm2'):
                for suffix in ('weight','bias'):need(b+norm+'.'+suffix,(512,),0)
            need(b+'attn.w_qkv.weight',(1536,512),8)
            for name,shape in (('attn.out_proj',(512,512)),('ffn.net.0',(2048,512)),('ffn.net.3',(512,2048))):
                need(b+name+'.weight',shape,8);need(b+name+'.bias',(shape[0],),0)
        for name,shape,kind in (('encoder_proj',(192,512),8),('subpixel_upsample',(1536,192,3),1),
                                ('head.first_hidden_to_hidden',(192,192),0),('head.single_hidden_to_spks',(8,192),0)):
            need(name+'.weight',shape,kind);need(name+'.bias',(shape[0],),0)
        self.fb=self.array('preprocessor.fb')


@dataclass(frozen=True)
class Geometry:
    cache: int=264
    fifo: int=40
    chunk: int=340
    update: int=300
    left: int=0
    right: int=40

    def validate(self,model):
        if self.chunk<1 or self.update<1 or min(self.fifo,self.left,self.right)<0:
            raise ValueError('Invalid diarization streaming geometry')
        minimum=(1+model.metadata['sortformer.scoring.spkcache_sil_frames_per_spk'])*8
        if self.cache<minimum:raise ValueError('Speaker cache is too small')
        total=self.cache+self.fifo+self.chunk+self.left+self.right
        # Dense attention is bounded by the declared scratch arena, independent of audio length.
        if total>768:raise ValueError('Diarizer geometry exceeds the 768-frame CUDA memory contract')


class SpeakerState:
    """NumPy port of upstream inference AOSC; arrays remain bounded by geometry."""
    def __init__(self,model,geometry):
        self.geo=geometry
        self.sc={k.rsplit('.',1)[-1]:v for k,v in model.metadata.items() if k.startswith('sortformer.scoring.')}
        self.silence=model.array('learnable_sil_emb').copy()
        self.cache=np.empty((0,512),np.float32)
        self.fifo=np.empty((0,512),np.float32)
        self.cache_probs=None
        self.compressions=0
        self.max_cache=self.max_fifo=0

    def update(self,embeddings,probabilities,left,right):
        ncache,nfifo=len(self.cache),len(self.fifo)
        valid=len(embeddings)-left-right
        if valid<=0:return
        chunk=probabilities[ncache+nfifo+left:ncache+nfifo+left+valid]
        fifo_probs=np.concatenate((probabilities[ncache:ncache+nfifo],chunk))
        self.fifo=np.concatenate((self.fifo,embeddings[left:left+valid]))
        if len(self.fifo)>self.geo.fifo:
            pop=min(len(self.fifo),max(self.geo.update,len(self.fifo)-self.geo.fifo))
            self.cache=np.concatenate((self.cache,self.fifo[:pop]))
            if self.cache_probs is not None:
                self.cache_probs=np.concatenate((self.cache_probs,fifo_probs[:pop]))
            elif len(self.cache)>self.geo.cache:
                self.cache_probs=np.concatenate((probabilities[:ncache],fifo_probs[:pop]))
            self.fifo=self.fifo[pop:].copy()
            if len(self.cache)>self.geo.cache:self.compress()
        self.max_cache=max(self.max_cache,len(self.cache));self.max_fifo=max(self.max_fifo,len(self.fifo))
        assert len(self.cache)<=self.geo.cache and len(self.fifo)<=self.geo.fifo

    def compress(self):
        p=self.cache_probs;n=len(p);cap=self.geo.cache;sc=self.sc
        floor=np.float32(sc['pred_score_threshold']);half=np.float32(np.log(.5))
        log1=np.log(np.maximum(1-p,floor))
        scores=np.log(np.maximum(p,floor))-log1+log1.sum(axis=1,keepdims=True)-half
        scores[p<=.5]=-np.inf
        per=cap//8-sc['spkcache_sil_frames_per_spk']
        clean=(scores>0).sum(axis=0)>=math.floor(per*sc['min_pos_scores_rate'])
        scores[:,clean]=np.where(scores[:,clean]>0,scores[:,clean],-np.inf)
        scores[cap:]+=np.float32(sc['scores_boost_latest'])
        for rate,boost in (('strong_boost_rate',-2*half),('weak_boost_rate',-half)):
            count=min(n,math.floor(per*sc[rate]))
            if count:
                ids=np.argsort(-scores,axis=0,kind='stable')[:count]
                scores[ids,np.arange(8)]+=boost
        padded=np.pad(scores,((0,sc['spkcache_sil_frames_per_spk']),(0,0)),constant_values=np.inf)
        flat=padded.T.reshape(-1)
        chosen=np.argsort(-flat,kind='stable')[:cap]
        disabled=np.isneginf(flat[chosen])
        chosen[disabled]=99999*len(padded)+99999
        chosen.sort()
        indices=chosen%len(padded)
        good=(chosen<8*len(padded))&(indices<n)
        cache=np.tile(self.silence,(cap,1));probs=np.zeros((cap,8),np.float32)
        cache[good]=self.cache[indices[good]];probs[good]=p[indices[good]]
        self.cache,self.cache_probs=cache,probs
        self.compressions+=1


class DiarEncoder:
    def __init__(self,model,gpu):
        self.model,self.g=model,gpu

    def linear(self,x,name,rows,k,n,out=None,bias=True):
        g=self.g;out=g.scratch.alloc((rows,n)) if out is None else out
        g.blas.linear(x,g.weight(name+'.weight'),out,rows,k,n)
        if bias and name+'.bias' in g.weight_views:
            g.element('bias_act',[out,g.weight(name+'.bias'),n,rows*n,0],rows*n)
        return out

    def norm(self,x,name,rows,out=None):
        g=self.g;out=g.scratch.alloc((rows,512)) if out is None else out
        g.launch('layer_norm',[x,g.weight(name+'.weight'),g.weight(name+'.bias'),out,512],rows)
        return out

    def block(self,x,base,t):
        g=self.g;mark=g.scratch.offset
        xn=self.norm(x,base+'.norm1',t)
        packed=self.linear(xn,base+'.attn.w_qkv',t,512,1536)
        q,k,v=[g.scratch.alloc((8,t,64)) for _ in range(3)]
        g.element('diar_rope_qkv',[packed,q,k,v,t],t*512)
        scores=g.scratch.alloc((8,t,t));g.blas.scores(q,k,scores,t,t,width=64)
        g.launch('softmax_scaled',[scores,t,.125],8*t)
        heads=g.scratch.alloc((8,t,64));g.blas.values(scores,v,heads,t,t,width=64)
        merged=g.scratch.alloc((t,512));g.element('diar_merge_heads',[heads,merged,t],t*512)
        projected=self.linear(merged,base+'.attn.out_proj',t,512,512)
        g.element('residual',[x,projected,1.,t*512],t*512)
        g.scratch.reset(mark)
        rn=self.norm(x,base+'.norm2',t)
        ff=self.linear(rn,base+'.ffn.net.0',t,512,2048,bias=False)
        g.element('gelu_bias',[ff,g.weight(base+'.ffn.net.0.bias'),2048,t*2048],t*2048)
        down=self.linear(ff,base+'.ffn.net.3',t,2048,512)
        g.element('residual',[x,down,1.,t*512],t*512)
        g.scratch.reset(mark)

    def run_chunk(self,mel,state=None,layers=31):
        g=self.g;model=self.model
        pad=(-len(mel))%8
        stacked=np.pad(mel,((0,pad),(0,0))).reshape(-1,1024)
        g.scratch.reset();g.load_weights(model,['encoder.pre_encode.proj.weight'])
        inp=g.scratch.alloc(stacked.shape);g.upload(inp,stacked)
        emb=self.linear(inp,'encoder.pre_encode.proj',len(stacked),1024,512)
        embeddings=g.download(emb)
        prefix=np.empty((0,512),np.float32) if state is None else np.concatenate((state.cache,state.fifo))
        initial=np.concatenate((prefix,embeddings));t=len(initial)
        if t>768:raise ValueError('Diarizer input exceeds bounded CUDA contract')
        g.scratch.reset();x=g.scratch.alloc(initial.shape);g.upload(x,initial)
        g.load_weights(model,model.names('encoder.embed_norm.'));self.norm(x,'encoder.embed_norm',t,x)
        for i in range(layers):
            base=f'encoder.layers.{i}'
            g.load_weights(model,model.names(base+'.'));self.block(x,base,t)
        if layers!=31:return g.download(x),embeddings
        g.load_weights(model,model.names('encoder.final_norm.')+model.names('encoder_proj.')+
                       model.names('subpixel_upsample.')+model.names('head.first_hidden_to_hidden.')+
                       model.names('head.single_hidden_to_spks.'))
        self.norm(x,'encoder.final_norm',t,x)
        proj=self.linear(x,'encoder_proj',t,512,192)
        columns=g.scratch.alloc((t,576));g.element('diar_conv_columns',[proj,columns,t],t*576)
        up=self.linear(columns,'subpixel_upsample',t,576,1536,bias=False)
        g.element('bias_act',[up,g.weight('subpixel_upsample.bias'),1536,t*1536,1],t*1536)
        hidden=self.linear(up,'head.first_hidden_to_hidden',t*8,192,192,bias=False)
        g.element('bias_act',[hidden,g.weight('head.first_hidden_to_hidden.bias'),192,t*1536,1],t*1536)
        logits=self.linear(hidden,'head.single_hidden_to_spks',t*8,192,8,bias=False)
        g.element('sigmoid_bias',[logits,g.weight('head.single_hidden_to_spks.bias'),8,t*64],t*64)
        return g.download(logits),embeddings


def mel_window(source,start,end,model):
    """Global-clock raw log-mels; real context at internal boundaries, zero at file edges."""
    lo=start*160-256;hi=(end-1)*160+256
    first=max(0,lo);last=min(len(source),hi)
    signal=source.read(first,max(0,last-first))
    if len(signal):
        previous=source.read(first-1,1)[0] if first else 0.
        pre=signal.copy();pre[1:]-=np.float32(model.metadata['sortformer.preprocessor.preemph'])*signal[:-1]
        if first:pre[0]-=np.float32(model.metadata['sortformer.preprocessor.preemph'])*previous
    else:pre=signal
    padded=np.pad(pre,(max(0,-lo),max(0,hi-len(source))))
    frames=np.lib.stride_tricks.sliding_window_view(padded,512)[::160][:end-start]
    window=np.zeros(512,np.float32)
    phase=np.float32(2*np.pi)*np.arange(400,dtype=np.float32)/np.float32(399)
    window[56:456]=np.float32(.5)*(1-np.cos(phase))
    out=np.empty((end-start,128),np.float32)
    for i in range(0,len(out),512):
        fft=np.fft.rfft(frames[i:i+512]*window,axis=1)
        power=(fft.real**2+fft.imag**2).astype(np.float32)
        out[i:i+512]=np.log(power@model.fb.T+np.float32(model.metadata['sortformer.preprocessor.log_zero_guard']))
    return out


def speaker_segments(probs,duration,onset=.641,offset=.561,pad_onset=.229,pad_offset=.079,
                     min_on=.511,min_off=.296):
    """Upstream hysteresis -> padding -> gap merge -> short segment removal."""
    out=[]
    for speaker in range(8):
        active=False;begin=0;spans=[]
        for frame,p in enumerate(probs[:,speaker]):
            if not active and p>onset:active=True;begin=frame
            elif active and p<offset:
                active=False;spans.append((max(0,begin*.01-pad_onset),min(duration,frame*.01+pad_offset)))
        if active:spans.append((max(0,begin*.01-pad_onset),duration))
        merged=[]
        for a,b in spans:
            if merged and a-merged[-1][1]<min_off:merged[-1]=(merged[-1][0],max(merged[-1][1],b))
            else:merged.append((a,b))
        out.extend({'start':a,'end':b,'speaker':f'speaker_{speaker+1}'} for a,b in merged if b-a>=min_on)
    return sorted(out,key=lambda s:(s['start'],s['speaker']))


def tag_words(words,probs):
    """Average first 160 ms of each word, matching upstream's word anchor."""
    for word in words:
        start=min(max(0,int(word['start']*100)),len(probs)-1)
        end=max(start+1,min(len(probs),int(math.ceil(word['end']*100)),start+16))
        activity=np.asarray(probs[start:end]).mean(axis=0)
        best=int(activity.argmax())
        word['speaker']=f'speaker_{best+1}' if activity[best]>=.3 else 'unknown'
        word['speaker_confidence']=float(activity[best])
        # This reports simultaneous activity, not separated overlapping transcripts.
        word['active_speakers']=[f'speaker_{i+1}' for i,p in enumerate(activity) if p>=.5]


def speaker_turns(words):
    turns=[]
    for word in words:
        if turns and turns[-1]['speaker']==word['speaker'] and word['start']-turns[-1]['end']<=1.:
            turns[-1]['end']=max(turns[-1]['end'],word['end']);turns[-1]['text']+=' '+word['word']
        else:turns.append({'start':word['start'],'end':word['end'],'speaker':word['speaker'],'text':word['word']})
    return turns


def diarize(source,model,gpu,words,geometry=None,progress=False,probability_output=None):
    geometry=geometry or Geometry();geometry.validate(model)
    state=SpeakerState(model,geometry);engine=DiarEncoder(model,gpu)
    count=len(source)//160+1;duration=len(source)/16000
    started=time.perf_counter();frontend=inference=0.;chunks=0
    # A temporary mapped timeline bounds Python heap usage and allows word alignment
    # without retaining the waveform/activations for the whole recording.
    with tempfile.TemporaryDirectory(prefix='parakeet-diar-') as directory:
        probabilities=np.memmap(Path(directory)/'probs.f32',mode='w+',dtype=np.float32,shape=(count,8))
        for start in range(0,count,geometry.chunk*8):
            end=min(count,start+geometry.chunk*8)
            left=min(geometry.left*8,start);right=min(geometry.right*8,count-end)
            now=time.perf_counter();mel=mel_window(source,start-left,end+right,model);frontend+=time.perf_counter()-now
            now=time.perf_counter();preds,embs=engine.run_chunk(mel,state);inference+=time.perf_counter()-now
            prefix=len(state.cache)+len(state.fifo);lc=int(math.floor(left/8+.5));rc=int(math.ceil(right/8))
            coarse=preds.reshape(-1,8,8).mean(axis=1)
            state.update(embs,coarse,lc,rc)
            off=(prefix+lc)*8
            probabilities[start:end]=preds[off:off+end-start]
            chunks+=1
            if progress:print(f'diarization chunk {chunks}: {min(end*.01,duration):.2f}s / {duration:.2f}s',file=sys.stderr,flush=True)
        tag_words(words,probabilities)
        spans=speaker_segments(probabilities,duration)
        if probability_output:
            probability_output=Path(probability_output);probability_output.parent.mkdir(parents=True,exist_ok=True)
            np.save(probability_output,np.asarray(probabilities))
        del probabilities
    return {'model':model.path.name,'geometry':asdict(geometry),'chunks':chunks,'segments':spans,
            'speakers':sorted({s['speaker'] for s in spans}),
            'state':{'max_cache_frames':state.max_cache,'max_fifo_frames':state.max_fifo,
                     'compressions':state.compressions},
            'seconds':{'frontend':frontend,'inference':inference,'total':time.perf_counter()-started}}
