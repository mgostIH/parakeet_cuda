# SPDX-License-Identifier: Apache-2.0
"""Layer streaming and two activation schedules sharing the same CUDA math."""
import sys
import time
import numpy as np
from .gpu import Buffer


class Encoder:
    def __init__(self, model, gpu, tile=256, stem_tile=64, query_tile=64, key_tile=256):
        self.model,self.g=model,gpu
        self.tile,self.stem_tile,self.qt,self.kt=tile,stem_tile,query_tile,key_tile
        self.profile=[]

    def linear(self,x,name,rows,k,n,out=None):
        out=self.g.scratch.alloc((rows,n)) if out is None else out
        self.g.blas.linear(x,self.g.weight(name+'.weight'),out,rows,k,n)
        bias=name+'.bias'
        if bias in self.g.weight_views:
            self.g.element('bias_act',[out,self.g.weight(bias),n,rows*n,0],rows*n)
        return out

    def norm(self,x,name,rows,out=None):
        out=self.g.scratch.alloc((rows,1024)) if out is None else out
        self.g.launch('layer_norm',[x,self.g.weight(name+'.weight'),self.g.weight(name+'.bias'),out,1024],rows)
        return out

    def ff(self,x,base,which,rows):
        mark=self.g.scratch.offset
        norm=self.norm(x,base+'.norm_feed_forward'+str(which),rows)
        ff=base+'.feed_forward'+str(which)
        up=self.linear(norm,ff+'.linear1',rows,1024,4096)
        self.g.element('silu',[up,rows*4096],rows*4096)
        down=self.linear(up,ff+'.linear2',rows,4096,1024)
        self.g.element('residual',[x,down,.5,rows*1024],rows*1024)
        self.g.scratch.reset(mark)

    def stem(self,features):
        g=self.g;m=len(features);t1=(m+1)//2;t2=(t1+1)//2;t3=(t2+1)//2
        g.load_weights(self.model,self.model.names('encoder.pre_encode.'))
        g.scratch.reset()
        mel=g.scratch.alloc(features.shape);g.upload(mel,features)
        output=g.scratch.alloc((t3,1024));mark=g.scratch.offset
        for s in range(0,t3,self.stem_tile):
            e=min(t3,s+self.stem_tile)
            b2=max(0,2*s-1);e2=min(t2,2*(e-1)+2)
            b1=max(0,2*b2-1);e1=min(t1,2*(e2-1)+2)
            a=g.scratch.alloc((e1-b1,64,256))
            name='encoder.pre_encode.conv.'
            g.element('conv_first',[mel,g.weight(name+'0.weight'),g.weight(name+'0.bias'),a,m,128,b1,e1-b1,64,256],(e1-b1)*64*256)
            b=g.scratch.alloc((e2-b2,32,256))
            g.element('conv_dw2',[a,g.weight(name+'2.weight'),g.weight(name+'2.bias'),b,e1-b1,64,b1,b2,e2-b2,32,256],(e2-b2)*32*256)
            c=g.scratch.alloc((e2-b2,32,256))
            g.blas.linear(b,g.weight(name+'3.weight'),c,(e2-b2)*32,256,256)
            g.element('bias_act',[c,g.weight(name+'3.bias'),256,(e2-b2)*32*256,1],(e2-b2)*32*256)
            d=g.scratch.alloc((e-s,16,256))
            g.element('conv_dw2',[c,g.weight(name+'5.weight'),g.weight(name+'5.bias'),d,e2-b2,32,b2,s,e-s,16,256],(e-s)*16*256)
            z=g.scratch.alloc((e-s,16,256))
            g.blas.linear(d,g.weight(name+'6.weight'),z,(e-s)*16,256,256)
            g.element('bias_act',[z,g.weight(name+'6.bias'),256,(e-s)*16*256,1],(e-s)*16*256)
            flat=g.scratch.alloc((e-s,4096))
            g.element('flatten_stem',[z,flat,e-s,16,256],(e-s)*4096)
            target=output.view(s*1024*4,(e-s)*1024*4,(e-s,1024))
            self.linear(flat,'encoder.pre_encode.out',e-s,4096,1024,target)
            g.scratch.reset(mark)
        # Copy to the front of the arena for a compact persistent activation.
        host=g.download(output);g.scratch.reset()
        return host

    def positions(self,t):
        limit=self.model.metadata['asr.encoder.pos_emb_max_len']
        if t>limit:raise ValueError(f'{t} encoder frames exceeds positional limit {limit}; use segmented mode')
        return self.model.array('encoder.pos_enc.pe')[limit-t:limit+t-1]

    def qkv(self,x,base,t):
        g=self.g
        q=g.weight(base+'.self_attn.linear_q.weight')
        k=g.weight(base+'.self_attn.linear_k.weight')
        v=g.weight(base+'.self_attn.linear_v.weight')
        if (k.pointer-q.pointer,v.pointer-q.pointer)!=(4*1024**2,8*1024**2):
            raise ValueError('QKV weight packing mismatch')
        packed=Buffer(q.pointer,12*1024**2,(3072,1024))
        out=g.scratch.alloc((t,3072));g.blas.linear(x,packed,out,t,1024,3072)
        return out

    def load_layer(self,layer):
        base=f'encoder.layers.{layer}'
        names=self.model.names(base+'.')
        qkv=[base+'.self_attn.linear_'+c+'.weight' for c in ('q','k','v')]
        self.g.load_weights(self.model,qkv+[n for n in names if n not in qkv])
        return base

    def attention_resident(self,x,base,t):
        g=self.g;mark=g.scratch.offset
        norm=self.norm(x,base+'.norm_self_att',t)
        qkv=self.qkv(norm,base,t)
        q,k,v=[g.scratch.alloc((8,t,128)) for _ in range(3)]
        g.element('split_qkv',[qkv,q,k,v,t],t*1024)
        qu,qv=[g.scratch.alloc((8,t,128)) for _ in range(2)]
        g.element('query_bias',[q,g.weight(base+'.self_attn.pos_bias_u'),g.weight(base+'.self_attn.pos_bias_v'),qu,qv,t],t*1024)
        pe=self.positions(t);p=len(pe)
        pin=g.scratch.alloc(pe.shape);g.upload(pin,pe)
        proj=self.linear(pin,base+'.self_attn.linear_pos',p,1024,1024)
        ph=g.scratch.alloc((8,p,128));g.element('split_heads',[proj,ph,p],p*1024)
        scores=g.scratch.alloc((8,t,t));rel=g.scratch.alloc((8,t,p))
        g.blas.scores(qu,k,scores,t,t);g.blas.scores(qv,ph,rel,t,p)
        g.element('rel_scores',[scores,rel,t,t,p],8*t*t)
        g.launch('softmax',[scores,t],8*t)
        heads=g.scratch.alloc((8,t,128));g.blas.values(scores,v,heads,t,t)
        merged=g.scratch.alloc((t,1024));g.element('merge_heads',[heads,merged,t],t*1024)
        projected=self.linear(merged,base+'.self_attn.linear_out',t,1024,1024)
        g.element('residual',[x,projected,1.,t*1024],t*1024)
        g.scratch.reset(mark)

    def conv(self,x,base,t):
        g=self.g;mark=g.scratch.offset
        z=self.norm(x,base+'.norm_conv',t)
        pw=self.linear(z,base+'.conv.pointwise_conv1',t,1024,2048)
        gate=g.scratch.alloc((t,1024));g.element('glu',[pw,gate,1024,t*1024],t*1024)
        dw=g.scratch.alloc((t,1024))
        bn=base+'.conv.batch_norm'
        g.element('depthwise1',[gate,g.weight(base+'.conv.depthwise_conv.weight'),
            g.weight(bn+'.weight'),g.weight(bn+'.bias'),g.weight(bn+'.running_mean'),
            g.weight(bn+'.running_var'),dw,t,1024,0,t],t*1024)
        out=self.linear(dw,base+'.conv.pointwise_conv2',t,1024,1024)
        g.element('residual',[x,out,1.,t*1024],t*1024)
        g.scratch.reset(mark)

    def block_resident(self,x,base,t):
        self.ff(x,base,1,t);self.attention_resident(x,base,t);self.conv(x,base,t);self.ff(x,base,2,t)
        mark=self.g.scratch.offset
        norm=self.norm(x,base+'.norm_out',t)
        self.g.driver.call('cuMemcpyDtoD_v2',x.pointer,norm.pointer,x.nbytes)
        self.g.scratch.reset(mark)

    def block_spill(self,host,base):
        g=self.g;t=len(host)
        residual=np.empty_like(host);qkv=np.empty((t,3072),dtype=np.float32)
        for s in range(0,t,self.tile):
            e=min(t,s+self.tile);n=e-s;g.scratch.reset()
            x=g.scratch.alloc((n,1024));g.upload(x,host[s:e]);self.ff(x,base,1,n)
            residual[s:e]=g.download(x)
            norm=self.norm(x,base+'.norm_self_att',n)
            qkv[s:e]=g.download(self.qkv(norm,base,n))
        pe=self.positions(t);positions=np.empty_like(pe)
        for s in range(0,len(pe),self.tile):
            e=min(len(pe),s+self.tile);g.scratch.reset()
            x=g.scratch.alloc((e-s,1024));g.upload(x,pe[s:e])
            positions[s:e]=g.download(self.linear(x,base+'.self_attn.linear_pos',e-s,1024,1024))
        attended=np.empty_like(host)
        for s in range(0,t,self.qt):
            e=min(t,s+self.qt);n=e-s;g.scratch.reset()
            q=g.scratch.alloc((8,n,128));g.upload(q,np.ascontiguousarray(qkv[s:e,:1024].reshape(n,8,128).transpose(1,0,2)))
            qu,qv=[g.scratch.alloc((8,n,128)) for _ in range(2)]
            g.element('query_bias',[q,g.weight(base+'.self_attn.pos_bias_u'),g.weight(base+'.self_attn.pos_bias_v'),qu,qv,n],n*1024)
            acc=g.scratch.alloc((8,n,128));state=g.scratch.alloc((8*n,2));alpha=g.scratch.alloc(8*n)
            mark=g.scratch.offset
            for ks in range(0,t,self.kt):
                ke=min(t,ks+self.kt);kn=ke-ks;p=n+kn-1
                k,v=[g.scratch.alloc((8,kn,128)) for _ in range(2)]
                for dst,offset in ((k,1024),(v,2048)):
                    data=qkv[ks:ke,offset:offset+1024].reshape(kn,8,128).transpose(1,0,2)
                    g.upload(dst,np.ascontiguousarray(data))
                start=t-1+ks-s-(n-1)
                ph=g.scratch.alloc((8,p,128))
                g.upload(ph,np.ascontiguousarray(positions[start:start+p].reshape(p,8,128).transpose(1,0,2)))
                scores=g.scratch.alloc((8,n,kn));rel=g.scratch.alloc((8,n,p))
                g.blas.scores(qu,k,scores,n,kn);g.blas.scores(qv,ph,rel,n,p)
                g.element('rel_scores',[scores,rel,n,kn,p],8*n*kn)
                g.launch('online_probs',[scores,state,alpha,kn,n,int(ks==0)],8*n)
                part=g.scratch.alloc((8,n,128));g.blas.values(scores,v,part,n,kn)
                g.element('online_accum',[acc,part,alpha,n*1024,int(ks==0)],n*1024)
                g.scratch.reset(mark)
            g.element('online_finish',[acc,state,n*1024],n*1024)
            merged=g.scratch.alloc((n,1024));g.element('merge_heads',[acc,merged,n],n*1024)
            out=self.linear(merged,base+'.self_attn.linear_out',n,1024,1024)
            attended[s:e]=residual[s:e]+g.download(out)
        output=np.empty_like(host)
        for s in range(0,t,self.tile):
            e=min(t,s+self.tile);lo=max(0,s-4);hi=min(t,e+4);n=e-s;g.scratch.reset()
            halo=g.scratch.alloc((hi-lo,1024));g.upload(halo,attended[lo:hi])
            norm=self.norm(halo,base+'.norm_conv',hi-lo)
            pw=self.linear(norm,base+'.conv.pointwise_conv1',hi-lo,1024,2048)
            gate=g.scratch.alloc((hi-lo,1024));g.element('glu',[pw,gate,1024,(hi-lo)*1024],(hi-lo)*1024)
            dw=g.scratch.alloc((n,1024));bn=base+'.conv.batch_norm'
            g.element('depthwise1',[gate,g.weight(base+'.conv.depthwise_conv.weight'),g.weight(bn+'.weight'),
                g.weight(bn+'.bias'),g.weight(bn+'.running_mean'),g.weight(bn+'.running_var'),dw,hi-lo,1024,s-lo,n],n*1024)
            out=self.linear(dw,base+'.conv.pointwise_conv2',n,1024,1024)
            x=g.scratch.alloc((n,1024));g.upload(x,attended[s:e]);g.element('residual',[x,out,1.,n*1024],n*1024)
            self.ff(x,base,2,n)
            output[s:e]=g.download(self.norm(x,base+'.norm_out',n))
        return output

    def run(self,features,spill=False,layers=24,progress=False):
        g=self.g
        if (len(features)+7)//8>self.model.metadata['asr.encoder.pos_emb_max_len']:
            raise ValueError('Input exceeds the full-context positional limit; choose segmented mode')
        start=time.perf_counter();host=self.stem(features);g.driver.synchronize()
        self.profile=[{'stage':'stem','seconds':time.perf_counter()-start}]
        t=len(host);spill=spill or t>512
        if not spill:
            g.scratch.reset();x=g.scratch.alloc(host.shape);g.upload(x,host);mark=g.scratch.offset
        for i in range(layers):
            start=time.perf_counter();base=self.load_layer(i)
            if spill:host=self.block_spill(host,base)
            else:g.scratch.reset(mark);self.block_resident(x,base,t)
            g.driver.synchronize();elapsed=time.perf_counter()-start
            self.profile.append({'stage':f'layer_{i}','seconds':elapsed})
            if progress:print(f'encoder layer {i+1}/{layers}: {elapsed:.3f}s',file=sys.stderr)
        if not spill:host=g.download(x)
        g.scratch.reset()
        return host

    def project(self,host):
        g=self.g;g.load_weights(self.model,self.model.names('joint.enc.'))
        out=np.empty((len(host),640),dtype=np.float32)
        for s in range(0,len(host),self.tile):
            e=min(len(host),s+self.tile);g.scratch.reset()
            x=g.scratch.alloc((e-s,1024));g.upload(x,host[s:e])
            out[s:e]=g.download(self.linear(x,'joint.enc',e-s,1024,640))
        g.scratch.reset();return out
