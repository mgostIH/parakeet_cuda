# SPDX-License-Identifier: Apache-2.0
"""TDT state machine with CPU and CUDA predictor/joint implementations."""
import numpy as np


class CPUDecoder:
    def __init__(self,model):
        self.model=model
        names=model.names('decoder.prediction.dec_rnn.')+model.names('joint.pred.')+model.names('joint.joint_net.')
        self.w={n:np.ascontiguousarray(model.array(n).astype(np.float32)) for n in names}
        self.h=[np.zeros(640,np.float32) for _ in range(2)]
        self.c=[np.zeros(640,np.float32) for _ in range(2)]
        self.candidate=None;self.pred=None

    def predict(self,token):
        x=self.model.array('decoder.prediction.embed.weight')[token].astype(np.float32)
        hn,cn=[],[]
        for l in range(2):
            ih=f'decoder.prediction.dec_rnn.lstm.ih_l{l}'
            hh=f'decoder.prediction.dec_rnn.lstm.hh_l{l}'
            z=self.w[ih+'.weight']@x+self.w[ih+'.bias']+self.w[hh+'.weight']@self.h[l]+self.w[hh+'.bias']
            i,f,c,o=np.split(z,4)
            sig=lambda a: 1/(1+np.exp(-np.clip(a,-80,80)))
            state=sig(f)*self.c[l]+sig(i)*np.tanh(c)
            x=sig(o)*np.tanh(state);hn.append(x);cn.append(state)
        self.candidate=(hn,cn)
        self.pred=self.w['joint.pred.weight']@x+self.w['joint.pred.bias']

    def joint(self,frame):
        z=self.w['joint.joint_net.2.weight']@np.maximum(frame+self.pred,0)+self.w['joint.joint_net.2.bias']
        return int(np.argmax(z[:8193])),int(np.argmax(z[8193:]))

    def commit(self):
        self.h,self.c=self.candidate


class CUDADecoder:
    def __init__(self,model,gpu):
        self.model,self.g=model,gpu
        names=model.names('decoder.prediction.dec_rnn.')+model.names('joint.pred.')+model.names('joint.joint_net.')
        gpu.load_weights(model,names);gpu.scratch.reset()
        self.h,self.c,self.hn,self.cn=[[gpu.scratch.alloc(640) for _ in range(2)] for _ in range(4)]
        for dst in self.h+self.c:gpu.upload(dst,np.zeros(640,np.float32))
        self.embed=gpu.scratch.alloc(640)
        self.ih,self.hh=[gpu.scratch.alloc(2560) for _ in range(2)]
        self.pred=gpu.scratch.alloc(640);self.enc=gpu.scratch.alloc(640);self.act=gpu.scratch.alloc(640)
        self.logits=gpu.scratch.alloc(8198);self.ids=gpu.scratch.alloc(2,np.int32)
        self.predict_calls=self.joint_calls=0

    def linear(self,x,name,out,k,n):
        self.g.blas.linear(x,self.g.weight(name+'.weight'),out,1,k,n)
        self.g.element('bias_act',[out,self.g.weight(name+'.bias'),n,n,0],n)

    def predict(self,token):
        g=self.g;g.upload(self.embed,self.model.array('decoder.prediction.embed.weight')[token].astype(np.float32))
        x=self.embed
        for l in range(2):
            ih=f'decoder.prediction.dec_rnn.lstm.ih_l{l}'
            hh=f'decoder.prediction.dec_rnn.lstm.hh_l{l}'
            g.blas.linear(x,g.weight(ih+'.weight'),self.ih,1,640,2560)
            g.blas.linear(self.h[l],g.weight(hh+'.weight'),self.hh,1,640,2560)
            g.element('lstm',[self.ih,self.hh,g.weight(ih+'.bias'),g.weight(hh+'.bias'),self.c[l],self.hn[l],self.cn[l],640],640)
            x=self.hn[l]
        self.linear(x,'joint.pred',self.pred,640,640);self.predict_calls+=1

    def joint(self,frame):
        g=self.g;g.upload(self.enc,frame)
        g.element('joint_relu',[self.enc,self.pred,self.act,640],640)
        self.linear(self.act,'joint.joint_net.2',self.logits,640,8198)
        g.launch('argmax_tdt',[self.logits,self.ids],2)
        ids=g.download(self.ids,dtype=np.int32);self.joint_calls+=1
        return int(ids[0]),int(ids[1])

    def commit(self):
        self.h,self.hn=self.hn,self.h;self.c,self.cn=self.cn,self.c


def decode(projected,decoder,model):
    ids,times,trace=[],[],[]
    blank=8192;previous=blank;valid=False;t=0;calls=0
    durations=model.metadata['asr.tdt.durations']
    while t<len(projected):
        symbols=0;skip=0
        while symbols<10:
            if not valid:decoder.predict(previous);valid=True
            token,duration=decoder.joint(projected[t]);calls+=1
            skip=durations[duration]
            if token==blank and skip==0:skip=1
            trace.append([t,token,skip])
            if token!=blank:
                ids.append(token);times.append([t*.08,(t+max(1,skip))*.08])
                previous=token;decoder.commit();valid=False
            symbols+=1;t+=skip
            if skip:break
        if symbols==10 and skip==0:t+=1
    return {'text':model.text(ids),'tokens':ids,'token_times':times,
            'decode_trace':trace,'joint_calls':calls}
