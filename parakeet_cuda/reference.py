# SPDX-License-Identifier: Apache-2.0
"""Independent NumPy references for the fixed numerical contract."""
import numpy as np


def norm(x,w,b):
    x=x.astype(np.float64);mean=x.mean(-1,keepdims=True)
    return (((x-mean)/np.sqrt(((x-mean)**2).mean(-1,keepdims=True)+1e-5))*w+b).astype(np.float32)


def stem(features,model):
    x=features[:,:,None].astype(np.float32)
    for index,depthwise in ((0,False),(2,True),(5,True)):
        name=f'encoder.pre_encode.conv.{index}'
        w=model.array(name+'.weight').astype(np.float32).reshape(256,-1,3,3)
        b=model.array(name+'.bias').reshape(256)
        nt,nf=(len(x)+1)//2,(x.shape[1]+1)//2
        padded=np.pad(x,((1,1),(1,1),(0,0)))
        y=np.broadcast_to(b,(nt,nf,256)).copy()
        for kt in range(3):
            for kf in range(3):
                patch=padded[kt:kt+nt*2:2,kf:kf+nf*2:2]
                if depthwise:y+=patch*w[:,0,kt,kf]
                else:y+=patch*w[:,0,kt,kf]
        if depthwise:
            pw=f'encoder.pre_encode.conv.{index+1}'
            y=y@model.array(pw+'.weight').astype(np.float32).reshape(256,256).T+model.array(pw+'.bias').reshape(256)
        x=np.maximum(y,0)
    flat=x.transpose(0,2,1).reshape(len(x),4096)
    return flat@model.array('encoder.pre_encode.out.weight').T+model.array('encoder.pre_encode.out.bias')


def block(x,model,layer=0):
    base=f'encoder.layers.{layer}';t=len(x)
    w={n[len(base)+1:]:model.array(n).astype(np.float32) for n in model.names(base+'.')}
    def ln(x,name):return norm(x,w[name+'.weight'].reshape(-1),w[name+'.bias'].reshape(-1))
    def linear(x,name):return x@w[name+'.weight'].reshape(w[name+'.weight'].shape[0],-1).T
    def ff(x,n):
        z=linear(ln(x,f'norm_feed_forward{n}'),f'feed_forward{n}.linear1')
        z=z/(1+np.exp(-np.clip(z,-80,80)))
        return x+.5*linear(z,f'feed_forward{n}.linear2')
    x=ff(x,1)
    n=ln(x,'norm_self_att')
    q,k,v=[linear(n,'self_attn.linear_'+c).reshape(t,8,128).transpose(1,0,2) for c in 'qkv']
    limit=model.metadata['asr.encoder.pos_emb_max_len']
    pos=linear(model.array('encoder.pos_enc.pe')[limit-t:limit+t-1],'self_attn.linear_pos').reshape(2*t-1,8,128).transpose(1,0,2)
    u=w['self_attn.pos_bias_u'].reshape(8,128);vb=w['self_attn.pos_bias_v'].reshape(8,128)
    ac=(q+u[:,None,:])@k.transpose(0,2,1)
    bd=(q+vb[:,None,:])@pos.transpose(0,2,1)
    # Reproduce the upstream pad/reshape/drop/reshape relative shift independently.
    shifted=np.pad(bd,((0,0),(0,0),(1,0))).reshape(8,2*t,t)[:,1:,:].reshape(8,t,2*t-1)[:,:,:t]
    scores=(ac+shifted)/np.sqrt(128)
    exp=np.exp(scores-scores.max(-1,keepdims=True));prob=exp/exp.sum(-1,keepdims=True)
    context=(prob@v).transpose(1,0,2).reshape(t,1024)
    x=x+linear(context,'self_attn.linear_out')
    pw=linear(ln(x,'norm_conv'),'conv.pointwise_conv1')
    a,b=np.split(pw,2,axis=-1);gate=a/(1+np.exp(-np.clip(b,-80,80)))
    padded=np.pad(gate,((4,4),(0,0)));dw=np.zeros_like(gate)
    kernel=w['conv.depthwise_conv.weight'].reshape(1024,9)
    for k in range(9):dw+=padded[k:k+t]*kernel[:,k]
    bn='conv.batch_norm.'
    dw=(dw-w[bn+'running_mean'].reshape(-1))/np.sqrt(w[bn+'running_var'].reshape(-1)+1e-5)*w[bn+'weight'].reshape(-1)+w[bn+'bias'].reshape(-1)
    dw=dw/(1+np.exp(-np.clip(dw,-80,80)))
    x=x+linear(dw,'conv.pointwise_conv2')
    x=ff(x,2)
    return ln(x,'norm_out')
