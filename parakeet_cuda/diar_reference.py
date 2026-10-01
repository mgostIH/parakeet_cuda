# SPDX-License-Identifier: Apache-2.0
"""Independent NumPy/float64 goldens for the fixed Nemotron CUDA contract."""
import math
import numpy as np


def norm(x,w,b):
    y=x.astype(np.float64);mean=y.mean(axis=-1,keepdims=True)
    return ((y-mean)/np.sqrt(((y-mean)**2).mean(axis=-1,keepdims=True)+1e-5)*w+b).astype(np.float32)


def linear(x,model,name):
    y=x.astype(np.float64)@model.array(name+'.weight').astype(np.float64).T
    if name+'.bias' in model.tensors:y+=model.array(name+'.bias')
    return y.astype(np.float32)


def rotary(x):
    # Time-major heads; explicit split-half rotation, separate from kernel indexing.
    y=x.astype(np.float64).reshape(len(x),8,64)
    theta=np.arange(len(x))[:,None,None]*10000.**(-np.arange(32)[None,None,:]/32)
    a,b=y[:,:,:32].copy(),y[:,:,32:].copy()
    y[:,:,:32]=a*np.cos(theta)-b*np.sin(theta)
    y[:,:,32:]=a*np.sin(theta)+b*np.cos(theta)
    return y.transpose(1,0,2).astype(np.float32)


def block(x,model,index=0):
    b=f'encoder.layers.{index}'
    def ln(a,name):return norm(a,model.array(b+'.'+name+'.weight'),model.array(b+'.'+name+'.bias'))
    q,k,v=np.split(linear(ln(x,'norm1'),model,b+'.attn.w_qkv'),3,axis=-1)
    q,k=rotary(q),rotary(k)
    v=v.reshape(len(x),8,64).transpose(1,0,2)
    scores=np.matmul(q.astype(np.float64),k.astype(np.float64).transpose(0,2,1))/8
    scores-=scores.max(axis=-1,keepdims=True);prob=np.exp(scores);prob/=prob.sum(axis=-1,keepdims=True)
    attended=np.matmul(prob,v.astype(np.float64)).transpose(1,0,2).reshape(len(x),512).astype(np.float32)
    residual=x+linear(attended,model,b+'.attn.out_proj')
    ff=linear(ln(residual,'norm2'),model,b+'.ffn.net.0')
    erf=np.fromiter((math.erf(float(v)/math.sqrt(2)) for v in ff.flat),dtype=np.float64,count=ff.size).reshape(ff.shape)
    gelu=(ff.astype(np.float64)*.5*(1+erf)).astype(np.float32)
    return residual+linear(gelu,model,b+'.ffn.net.3')


def chunk(mel,model,layers=31):
    stacked=np.zeros(((len(mel)+7)//8,8,128),np.float32)
    stacked.reshape(-1,128)[:len(mel)]=mel
    embeddings=linear(stacked.reshape(-1,1024),model,'encoder.pre_encode.proj')
    x=norm(embeddings,model.array('encoder.embed_norm.weight'),model.array('encoder.embed_norm.bias'))
    for i in range(layers):x=block(x,model,i)
    if layers!=31:return x,embeddings
    x=norm(x,model.array('encoder.final_norm.weight'),model.array('encoder.final_norm.bias'))
    x=linear(x,model,'encoder_proj')
    # Direct convolution reference, deliberately independent of im2col.
    weight=model.array('subpixel_upsample.weight').astype(np.float64)
    conv=np.tile(model.array('subpixel_upsample.bias'),(len(x),1)).astype(np.float64)
    for time in range(len(x)):
        for dt in range(3):
            other=time+dt-1
            if 0<=other<len(x):conv[time]+=weight[:,:,dt]@x[other].astype(np.float64)
    high=np.maximum(conv,0).reshape(-1,192).astype(np.float32)
    hidden=np.maximum(linear(high,model,'head.first_hidden_to_hidden'),0)
    logits=linear(hidden,model,'head.single_hidden_to_spks')
    return (1/(1+np.exp(-logits.astype(np.float64)))).astype(np.float32),embeddings
