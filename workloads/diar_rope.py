"""Fixed Maxwell packed QKV/NEOX RoPE contract, including maximum geometry."""
from pathlib import Path
import numpy as np
from txir_tools.compiler import CUDAKernel
from txir_tools.workload import Case


def build_kernel():
    return CUDAKernel(Path(__file__).resolve().parents[1]/'parakeet_cuda/kernels/runtime.cu',entry='diar_rope_qkv',arch='sm_50')


def cases():
    rng=np.random.default_rng(1813)
    for t in (1,13,67,684,768):
        x=rng.normal(size=(t,1536)).astype(np.float32)
        goldens=[]
        for c in range(3):
            y=x[:,c*512:(c+1)*512].astype(np.float64).reshape(t,8,64)
            if c<2:
                angles=np.arange(t)[:,None,None]/10000.**(np.arange(32)[None,None,:]/32)
                a,b=y[:,:,:32].copy(),y[:,:,32:].copy()
                y=np.concatenate((a*np.cos(angles)-b*np.sin(angles),a*np.sin(angles)+b*np.cos(angles)),axis=2)
            goldens.append(y.transpose(1,0,2).astype(np.float32))
        outputs=[np.empty((8,t,64),np.float32) for _ in range(3)]
        yield Case(f'frames_{t}',[x,*outputs,np.int32(t)],{i+1:y for i,y in enumerate(goldens)},
                   grid=(t*512+255)//256,block=256,atol=1e-4,rtol=4e-5)
