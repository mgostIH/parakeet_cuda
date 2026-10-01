"""Fixed Q8 x FP32 contract: real projection shape plus tile tails."""
from pathlib import Path
import numpy as np
from txir_tools.compiler import CUDAKernel
from txir_tools.workload import Case


def build_kernel():
    return CUDAKernel(Path(__file__).resolve().parents[1]/'kernels/q8_gemm.cu',arch='sm_50')


def cases():
    rng=np.random.default_rng(903)
    for m,k,n in ((1,32,3),(17,640,65),(33,1024,129),(441,1024,4096)):
        scales=rng.uniform(.0001,.02,(n*k//32,1)).astype(np.float16)
        values=rng.integers(-127,128,(n*k//32,32),dtype=np.int8)
        packed=np.empty((len(scales),34),np.uint8)
        packed[:,:2]=scales.view(np.uint8).reshape(-1,2);packed[:,2:]=values.view(np.uint8)
        weight=(scales.astype(np.float64)*values.astype(np.float64)).reshape(n,k)
        x=rng.normal(size=(m,k)).astype(np.float32)
        golden=(x.astype(np.float64)@weight.T).astype(np.float32)
        out=np.empty((m,n),np.float32)
        yield Case(f'm{m}_k{k}_n{n}',[packed,x,out,np.int32(m),np.int32(k),np.int32(n)],
                   {2:golden},grid=((m+31)//32)*((n+63)//64),block=256,atol=8e-4,rtol=3e-4)
