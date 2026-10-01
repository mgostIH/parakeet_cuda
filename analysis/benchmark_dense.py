"""Paired CUDA graph timings: direct Q8 GEMM vs expansion plus cuBLAS."""
import ctypes as C
import json
from pathlib import Path
import numpy as np
from parakeet_cuda.gpu import GPU,ROOT,gpu_lock,Buffer
from txir_tools.compiler import CUDAKernel,compile_kernel
from workloads.q8_gemm import cases


def main():
    output=ROOT/'artifacts/dense-comparison';output.mkdir(parents=True,exist_ok=False)
    compiled=compile_kernel(CUDAKernel(ROOT/'kernels/q8_gemm.cu',arch='sm_50'),output/'compile')
    results=[]
    with gpu_lock(),GPU() as g:
        d=g.driver;module=C.c_void_p();f=C.c_void_p()
        d.call('cuModuleLoad',C.byref(module),str(compiled.cubin).encode())
        d.call('cuModuleGetFunction',C.byref(f),module,b'q8_gemm')
        for name,args in {'cuStreamBeginCapture':[C.c_void_p,C.c_int],
                          'cuStreamEndCapture':[C.c_void_p,C.POINTER(C.c_void_p)]}.items():
            fn=getattr(d.lib,name);fn.argtypes=args;fn.restype=C.c_int
        stream=C.c_void_p();d.call('cuStreamCreate',C.byref(stream),1)
        g.blas.call('cublasSetStream_v2',g.blas.handle,stream)
        # Setting the stream resets workspace; restore the bounded allocation.
        g.blas.call('cublasSetWorkspace_v2',g.blas.handle,C.c_void_p(g.blas_workspace.base.value),g.blas_workspace.size)
        try:
            for case in cases():
                raw,x,_,m,k,n=case.args;m,k,n=int(m),int(k),int(n)
                g.scratch.reset();g.weights.reset();g.raw.reset()
                rb=g.raw.alloc(raw.size,np.uint8);xb=g.scratch.alloc(x.shape);yb=g.scratch.alloc((m,n));wb=g.weights.alloc((n,k))
                g.upload(rb,raw);g.upload(xb,x)
                # Warm both, independently compare each result to the fixed golden.
                params=[C.c_uint64(b.pointer) for b in (rb,xb,yb)]+[C.c_int(v) for v in (m,k,n)]
                ptrs=(C.c_void_p*len(params))(*(C.addressof(v) for v in params))
                def direct():d.call('cuLaunchKernel',f,((m+31)//32)*((n+63)//64),1,1,256,1,1,0,stream,ptrs,None)
                # GPU.element uses default stream; capture expansion explicitly on this stream.
                ef=g.functions.get('dequant_q8')
                if ef is None:
                    ef=C.c_void_p();d.call('cuModuleGetFunction',C.byref(ef),g.module,b'dequant_q8')
                ev=[C.c_uint64(rb.pointer),C.c_uint64(wb.pointer),C.c_int(n*k)]
                ep=(C.c_void_p*3)(*(C.addressof(v) for v in ev))
                def control():
                    d.call('cuLaunchKernel',ef,(n*k+255)//256,1,1,256,1,1,0,stream,ep,None)
                    g.blas.linear(xb,wb,yb,m,k,n)
                for fn in (direct,control):
                    fn();d.synchronize();np.testing.assert_allclose(g.download(yb),case.expected[2],atol=case.atol,rtol=case.rtol)
                graphs=[];events=[]
                for fn in (direct,control):
                    graph,exe,start,end=[C.c_void_p() for _ in range(4)]
                    d.call('cuEventCreate',C.byref(start),0);d.call('cuEventCreate',C.byref(end),0)
                    d.call('cuStreamBeginCapture',stream,0)
                    for _ in range(10):fn()
                    d.call('cuStreamEndCapture',stream,C.byref(graph))
                    d.call('cuGraphInstantiateWithFlags',C.byref(exe),graph,0)
                    d.call('cuGraphUpload',exe,stream);d.call('cuStreamSynchronize',stream)
                    graphs.append((graph,exe));events.append((start,end))
                samples=[[],[]]
                for rep in range(10):
                    for index in ((0,1) if rep%2==0 else (1,0)):
                        start,end=events[index];d.call('cuEventRecord',start,stream)
                        d.call('cuGraphLaunch',graphs[index][1],stream);d.call('cuEventRecord',end,stream);d.call('cuEventSynchronize',end)
                        ms=C.c_float();d.call('cuEventElapsedTime',C.byref(ms),start,end)
                        if rep>=2:samples[index].append(ms.value*1000/10)
                record={'case':case.name,'direct_us':samples[0],'expand_sgemm_us':samples[1]}
                results.append(record);print(json.dumps(record),flush=True)
                for graph,exe in graphs:d.call('cuGraphExecDestroy',exe);d.call('cuGraphDestroy',graph)
                for start,end in events:d.call('cuEventDestroy_v2',start);d.call('cuEventDestroy_v2',end)
        finally:
            d.call('cuStreamDestroy_v2',stream);d.call('cuModuleUnload',module)
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
