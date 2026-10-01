# SPDX-License-Identifier: Apache-2.0
"""Project-owned arena/launch/cuBLAS adapter over the shared Driver API."""
import ctypes as C
import ctypes.util
import os
from pathlib import Path
import resource
import time
import numpy as np
from .cuda_driver import Driver,gpu_lock
from .compiler import compile_runtime
from .paths import ROOT

MIB = 1024**2


class Buffer:
    def __init__(self, pointer, nbytes, shape=None):
        self.pointer, self.nbytes, self.shape = int(pointer), int(nbytes), shape

    def view(self, offset, nbytes, shape=None):
        if offset<0 or nbytes<0 or offset+nbytes>self.nbytes:
            raise ValueError('Device view exceeds buffer')
        return Buffer(self.pointer+offset, nbytes, shape)


class Arena:
    def __init__(self, gpu, size, name):
        self.gpu, self.size, self.name = gpu, size, name
        self.base = C.c_uint64()
        gpu.driver.call('cuMemAlloc_v2', C.byref(self.base), size)
        self.offset = self.peak = 0

    def alloc(self, shape, dtype=np.float32):
        shape = (shape,) if isinstance(shape,int) else tuple(shape)
        size = int(np.prod(shape))*np.dtype(dtype).itemsize
        offset = (self.offset+255)//256*256
        if offset+size>self.size:
            raise MemoryError(f'{self.name}: {offset+size} bytes exceeds fixed {self.size}-byte arena')
        self.offset=offset+size;self.peak=max(self.offset,self.peak)
        return Buffer(self.base.value+offset,size,shape)

    def reset(self, mark=0):
        # One ordered execution stream: previous readers finish before reuse.
        self.offset=mark

    def close(self):
        if self.base.value:
            self.gpu.driver.call('cuMemFree_v2',self.base);self.base.value=0


class Blas:
    def __init__(self):
        explicit=os.environ.get('PARAKEET_CUBLAS')
        candidates=[explicit] if explicit else ['libcublas.so.12',ctypes.util.find_library('cublas'),'libcublas.so.13']
        if not explicit:
            for base in sorted(Path('/usr/local').glob('cuda*')):
                candidates.extend(str(p) for folder in ('lib64','targets/x86_64-linux/lib')
                                  for p in (base/folder).glob('libcublas.so.*') if 'Lt' not in p.name)
        errors=[]
        for name in candidates:
            if not name:continue
            try:self.lib=C.CDLL(name);break
            except OSError as error:errors.append(str(error))
        else:raise RuntimeError('cuBLAS was not found. Install the CUDA toolkit or set PARAKEET_CUBLAS to libcublas.so.12. '+ '; '.join(errors))
        signatures={
            'cublasCreate_v2':[C.POINTER(C.c_void_p)],
            'cublasDestroy_v2':[C.c_void_p],
            'cublasSetStream_v2':[C.c_void_p,C.c_void_p],
            'cublasSetWorkspace_v2':[C.c_void_p,C.c_void_p,C.c_size_t],
            'cublasSgemm_v2':[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,
                C.POINTER(C.c_float),C.c_void_p,C.c_int,C.c_void_p,C.c_int,
                C.POINTER(C.c_float),C.c_void_p,C.c_int],
            'cublasSgemmStridedBatched':[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,
                C.POINTER(C.c_float),C.c_void_p,C.c_int,C.c_longlong,
                C.c_void_p,C.c_int,C.c_longlong,C.POINTER(C.c_float),C.c_void_p,C.c_int,C.c_longlong,C.c_int],
        }
        for name,args in signatures.items():
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=C.c_int
        self.handle=C.c_void_p();self.call('cublasCreate_v2',C.byref(self.handle))
        self.one,self.zero=C.c_float(1),C.c_float(0)

    def call(self,name,*args):
        status=getattr(self.lib,name)(*args)
        if status:raise RuntimeError(f'{name}: cuBLAS status {status}')

    def linear(self,x,w,out,rows,k,n):
        # Row-major X @ W.T becomes column-major W @ X.T.
        self.call('cublasSgemm_v2',self.handle,1,0,n,rows,k,C.byref(self.one),
                  C.c_void_p(w.pointer),k,C.c_void_p(x.pointer),k,C.byref(self.zero),C.c_void_p(out.pointer),n)

    def scores(self,q,k,out,qt,kt,heads=8,width=128):
        self.call('cublasSgemmStridedBatched',self.handle,1,0,kt,qt,width,C.byref(self.one),
                  C.c_void_p(k.pointer),width,kt*width,C.c_void_p(q.pointer),width,qt*width,
                  C.byref(self.zero),C.c_void_p(out.pointer),kt,qt*kt,heads)

    def values(self,prob,v,out,qt,kt,heads=8,width=128):
        self.call('cublasSgemmStridedBatched',self.handle,0,0,width,qt,kt,C.byref(self.one),
                  C.c_void_p(v.pointer),width,kt*width,C.c_void_p(prob.pointer),kt,qt*kt,
                  C.byref(self.zero),C.c_void_p(out.pointer),width,qt*width,heads)

    def close(self):
        if self.handle.value:
            self.call('cublasDestroy_v2',self.handle);self.handle.value=None


class GPU:
    def __init__(self,budget_mib=384,scratch_mib=96,compact_cache_mib=0,device=0):
        self.driver=Driver(device);self.arenas=[];self.blas=None;self.module=C.c_void_p()
        try:
            self.device_info=self.driver.info()
            if int(self.device_info['arch'][3:])<50:
                raise ValueError('Compute capability 5.0 or later is required')
            self.driver.lib.cuMemGetInfo_v2.argtypes=[C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)]
            self.driver.lib.cuMemGetInfo_v2.restype=C.c_int
            self.driver.lib.cuMemcpyDtoD_v2.argtypes=[C.c_uint64,C.c_uint64,C.c_size_t]
            self.driver.lib.cuMemcpyDtoD_v2.restype=C.c_int
            self.free_before=self.free()
            self.blas=Blas()
            self.free_after_blas=self.free()
            self.functions={}
            cubin=compile_runtime(self.device_info['arch'])
            self.driver.call('cuModuleLoad',C.byref(self.module),str(cubin).encode())
            sizes={'weights':100*MIB,'raw':28*MIB,'scratch':scratch_mib*MIB,'blas_workspace':16*MIB}
            if compact_cache_mib:
                sizes['compact_cache']=compact_cache_mib*MIB
            self.reserved=sum(sizes.values())
            if self.reserved>budget_mib*MIB:
                raise MemoryError(f'Plan needs {self.reserved/MIB:.0f} MiB; budget is {budget_mib}')
            if self.reserved+192*MIB>self.free():
                raise MemoryError('Insufficient free VRAM for this plan plus 192 MiB desktop reserve')
            for name,size in sizes.items():
                arena=Arena(self,size,name);self.arenas.append(arena);setattr(self,name,arena)
            self.blas.call('cublasSetWorkspace_v2',self.blas.handle,
                           C.c_void_p(self.blas_workspace.base.value),self.blas_workspace.size)
            self.free_after_arenas=self.free()
            self.weight_views={}
            self.cached_model=None
            self.timings={}
        except Exception:
            self.close();raise

    def free(self):
        free,total=C.c_size_t(),C.c_size_t()
        self.driver.call('cuMemGetInfo_v2',C.byref(free),C.byref(total));return free.value

    def launch(self,name,args,grid,block=256):
        if name not in self.functions:
            f=C.c_void_p();self.driver.call('cuModuleGetFunction',C.byref(f),self.module,name.encode());self.functions[name]=f
        values=[C.c_uint64(a.pointer) if isinstance(a,Buffer) else
                C.c_float(a) if isinstance(a,float) else C.c_int(a) for a in args]
        ptrs=(C.c_void_p*len(values))(*(C.addressof(x) for x in values))
        self.driver.call('cuLaunchKernel',self.functions[name],grid,1,1,block,1,1,0,None,ptrs,None)

    def element(self,name,args,n):
        self.launch(name,args,(n+255)//256)

    def upload(self,buffer,array):
        array=np.ascontiguousarray(array)
        if array.nbytes>buffer.nbytes:raise ValueError('Upload exceeds buffer')
        self.driver.call('cuMemcpyHtoD_v2',buffer.pointer,C.c_void_p(array.ctypes.data),array.nbytes)

    def download(self,buffer,shape=None,dtype=np.float32):
        out=np.empty(buffer.shape if shape is None else shape,dtype=dtype)
        if out.nbytes>buffer.nbytes:raise ValueError('Download exceeds buffer')
        self.driver.call('cuMemcpyDtoH_v2',C.c_void_p(out.ctypes.data),buffer.pointer,out.nbytes)
        return out

    def load_weights(self,model,names):
        self.weights.reset();self.raw.reset();self.weight_views={}
        for name in names:
            t=model.tensors[name]
            out=self.weights.alloc(t.shape)
            if t.kind==0:
                if model is self.cached_model:
                    self.driver.call('cuMemcpyDtoD_v2',out.pointer,self.cached_raw(name).pointer,t.nbytes)
                else:self.upload(out,model.array(name))
            else:
                if model is self.cached_model:raw=self.cached_raw(name)
                else:
                    raw=self.raw.alloc(t.nbytes,np.uint8);self.upload(raw,model.raw(name))
                self.element('dequant_q8' if t.kind==8 else 'expand_half',[raw,out,t.elements],t.elements)
            self.weight_views[name]=out
        return self.weight_views

    def cache_model(self,model):
        if not hasattr(self,'compact_cache'):
            raise ValueError('No compact model cache was reserved')
        self.compact_cache.reset()
        self.cached_bytes=self.compact_cache.alloc(model.mapping.size,np.uint8)
        self.upload(self.cached_bytes,model.mapping)
        self.cached_model=model

    def cached_raw(self,name):
        t=self.cached_model.tensors[name]
        return self.cached_bytes.view(self.cached_model.data_start+t.offset,t.nbytes)

    def weight(self,name):
        return self.weight_views[name]

    def stats(self):
        return {'device':self.device_info,'reserved_mib':self.reserved/MIB,'free_before_mib':self.free_before/MIB,
                'free_after_blas_mib':self.free_after_blas/MIB,'free_after_arenas_mib':self.free_after_arenas/MIB,
                'free_at_finish_mib':self.free()/MIB,'physical_allocation_count':len(self.arenas),
                'host_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                'arena_high_water_mib':{a.name:a.peak/MIB for a in self.arenas}}

    def close(self):
        if hasattr(self,'driver'):
            try:self.driver.synchronize()
            finally:
                if self.blas:self.blas.close()
                for a in reversed(self.arenas):a.close()
                if self.module.value:self.driver.call('cuModuleUnload',self.module);self.module.value=None
                self.driver.__exit__()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
