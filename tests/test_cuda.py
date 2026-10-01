import json
from pathlib import Path
import struct
import os
import unittest
import numpy as np
from parakeet_cuda.gguf import Model
from parakeet_cuda.gpu import GPU,ROOT,gpu_lock
from parakeet_cuda.encoder import Encoder
from parakeet_cuda import reference


class CUDAContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('PARAKEET_RUN_GPU_TESTS')!='1':
            raise unittest.SkipTest('Set PARAKEET_RUN_GPU_TESTS=1 to run CUDA/model contracts')
        if not (ROOT/'models/parakeet-tdt-0.6b-v3.q8_0.gguf').is_file():
            raise unittest.SkipTest('Download Parakeet weights to run CUDA/model contracts')
        cls.lock=gpu_lock();cls.lock.__enter__()
        cls.g=GPU();cls.model=Model(ROOT/'models/parakeet-tdt-0.6b-v3.q8_0.gguf')
        cls.errors={}

    @classmethod
    def tearDownClass(cls):
        (ROOT/'artifacts/cuda_contract.json').write_text(json.dumps({'errors':cls.errors,'memory':cls.g.stats()},indent=2)+'\n')
        cls.g.close();cls.lock.__exit__(None,None,None)

    def check(self,name,actual,expected,atol=2e-4,rtol=2e-4):
        self.errors[name]={'max_abs':float(np.max(np.abs(actual-expected))),'shape':list(actual.shape)}
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual,expected,atol=atol,rtol=rtol)

    def test_q8_decode_and_gemv(self):
        g=self.g;rng=np.random.default_rng(17)
        for n,k in ((3,32),(7,640),(5,1024)):
            scales=rng.uniform(.001,.1,(n*k//32)).astype(np.float16)
            values=rng.integers(-128,128,(n*k//32,32),dtype=np.int8)
            raw=np.empty((len(scales),34),dtype=np.uint8)
            raw[:,:2]=scales.view(np.uint8).reshape(-1,2);raw[:,2:]=values.view(np.uint8)
            golden=np.empty(n*k,np.float32)
            # Scalar byte interpretation is independent of the production decoder.
            data=raw.tobytes()
            for i in range(n*k):golden[i]=struct.unpack_from('<e',data,(i//32)*34)[0]*struct.unpack_from('b',data,(i//32)*34+2+i%32)[0]
            g.scratch.reset();buf=g.scratch.alloc(raw.size,np.uint8);g.upload(buf,raw)
            out=g.scratch.alloc(n*k);g.element('dequant_q8',[buf,out,n*k],n*k)
            self.check(f'q8_{n}_{k}',g.download(out),golden,0,0)
            x=rng.normal(size=k).astype(np.float32);xb=g.scratch.alloc(k);g.upload(xb,x)
            y=g.scratch.alloc(n);g.launch('q8_gemv',[buf,xb,y,k,n],n)
            self.check(f'gemv_{n}_{k}',g.download(y),(golden.reshape(n,k).astype(np.float64)@x).astype(np.float32),2e-4,2e-4)

    def test_layer_norm(self):
        g=self.g;rng=np.random.default_rng(4)
        for rows,width in ((1,640),(3,1024)):
            x=rng.normal(size=(rows,width)).astype(np.float32);w=rng.normal(size=width).astype(np.float32);b=rng.normal(size=width).astype(np.float32)
            g.scratch.reset();bufs=[g.scratch.alloc(a.shape) for a in (x,w,b,x)]
            for dst,a in zip(bufs,(x,w,b)):g.upload(dst,a)
            g.launch('layer_norm',bufs+[width],rows)
            self.check(f'norm_{width}',g.download(bufs[-1]),reference.norm(x,w,b))

    def test_stem_seams(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(193,128)).astype(np.float32)
        expected=reference.stem(x,self.model)
        for tile in (3,11,64):
            actual=Encoder(self.model,self.g,stem_tile=tile).stem(x)
            self.check(f'stem_{tile}',actual,expected,2e-3,3e-4)

    def test_complete_block_and_spill(self):
        rng=np.random.default_rng(5);host=rng.normal(size=(13,1024)).astype(np.float32)
        expected=reference.block(host,self.model)
        engine=Encoder(self.model,self.g,tile=5,query_tile=3,key_tile=4)
        base=engine.load_layer(0);g=self.g;g.scratch.reset()
        x=g.scratch.alloc(host.shape);g.upload(x,host);engine.block_resident(x,base,len(host))
        self.check('block_resident',g.download(x),expected,2e-3,5e-4)
        actual=engine.block_spill(host,base)
        self.check('block_spill',actual,expected,2e-3,5e-4)

    def test_tdt_argmax_ties(self):
        g=self.g;g.scratch.reset();x=np.zeros(8198,np.float32);x[19]=x[37]=2;x[8195]=x[8197]=3
        inp=g.scratch.alloc(8198);out=g.scratch.alloc(2,np.int32);g.upload(inp,x)
        g.launch('argmax_tdt',[inp,out],2)
        np.testing.assert_array_equal(g.download(out,dtype=np.int32),[19,2])


if __name__=='__main__':unittest.main()
