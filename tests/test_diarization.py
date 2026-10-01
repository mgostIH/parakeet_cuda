import ctypes as C
import json
import os
from pathlib import Path
import tempfile
import unittest
import wave
import numpy as np
from parakeet_cuda.frontend import PCMSource
from parakeet_cuda.diarization import DiarModel,DiarEncoder,Geometry,SpeakerState,mel_window,tag_words,speaker_segments
from parakeet_cuda import diar_reference as golden
from parakeet_cuda.gpu import GPU,ROOT,gpu_lock


class DiarCUDAContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('PARAKEET_RUN_GPU_TESTS')!='1':
            raise unittest.SkipTest('Set PARAKEET_RUN_GPU_TESTS=1 to run CUDA/model contracts')
        if not (ROOT/'models/Nemotron-3-Diarization.q8_0.gguf').is_file():
            raise unittest.SkipTest('Download Nemotron weights to run CUDA/model contracts')
        cls.lock=gpu_lock();cls.lock.__enter__()
        cls.g=GPU(compact_cache_mib=104)
        cls.model=DiarModel(ROOT/'models/Nemotron-3-Diarization.q8_0.gguf')
        cls.g.cache_model(cls.model);cls.engine=DiarEncoder(cls.model,cls.g);cls.errors={}

    @classmethod
    def tearDownClass(cls):
        (ROOT/'artifacts/diar_contract.json').write_text(json.dumps({'errors':cls.errors,'memory':cls.g.stats()},indent=2)+'\n')
        cls.g.close();cls.lock.__exit__(None,None,None)

    def check(self,name,actual,expected,atol=3e-4,rtol=3e-4):
        self.errors[name]={'max_abs':float(np.max(np.abs(actual-expected))),'shape':list(actual.shape)}
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual,expected,atol=atol,rtol=rtol)

    def test_rotary_split_tails(self):
        g=self.g;rng=np.random.default_rng(184)
        for t in (1,13,67):
            packed=rng.normal(size=(t,1536)).astype(np.float32)
            g.scratch.reset();inp=g.scratch.alloc(packed.shape);g.upload(inp,packed)
            outputs=[g.scratch.alloc((8,t,64)) for _ in range(3)]
            g.element('diar_rope_qkv',[inp,*outputs,t],t*512)
            for j in range(3):
                expected=golden.rotary(packed[:,j*512:(j+1)*512]) if j<2 else packed[:,1024:].reshape(t,8,64).transpose(1,0,2)
                self.check(f'rope_{t}_{j}',g.download(outputs[j]),expected,2e-5,2e-5)

    def test_real_transformer_blocks(self):
        rng=np.random.default_rng(121);g=self.g
        for t,index in ((1,0),(13,15),(67,30)):
            x=rng.normal(size=(t,512)).astype(np.float32)
            expected=golden.block(x,self.model,index)
            base=f'encoder.layers.{index}'
            g.scratch.reset();inp=g.scratch.alloc(x.shape);g.upload(inp,x)
            g.load_weights(self.model,self.model.names(base+'.'));self.engine.block(inp,base,t)
            self.check(f'block_{t}_{index}',g.download(inp),expected,1e-3,5e-4)

    def test_complete_model_and_padded_tail(self):
        rng=np.random.default_rng(166)
        for frames in (1,81):
            mel=rng.normal(-3,1,size=(frames,128)).astype(np.float32)
            expected,emb=golden.chunk(mel,self.model)
            actual,actual_emb=self.engine.run_chunk(mel)
            self.check(f'embeddings_{frames}',actual_emb,emb)
            self.check(f'full_model_{frames}',actual,expected,2e-4,2e-4)


class DiarHostContract(unittest.TestCase):
    @unittest.skipUnless((ROOT/'models/Nemotron-3-Diarization.q8_0.gguf').is_file(),'Optional downloaded diarization weights required')
    def test_mapped_audio_and_frontend_seams(self):
        model=DiarModel(ROOT/'models/Nemotron-3-Diarization.q8_0.gguf')
        rng=np.random.default_rng(910);samples=rng.integers(-12000,12000,16273,dtype=np.int16)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'audio.wav'
            with wave.open(str(path),'wb') as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(samples.tobytes())
            with PCMSource(path) as source:
                np.testing.assert_array_equal(source.read(),samples.astype(np.float32)/32768)
                count=len(source)//160+1
                full=mel_window(source,0,count,model)
                for start,end in ((0,1),(1,5),(15,31),(count-2,count)):
                    np.testing.assert_allclose(mel_window(source,start,end,model),full[start:end],atol=2e-6,rtol=2e-6)

    def test_word_anchors_and_overlap(self):
        probabilities=np.zeros((100,8),np.float32);probabilities[:50,0]=.9;probabilities[50:,1]=.8
        probabilities[20:30,1]=.7
        words=[{'word':'one','start':.1,'end':.15},{'word':'both','start':.2,'end':.3},
               {'word':'two','start':.6,'end':.9},{'word':'edge','start':1.,'end':1.}]
        tag_words(words,probabilities)
        self.assertEqual([w['speaker'] for w in words],['speaker_1','speaker_1','speaker_2','speaker_2'])
        self.assertEqual(words[1]['active_speakers'],['speaker_1','speaker_2'])
        spans=speaker_segments(probabilities,1.,pad_onset=0,pad_offset=0,min_on=0,min_off=0)
        self.assertEqual([(s['start'],s['end'],s['speaker']) for s in spans],[(0.,.5,'speaker_1'),(.2,.3,'speaker_2'),(.5,1.,'speaker_2')])

    def test_state_against_upstream_cpp(self):
        path=ROOT/'artifacts/diar-oracle-build/bin/libdiar_state_reference.so'
        if not path.exists():self.skipTest('Build validation-only upstream AOSC adapter first')
        lib=C.CDLL(str(path));ptr=C.POINTER(C.c_float)
        lib.state_new.argtypes=[ptr];lib.state_new.restype=C.c_void_p
        lib.state_update.argtypes=[C.c_void_p,ptr,C.c_int,ptr,C.c_int,C.c_int]
        for name in ('state_cache_frames','state_fifo_frames'):
            getattr(lib,name).argtypes=[C.c_void_p];getattr(lib,name).restype=C.c_int
        for name in ('state_cache','state_fifo','state_probs'):
            getattr(lib,name).argtypes=[C.c_void_p];getattr(lib,name).restype=ptr
        lib.state_delete.argtypes=[C.c_void_p]
        model=DiarModel(ROOT/'models/Nemotron-3-Diarization.q8_0.gguf')
        state=SpeakerState(model,Geometry());p=lib.state_new(state.silence.ctypes.data_as(ptr))
        rng=np.random.default_rng(917)
        try:
            # Clean speech, overlaps, silence, tied scores, partial tail, and repeated compressions.
            for step,t in enumerate([380]*8+[19]):
                embeddings=rng.normal(size=(t,512)).astype(np.float32)
                count=len(state.cache)+len(state.fifo)+t
                probs=rng.uniform(0,.2,size=(count,8)).astype(np.float32)
                for frame in range(count):
                    speaker=(frame//47+step)%8;probs[frame,speaker]=rng.uniform(.55,.99)
                if step==1:probs[:]=0
                if step==2:probs[:]=.9
                if step==3:probs[:]=.6
                if step==4:probs[:]=.25
                rc=40 if t==380 else 0
                lib.state_update(p,embeddings.ctypes.data_as(ptr),t,probs.ctypes.data_as(ptr),0,rc)
                state.update(embeddings,probs,0,rc)
                nc,nf=lib.state_cache_frames(p),lib.state_fifo_frames(p)
                self.assertEqual((len(state.cache),len(state.fifo)),(nc,nf))
                for name,actual,shape in (('state_cache',state.cache,(nc,512)),('state_fifo',state.fifo,(nf,512))):
                    if actual.size:
                        expected=np.ctypeslib.as_array(getattr(lib,name)(p),shape=(actual.size,)).reshape(shape)
                        np.testing.assert_array_equal(actual,expected)
                if state.cache_probs is not None:
                    expected=np.ctypeslib.as_array(lib.state_probs(p),shape=(nc*8,)).reshape(nc,8)
                    np.testing.assert_array_equal(state.cache_probs,expected)
            self.assertGreater(state.compressions,5)
        finally:lib.state_delete(p)


if __name__=='__main__':unittest.main()
