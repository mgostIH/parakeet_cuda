# SPDX-License-Identifier: Apache-2.0
"""Read-only GGUF mapping; Q8 decoding does not duplicate the whole model."""
from dataclasses import dataclass
import math
from pathlib import Path
import struct
import numpy as np


@dataclass(frozen=True)
class Tensor:
    name: str
    dims: tuple
    kind: int
    offset: int
    nbytes: int

    @property
    def shape(self):
        return tuple(reversed(self.dims))

    @property
    def elements(self):
        return math.prod(self.dims)


class Model:
    def __init__(self, path, *, validate=True):
        self.path = Path(path)
        self.metadata, self.tensors = {}, {}
        with self.path.open('rb') as f:
            def unpack(fmt):
                data = f.read(struct.calcsize('<' + fmt))
                if len(data) != struct.calcsize('<' + fmt):
                    raise ValueError('Truncated GGUF')
                return struct.unpack('<' + fmt, data)[0]
            def string():
                n = unpack('Q')
                data = f.read(n)
                if len(data) != n:
                    raise ValueError('Truncated GGUF string')
                return data.decode('utf8')
            def value(kind):
                fmts = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f',
                        7:'?', 10:'Q', 11:'q', 12:'d'}
                if kind == 8:
                    return string()
                if kind == 9:
                    item, n = unpack('I'), unpack('Q')
                    return [value(item) for _ in range(n)]
                return unpack(fmts[kind])
            if f.read(4) != b'GGUF' or unpack('I') != 3:
                raise ValueError('Expected GGUF version 3')
            nt, nk = unpack('Q'), unpack('Q')
            for _ in range(nk):
                key = string()
                self.metadata[key] = value(unpack('I'))
            for _ in range(nt):
                name = string()
                dims = tuple(unpack('Q') for _ in range(unpack('I')))
                kind, offset = unpack('I'), unpack('Q')
                n = math.prod(dims)
                if kind not in (0, 1, 8) or (kind == 8 and n % 32):
                    raise ValueError(f'Unsupported GGUF tensor {name}: {kind}')
                size = n * {0:4, 1:2, 8:34}[kind] // (32 if kind == 8 else 1)
                if name in self.tensors:
                    raise ValueError(f'Duplicate tensor {name}')
                self.tensors[name] = Tensor(name, dims, kind, offset, size)
            align = self.metadata.get('general.alignment', 32)
            self.data_start = (f.tell() + align - 1) // align * align
        self.mapping = np.memmap(self.path, mode='r', dtype=np.uint8)
        for t in self.tensors.values():
            if self.data_start + t.offset + t.nbytes > self.mapping.size:
                raise ValueError(f'Truncated tensor {t.name}')
        if not validate:
            return
        expected = {'asr.encoder.d_model':1024, 'asr.encoder.n_layers':24,
                    'asr.encoder.n_heads':8, 'asr.encoder.d_ff':4096,
                    'asr.encoder.subsampling_factor':8, 'asr.head_type':'tdt',
                    'asr.encoder.q8_layout':'block_q8_0'}
        for key, required in expected.items():
            if self.metadata.get(key) != required:
                raise ValueError(f'Unsupported model contract: {key}')
        self.validate_schema()

    def validate_schema(self):
        def require(name,dims,kind):
            t=self.tensors.get(name)
            if t is None or t.dims!=tuple(dims) or t.kind!=kind:
                raise ValueError(f'Unsupported tensor schema: {name}, expected {dims}, type {kind}')
        metadata={'asr.encoder.feat_in':128,'asr.encoder.use_bias':False,
                  'asr.encoder.xscaling':False,'asr.encoder.conv_kernel_size':9,
                  'asr.encoder.conv_norm':'batch_norm','asr.encoder.cache_supported':False,
                  'asr.encoder.offline_left_ctx':-1,'asr.encoder.offline_right_ctx':-1,
                  'asr.encoder.pos_emb_max_len':5000,'asr.preprocessor.sample_rate':16000,
                  'asr.preprocessor.n_fft':512,'asr.preprocessor.features':128,
                  'asr.preprocessor.normalize':'per_feature','asr.preprocessor.stft_center_window':True,
                  'asr.preprocessor.hann_periodic':False,'asr.preprocessor.mask_invalid_frames':True,
                  'asr.tdt.durations':[0,1,2,3,4]}
        for key,value in metadata.items():
            if self.metadata.get(key)!=value:raise ValueError(f'Unsupported model metadata: {key}')
        require('encoder.pos_enc.pe',(1024,9999),0)
        require('preprocessor.fb',(257,128),0)
        for i in (0,2,5):
            require(f'encoder.pre_encode.conv.{i}.weight',(3,3,1,256),1)
            require(f'encoder.pre_encode.conv.{i}.bias',(256,),0)
        for i in (3,6):
            require(f'encoder.pre_encode.conv.{i}.weight',(1,1,256,256),1)
            require(f'encoder.pre_encode.conv.{i}.bias',(256,),0)
        require('encoder.pre_encode.out.weight',(4096,1024),8)
        require('encoder.pre_encode.out.bias',(1024,),0)
        for layer in range(24):
            base=f'encoder.layers.{layer}.'
            for n in ('norm_feed_forward1','norm_self_att','norm_conv','norm_feed_forward2','norm_out'):
                for suffix in ('weight','bias'):require(base+n+'.'+suffix,(1024,),0)
            for n in ('feed_forward1','feed_forward2'):
                require(base+n+'.linear1.weight',(1024,4096),8)
                require(base+n+'.linear2.weight',(4096,1024),8)
            for n in ('q','k','v','out','pos'):require(base+'self_attn.linear_'+n+'.weight',(1024,1024),8)
            for n in ('u','v'):require(base+'self_attn.pos_bias_'+n,(128,8),0)
            require(base+'conv.pointwise_conv1.weight',(1024,2048),8)
            require(base+'conv.pointwise_conv2.weight',(1024,1024),8)
            require(base+'conv.depthwise_conv.weight',(9,1,1024),1)
            for n in ('weight','bias','running_mean','running_var'):require(base+'conv.batch_norm.'+n,(1024,),0)
        require('decoder.prediction.embed.weight',(640,8193),1)
        for l in range(2):
            for kind in ('ih','hh'):
                base=f'decoder.prediction.dec_rnn.lstm.{kind}_l{l}'
                require(base+'.weight',(640,2560),8);require(base+'.bias',(2560,),0)
        for name,dims in (('joint.enc',(1024,640)),('joint.pred',(640,640)),('joint.joint_net.2',(640,8198))):
            require(name+'.weight',dims,8);require(name+'.bias',(dims[1],),0)
        if len(self.metadata.get('asr.tokenizer.vocab',[]))!=8192:
            raise ValueError('Expected the embedded 8192-piece vocabulary')

    def raw(self, name):
        t = self.tensors[name]
        start = self.data_start + t.offset
        return self.mapping[start:start+t.nbytes]

    def array(self, name):
        t = self.tensors[name]
        raw = self.raw(name)
        if t.kind == 8:
            blocks = raw.reshape(-1, 34)
            scales = blocks[:, :2].copy().view('<f2').astype(np.float32)
            result = blocks[:, 2:].view(np.int8).astype(np.float32) * scales
        else:
            result = raw.view('<f4' if t.kind == 0 else '<f2')
        return result.reshape(t.shape)

    def names(self, prefix):
        return [n for n in self.tensors if n.startswith(prefix)]

    def text(self, ids):
        vocab = self.metadata['asr.tokenizer.vocab']
        chunks = bytearray()
        for token in ids:
            piece = vocab[token]
            if piece.startswith('<0x') and piece.endswith('>'):
                chunks.append(int(piece[3:-1], 16))
            elif piece not in ('<s>', '</s>', '<pad>'):
                chunks.extend(piece.replace('▁', ' ').encode('utf8'))
        return chunks.decode('utf8', errors='replace').strip()
