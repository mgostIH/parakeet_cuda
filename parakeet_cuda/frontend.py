# SPDX-License-Identifier: Apache-2.0
"""CPU FFT frontend, matching the inspected NeMo-Speech.cpp feature contract."""
import contextlib
import hashlib
from pathlib import Path
import subprocess
import struct
import tempfile
import wave
import numpy as np


class PCMSource:
    """One shared PCM16 mapping; read() converts only a requested window."""
    def __init__(self,path):
        self.path=Path(path)
        with wave.open(str(path),'rb') as wav:
            if (wav.getnchannels(),wav.getframerate(),wav.getsampwidth())!=(1,16000,2):
                raise ValueError('WAV must be mono 16 kHz PCM16')
            self.samples=wav.getnframes()
        with self.path.open('rb') as file:
            header=file.read(12)
            if header[:4]!=b'RIFF' or header[8:]!=b'WAVE':raise ValueError('Expected RIFF WAV')
            while True:
                chunk=file.read(8)
                if len(chunk)!=8:raise ValueError('WAV data chunk missing')
                name,size=struct.unpack('<4sI',chunk)
                if name==b'data':
                    offset=file.tell();break
                file.seek(size+(size&1),1)
        self.mapping=np.memmap(self.path,mode='r',dtype='<i2',offset=offset,shape=(self.samples,)) if self.samples else None

    def __len__(self):return self.samples

    def read(self,start=0,count=None):
        count=self.samples-start if count is None else count
        if start<0 or count<0 or start>self.samples:raise ValueError('Invalid PCM window')
        if self.mapping is None:return np.empty(0,np.float32)
        return self.mapping[start:start+count].astype(np.float32)/np.float32(32768)

    def __enter__(self):return self
    def __exit__(self,*_):
        if self.mapping is not None:self.mapping._mmap.close()


@contextlib.contextmanager
def pcm_file(path):
    path = Path(path)
    if path.suffix.lower() == '.wav':
        yield path
        return
    with tempfile.TemporaryDirectory(prefix='parakeet-') as directory:
        wav = Path(directory) / 'audio.wav'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
                        '-y', '-i', str(path), '-ac', '1', '-ar', '16000',
                        '-c:a', 'pcm_s16le', str(wav)], check=True)
        yield wav


def read_pcm(path):
    with wave.open(str(path), 'rb') as f:
        if (f.getnchannels(), f.getframerate(), f.getsampwidth()) != (1, 16000, 2):
            raise ValueError('WAV must be mono 16 kHz PCM16')
        data = f.readframes(f.getnframes())
    return np.frombuffer(data, dtype='<i2').astype(np.float32) / np.float32(32768)


def mel_features(audio, model):
    if not len(audio):
        raise ValueError('Audio is empty')
    m = model.metadata
    if m.get('asr.preprocessor.normalize') != 'per_feature':
        raise ValueError('Expected per_feature mel normalization')
    pre = audio.copy()
    pre[1:] -= np.float32(m['asr.preprocessor.preemph']) * audio[:-1]
    nfft, hop, win = 512, 160, 400
    window = np.zeros(nfft, dtype=np.float32)
    # Float32 window follows the C++ frontend, including symmetric Hann.
    phase = np.float32(2*np.pi) * np.arange(win, dtype=np.float32) / np.float32(win-1)
    window[(nfft-win)//2:(nfft+win)//2] = np.float32(.5) * (1-np.cos(phase))
    padded = np.pad(pre, (nfft//2, nfft//2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, nfft)[::hop]
    fb = model.array('preprocessor.fb').astype(np.float32)
    features = np.empty((len(frames), 128), dtype=np.float32)
    for start in range(0, len(frames), 512):
        fft = np.fft.rfft(frames[start:start+512] * window, axis=1)
        power = (fft.real**2 + fft.imag**2).astype(np.float32)
        features[start:start+512] = np.log(power @ fb.T + np.float32(2**-24))
    valid = min(len(audio)//hop, len(features))
    if valid:
        values = features[:valid].astype(np.float64)
        mean = values.mean(axis=0)
        variance = ((values-mean)**2).sum(axis=0) / max(1, valid-1)
        features[:valid] = (values-mean) / (np.sqrt(variance)+1e-5)
    features[valid:] = 0
    return features


def pcm_identity(audio):
    return {'samples':len(audio), 'duration':len(audio)/16000,
            'float32_sha256':hashlib.sha256(audio.tobytes()).hexdigest()}
