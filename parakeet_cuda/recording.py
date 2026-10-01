# SPDX-License-Identifier: Apache-2.0
"""Bounded WAV windows with explicit segment ownership for word timestamps."""
from dataclasses import dataclass
import wave
import numpy as np


@dataclass
class Segment:
    audio: np.ndarray
    offset: float
    keep_start: float
    keep_end: float


def wav_info(path):
    with wave.open(str(path),'rb') as f:
        if (f.getnchannels(),f.getframerate(),f.getsampwidth())!=(1,16000,2):
            raise ValueError('WAV must be mono 16 kHz PCM16')
        return f.getnframes()


def segments(path,seconds=45,overlap=2,source=None):
    if not 5<=seconds<=300 or not 0<=overlap<seconds/3:
        raise ValueError('Segment duration must be 5..300 seconds and overlap less than one third')
    maximum=int(seconds*16000);overlap_samples=int(overlap*16000)
    with wave.open(str(path),'rb') as f:
        total=f.getnframes();start=0;keep_start=0
        while start<total:
            if source is None:
                f.setpos(start)
                audio=np.frombuffer(f.readframes(min(maximum,total-start)),dtype='<i2').astype(np.float32)/32768
            else:audio=source.read(start,min(maximum,total-start))
            end=start+len(audio)
            if end<total:
                # Look for a quiet 200 ms region within the final fifth of the window.
                hop=3200;begin=max(int(len(audio)*.8),overlap_samples*2)
                stops=np.arange(begin,len(audio)-hop+1,hop)
                if len(stops):
                    energy=np.array([np.mean(audio[s:s+hop].astype(np.float64)**2) for s in stops])
                    cut=int(stops[int(energy.argmin())]+hop//2)
                    audio=audio[:cut];end=start+cut
                keep_end=(end-overlap_samples/2)/16000
            else:keep_end=total/16000
            yield Segment(audio,start/16000,keep_start,keep_end)
            if end==total:break
            keep_start=keep_end;start=end-overlap_samples


def words_from_tokens(ids,times,model,offset=0):
    vocab=model.metadata['asr.tokenizer.vocab'];words=[];cur=None;pieces=[]
    for token,(start,end) in zip(ids,times):
        piece=vocab[token]
        boundary=piece.startswith('▁')
        if boundary or cur is None:
            if cur:
                cur['word']=model.text(pieces)
                if cur['word']:words.append(cur)
            cur={'word':'','start':start+offset,'end':end+offset,'confidence':1}
            pieces=[]
        pieces.append(token)
        cur['end']=max(cur['end'],end+offset)
    if cur:
        cur['word']=model.text(pieces)
        if cur['word']:words.append(cur)
    return words


def merge_owned_words(previous,current):
    """Remove an overlapping repeated suffix/prefix while retaining real repeats."""
    def lexical(word):return ''.join(c.casefold() for c in word['word'] if c.isalnum())
    limit=min(12,len(previous),len(current))
    for count in range(limit,0,-1):
        old,new=previous[-count:],current[:count]
        if [lexical(w) for w in old]!=[lexical(w) for w in new]:continue
        # Require temporal agreement for each repeated word, not just equal text.
        if all(abs(a['start']-b['start'])<.5 and
               min(a['end'],b['end'])-max(a['start'],b['start'])>.04
               for a,b in zip(old,new)):
            current=current[count:];break
    previous.extend(current)
