import tempfile
from pathlib import Path
import unittest
import wave
import numpy as np
from parakeet_cuda.recording import segments,merge_owned_words
from parakeet_cuda.decoder import decode


class HostContract(unittest.TestCase):
    def test_segment_coverage_and_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'audio.wav'
            with wave.open(str(path),'wb') as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000)
                f.writeframes(np.zeros(21*16000,np.int16).tobytes())
            chunks=list(segments(path,seconds=7,overlap=1))
            self.assertEqual(chunks[0].keep_start,0)
            self.assertEqual(chunks[-1].keep_end,21)
            for i,c in enumerate(chunks):
                self.assertLessEqual(len(c.audio),7*16000)
                if i:self.assertEqual(c.keep_start,chunks[i-1].keep_end)
                self.assertGreater(c.keep_end,c.keep_start)

    def test_overlap_duplicate_and_real_repeat(self):
        def word(text,s,e):return {'word':text,'start':s,'end':e}
        a=[word('realization.',104,105.6)]
        merge_owned_words(a,[word('Realization.',104.06,107.98),word('The',107.98,108.3)])
        self.assertEqual([w['word'] for w in a],['realization.','The'])
        a=[word('very',1,1.2)]
        merge_owned_words(a,[word('very',1.25,1.45)])
        self.assertEqual(len(a),2)

    def test_tdt_blank_cache_and_zero_guard(self):
        class Model:
            metadata={'asr.tdt.durations':[0,1,2,3,4]}
            def text(self,ids):return str(ids)
        class Decoder:
            def __init__(self):self.predicted=[];self.committed=0;self.calls=0
            def predict(self,token):self.predicted.append(token)
            def joint(self,frame):
                self.calls+=1
                return (8192,0) if self.calls in (1,2) else (7,0)
            def commit(self):self.committed+=1
        d=Decoder();result=decode(np.zeros((3,640),np.float32),d,Model())
        self.assertEqual(d.predicted[:2],[8192,7])
        self.assertEqual(len(result['tokens']),10)
        self.assertEqual(d.committed,10)
        self.assertEqual(result['joint_calls'],12)


if __name__=='__main__':unittest.main()
