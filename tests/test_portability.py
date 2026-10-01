"""Host-only contracts for installation paths, downloads, reports and cache safety."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from parakeet_cuda import compiler
from parakeet_cuda.download import download,verified
from parakeet_cuda.report import write_report
from parakeet_cuda.paths import KERNEL_SOURCE


class PortabilityContract(unittest.TestCase):
    def test_cuda_source_is_packaged(self):
        source=KERNEL_SOURCE.read_text()
        self.assertIn('void dequant_q8',source)
        self.assertIn('void diar_rope_qkv',source)

    def test_download_rejects_corruption_and_preserves_existing_file(self):
        expected=b'verified weights';model={'repo':'test/model','revision':'fixed','filename':'test.gguf',
            'size':len(expected),'sha256':hashlib.sha256(expected).hexdigest(),'license':'test'}
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/'test.gguf';target.write_bytes(b'original')
            with patch('urllib.request.urlopen',return_value=io.BytesIO(b'corrupt')):
                with self.assertRaises(ValueError):download(model,temp)
            self.assertEqual(target.read_bytes(),b'original')
            self.assertEqual(list(Path(temp).glob('*.partial')),[])
            with patch('urllib.request.urlopen',return_value=io.BytesIO(expected)) as network:
                download(model,temp)
                self.assertEqual(network.call_args.args[0],'https://huggingface.co/test/model/resolve/fixed/test.gguf')
            self.assertTrue(verified(target,model))
            with patch('urllib.request.urlopen',side_effect=AssertionError('Already verified; no network')):
                download(model,temp)

    def test_reports_escape_transcript_and_link_large_audio(self):
        data={'pcm':{'duration':1.},'speaker_turns':[{'start':0.,'end':1.,'speaker':'speaker_1','text':'</script><script>bad()</script>'}],
              'diarization':{'segments':[{'start':0.,'end':1.,'speaker':'speaker_1'}]}}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);audio=root/'a & b.m4a';audio.write_bytes(b'tiny audio')
            output=root/'report.html';write_report(data,audio,output)
            text=output.read_text();self.assertIn('src="data:audio/',text)
            self.assertNotIn('</script><script>bad()',text)
            with audio.open('wb') as f:f.truncate(10*1024**2+1)
            write_report(data,audio,output);text=output.read_text()
            self.assertIn('src="a%20%26%20b.m4a"',text)
            self.assertNotIn('src="data:',text)
            self.assertLess(output.stat().st_size,10*1024)

    def test_compilation_cache_includes_arch_and_does_not_publish_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            def run(cmd,**kwargs):
                if cmd[1]=='--cubin' and '-arch=sm_86' in cmd:return subprocess.CompletedProcess(cmd,1,'','unsupported compiler fixture')
                Path(cmd[cmd.index('-o')+1]).write_bytes(b'compiled')
                return subprocess.CompletedProcess(cmd,0,'','')
            with patch.object(compiler,'find_nvcc',return_value=(Path('/fixture/nvcc'),'release 12.9, fixture')):
                with patch.object(compiler.subprocess,'run',side_effect=run) as invoke:
                    out=compiler.compile_runtime('sm_50',temp)
                    self.assertTrue(out.is_file());self.assertEqual(invoke.call_count,2)
                    self.assertEqual(compiler.compile_runtime('sm_50',temp),out)
                    self.assertEqual(invoke.call_count,2)
                    with self.assertRaises(RuntimeError):compiler.compile_runtime('sm_86',temp)
                    self.assertEqual(len(list(Path(temp).glob('*/compile.json'))),1)
                    self.assertTrue(list(Path(temp).glob('*.failure.log')))
                    self.assertEqual(out.read_bytes(),b'compiled')


if __name__=='__main__':unittest.main()
