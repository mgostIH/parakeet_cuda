"""Regenerate a listening report from a saved speaker-labeled transcript."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from parakeet_cuda.report import write_report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('transcript',type=Path);parser.add_argument('audio',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    write_report(json.loads(args.transcript.read_text()),args.audio,args.output)


if __name__=='__main__':main()
