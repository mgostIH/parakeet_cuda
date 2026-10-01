# Contributing

Use the standalone setup and validation commands in README.md. Changes should
retain explicit memory limits and independent NumPy goldens. New GPU support
needs a real device/model run: automatic compilation alone is not validation.

For a bug report, include GPU/driver/toolkit, Python/NumPy versions, command,
audio format/duration, expected versus actual behavior and stderr. A small
shareable sample is useful when reproduction depends on the waveform.

For an optimization, provide the exact source/target, fixed numerical
contract, fresh measurement output, warm/cold conditions, all raw timing
samples and process VRAM. Compare the same audio/mode/model hashes. Benchmark
noise, transcript agreement and model accuracy are separate questions.

No model weights, private recordings, transcripts, generated HTML, binary
runtimes or build artifacts belong in a source PR. These directories/formats
are ignored and checked in release archives. Model terms are separate from
this code's Apache-2.0 license.
