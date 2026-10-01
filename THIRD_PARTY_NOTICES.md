# Third-party notices

## NVIDIA NeMo-Speech.cpp

Architecture, frontend conventions, streaming AOSC semantics and validation
interfaces were developed with reference to NVIDIA's native runtime.
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Apache-2.0; full upstream text is in [licenses/NVIDIA-Apache-2.0.txt](licenses/NVIDIA-Apache-2.0.txt).
The Python AOSC implementation is a modified NumPy port of its inference
algorithm. Production kernels are implemented in this repository.
The optional C++ oracle builds unchanged runtime sources fetched separately.
Upstream pin: `4c101bc7113f49101a3e11d2c994c519f41939f6`.
Source: https://github.com/NVIDIA/NeMo-Speech.cpp

## Optional ggml validation dependency

The C++ oracle uses ggml, fetched separately under the excluded upstream
source directory. Copyright (c) 2023-2026 The ggml authors; MIT license.
Pin: `c03b4e2bcece5134827881af90242086daf75be5`.
Source/license: https://github.com/ggml-org/ggml/blob/master/LICENSE
No ggml sources or binaries are included in the production Python wheel.

## Model weights

Weights have their own terms and are not covered by this code's Apache-2.0
license. The downloader retrieves only the selected GGUF from NVIDIA's
pinned Hugging Face revisions, with size and SHA-256 verification.

- Parakeet TDT 0.6B v3: [NVIDIA model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), CC-BY-4.0.
- Nemotron 3 Diarization: [NVIDIA model card](https://huggingface.co/nvidia/Nemotron-3-Diarization), OpenMDW-1.1.

## Runtime dependencies

NumPy is installed separately under its BSD-3-Clause license. CUDA, cuBLAS
and the NVIDIA driver are user-installed NVIDIA components under their own
terms; this repository does not redistribute them. TIRx is an optional
external development tool, not a production dependency.
