# SPDX-License-Identifier: Apache-2.0
"""Linux CUDA Driver API adapter; no CUDA Python bindings or framework required."""
import ctypes as C
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path

class KernelNodeParams(C.Structure):
    # CUDA_KERNEL_NODE_PARAMS_v1, used by the unversioned Driver API symbol.
    _fields_ = [("func", C.c_void_p),
                *[(name, C.c_uint) for name in ("grid_x", "grid_y", "grid_z",
                                               "block_x", "block_y", "block_z", "shared_bytes")],
                ("parameters", C.POINTER(C.c_void_p)), ("extra", C.POINTER(C.c_void_p))]


@contextmanager
def gpu_lock(index=0):
    """Serialize cooperating local projects on this user's GPU."""
    path = Path(f"/tmp/txir-gpu-{os.getuid()}-{index}.lock")
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class Driver:
    def __init__(self, index=0):
        self.lib = C.CDLL("libcuda.so.1")
        signatures = {
            "cuInit": [C.c_uint],
            "cuDeviceGet": [C.POINTER(C.c_int), C.c_int],
            "cuDeviceGetName": [C.c_void_p, C.c_int, C.c_int],
            "cuDeviceComputeCapability": [C.POINTER(C.c_int), C.POINTER(C.c_int), C.c_int],
            "cuDeviceTotalMem_v2": [C.POINTER(C.c_size_t), C.c_int],
            "cuDevicePrimaryCtxRetain": [C.POINTER(C.c_void_p), C.c_int],
            "cuDevicePrimaryCtxRelease_v2": [C.c_int],
            "cuCtxSetCurrent": [C.c_void_p],
            "cuCtxGetCurrent": [C.POINTER(C.c_void_p)],
            "cuCtxSynchronize": [],
            "cuMemAlloc_v2": [C.POINTER(C.c_uint64), C.c_size_t],
            "cuMemFree_v2": [C.c_uint64],
            "cuMemcpyHtoD_v2": [C.c_uint64, C.c_void_p, C.c_size_t],
            "cuMemcpyDtoH_v2": [C.c_void_p, C.c_uint64, C.c_size_t],
            "cuModuleLoad": [C.POINTER(C.c_void_p), C.c_char_p],
            "cuModuleUnload": [C.c_void_p],
            "cuModuleGetFunction": [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
            "cuLaunchKernel": [C.c_void_p, C.c_uint, C.c_uint, C.c_uint,
                               C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_void_p,
                               C.POINTER(C.c_void_p), C.c_void_p],
            "cuEventCreate": [C.POINTER(C.c_void_p), C.c_uint],
            "cuEventRecord": [C.c_void_p, C.c_void_p],
            "cuEventSynchronize": [C.c_void_p],
            "cuEventElapsedTime": [C.POINTER(C.c_float), C.c_void_p, C.c_void_p],
            "cuEventDestroy_v2": [C.c_void_p],
            "cuStreamCreate": [C.POINTER(C.c_void_p), C.c_uint],
            "cuStreamSynchronize": [C.c_void_p],
            "cuStreamDestroy_v2": [C.c_void_p],
            "cuGraphCreate": [C.POINTER(C.c_void_p), C.c_uint],
            "cuGraphAddKernelNode": [C.POINTER(C.c_void_p), C.c_void_p,
                                     C.POINTER(C.c_void_p), C.c_size_t, C.POINTER(KernelNodeParams)],
            "cuGraphAddEventRecordNode": [C.POINTER(C.c_void_p), C.c_void_p,
                                          C.POINTER(C.c_void_p), C.c_size_t, C.c_void_p],
            "cuGraphInstantiateWithFlags": [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint64],
            "cuGraphUpload": [C.c_void_p, C.c_void_p],
            "cuGraphLaunch": [C.c_void_p, C.c_void_p],
            "cuGraphExecDestroy": [C.c_void_p],
            "cuGraphDestroy": [C.c_void_p],
            "cuGetErrorName": [C.c_int, C.POINTER(C.c_char_p)],
            "cuGetErrorString": [C.c_int, C.POINTER(C.c_char_p)],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, C.c_int
        self.call("cuInit", 0)
        self.device = C.c_int()
        self.call("cuDeviceGet", C.byref(self.device), index)
        self.context = C.c_void_p()
        self.previous_context = C.c_void_p()
        self.call("cuCtxGetCurrent", C.byref(self.previous_context))
        self.call("cuDevicePrimaryCtxRetain", C.byref(self.context), self.device)
        try:
            self.call("cuCtxSetCurrent", self.context)
        except Exception:
            self.call("cuDevicePrimaryCtxRelease_v2", self.device)
            raise

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            error, message = C.c_char_p(), C.c_char_p()
            self.lib.cuGetErrorName(status, C.byref(error))
            self.lib.cuGetErrorString(status, C.byref(message))
            raise RuntimeError(f"{name}: {error.value!r}: {message.value!r} (CUDA {status})")

    def info(self):
        name = C.create_string_buffer(256)
        major, minor, memory = C.c_int(), C.c_int(), C.c_size_t()
        self.call("cuDeviceGetName", name, len(name), self.device)
        self.call("cuDeviceComputeCapability", C.byref(major), C.byref(minor), self.device)
        self.call("cuDeviceTotalMem_v2", C.byref(memory), self.device)
        return {"name": name.value.decode(), "arch": f"sm_{major.value}{minor.value}",
                "total_memory_bytes": memory.value}

    def synchronize(self):
        self.call("cuCtxSynchronize")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            self.call("cuCtxSetCurrent", self.previous_context)
        finally:
            self.call("cuDevicePrimaryCtxRelease_v2", self.device)

