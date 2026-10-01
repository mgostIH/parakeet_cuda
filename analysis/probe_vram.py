#!/usr/bin/env python3
"""Read CUDA free/total memory after retaining GPU 0's primary context."""

import ctypes as c
import sys


lib = c.CDLL("libcuda.so.1")


def cuda(name, argtypes, *args):
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    fn.restype = c.c_int
    status = fn(*args)
    if status:
        raise RuntimeError(f"{name} failed with CUDA status {status}")


cuda("cuInit", [c.c_uint], 0)
device = c.c_int()
cuda("cuDeviceGet", [c.POINTER(c.c_int), c.c_int], c.byref(device), 0)
previous = c.c_void_p()
cuda("cuCtxGetCurrent", [c.POINTER(c.c_void_p)], c.byref(previous))
context = c.c_void_p()
cuda("cuDevicePrimaryCtxRetain", [c.POINTER(c.c_void_p), c.c_int], c.byref(context), device)
try:
    cuda("cuCtxSetCurrent", [c.c_void_p], context)
    free, total = c.c_size_t(), c.c_size_t()
    cuda("cuMemGetInfo_v2", [c.POINTER(c.c_size_t), c.POINTER(c.c_size_t)],
         c.byref(free), c.byref(total))
    print(f"CUDA context active: {free.value / 1048576:.1f} MiB free, "
          f"{total.value / 1048576:.1f} MiB total")
    if len(sys.argv) > 1:
        reserve_bytes = int(sys.argv[1]) * 1048576
        if free.value < reserve_bytes + 300 * 1048576:
            raise RuntimeError("Less than 300 MiB would remain; reservation skipped")
        pointer = c.c_uint64()
        cuda("cuMemAlloc_v2", [c.POINTER(c.c_uint64), c.c_size_t],
             c.byref(pointer), reserve_bytes)
        try:
            cuda("cuMemGetInfo_v2", [c.POINTER(c.c_size_t), c.POINTER(c.c_size_t)],
                 c.byref(free), c.byref(total))
            print(f"Reserved {reserve_bytes / 1048576:.1f} MiB temporarily; "
                  f"{free.value / 1048576:.1f} MiB free")
        finally:
            cuda("cuMemFree_v2", [c.c_uint64], pointer)
finally:
    cuda("cuCtxSetCurrent", [c.c_void_p], previous)
    cuda("cuDevicePrimaryCtxRelease_v2", [c.c_int], device)
