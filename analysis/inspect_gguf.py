#!/usr/bin/env python3
"""Print GGUF metadata and tensor storage without loading model weights."""

import collections
import math
import struct
import sys


class Reader:
    def __init__(self, handle):
        self.handle = handle

    def unpack(self, fmt):
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.handle.read(size))[0]

    def string(self):
        return self.handle.read(self.unpack("Q")).decode("utf-8")

    def value(self, kind):
        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                   6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if kind == 8:
            return self.string()
        if kind == 9:
            item_type, count = self.unpack("I"), self.unpack("Q")
            if count <= 20:
                return [self.value(item_type) for _ in range(count)]
            for _ in range(count):
                self.value(item_type)
            return f"array<{item_type}>[{count}]"
        return self.unpack(formats[kind])


def category(name):
    if name.startswith("encoder.layers."):
        return "encoder.layers.*." + name.split(".", 3)[3]
    if name.startswith("encoder.pre_encode."):
        return "encoder.pre_encode.*"
    return name.split(".", 1)[0] + ".*"


def main(path):
    with open(path, "rb") as handle:
        reader = Reader(handle)
        assert handle.read(4) == b"GGUF"
        version = reader.unpack("I")
        tensor_count, kv_count = reader.unpack("Q"), reader.unpack("Q")
        metadata = {}
        for _ in range(kv_count):
            key = reader.string()
            metadata[key] = reader.value(reader.unpack("I"))
        tensors = []
        for _ in range(tensor_count):
            name = reader.string()
            dimensions = [reader.unpack("Q") for _ in range(reader.unpack("I"))]
            kind, offset = reader.unpack("I"), reader.unpack("Q")
            elements = math.prod(dimensions)
            if kind == 0:
                nbytes = elements * 4
            elif kind == 1:
                nbytes = elements * 2
            elif kind == 8:
                nbytes = elements // 32 * 34
            else:
                raise ValueError(f"Unsupported tensor type {kind}: {name}")
            tensors.append((name, dimensions, kind, offset, nbytes))
        alignment = metadata.get("general.alignment", 32)
        data_start = (handle.tell() + alignment - 1) // alignment * alignment

    print(f"GGUF v{version}: {tensor_count} tensors, {kv_count} metadata entries")
    print(f"Tensor data starts at byte {data_start}")
    print("\nMetadata:")
    for key, value in metadata.items():
        if isinstance(value, str) and len(value) > 180:
            value = f"<string of {len(value)} characters>"
        print(f"  {key}: {value}")
    by_type = collections.Counter()
    by_family = collections.defaultdict(lambda: [0, 0])
    for name, dims, kind, offset, size in tensors:
        by_type[kind] += size
        family = category(name)
        by_family[family][0] += 1
        by_family[family][1] += size
    broad = collections.defaultdict(lambda: [0, 0])
    for name, dims, kind, offset, size in tensors:
        parts = name.split(".")
        if parts[:2] == ["encoder", "layers"]:
            group = "encoder.layers." + parts[3]
        elif parts[:2] == ["encoder", "pre_encode"]:
            group = "encoder.pre_encode"
        elif parts[:2] == ["encoder", "pos_enc"]:
            group = "encoder.pos_enc"
        else:
            group = parts[0]
        broad[group][0] += 1
        broad[group][1] += size
    print("\nTensor bytes by type (0=F32, 1=F16, 8=Q8_0):")
    for kind, size in sorted(by_type.items()):
        print(f"  {kind}: {size:,}")
    print("\nTensor bytes by component:")
    for group, (count, size) in sorted(broad.items(), key=lambda x: -x[1][1]):
        print(f"  {group}: {count} tensors, {size:,} bytes")
    print("\nLargest tensor families:")
    for family, (count, size) in sorted(by_family.items(), key=lambda x: -x[1][1])[:40]:
        print(f"  {family}: {count} tensors, {size:,} bytes")
    print("\nLargest individual tensors:")
    for name, dims, kind, offset, size in sorted(tensors, key=lambda x: -x[4])[:30]:
        print(f"  {name}: {dims}, type={kind}, {size:,} bytes")
    print("\nNon-layer tensors and layer 0 example:")
    for name, dims, kind, offset, size in tensors:
        if not name.startswith("encoder.layers.") or name.startswith("encoder.layers.0."):
            print(f"  {name}: {dims}, type={kind}, {size:,} bytes")


if __name__ == "__main__":
    main(sys.argv[1])
