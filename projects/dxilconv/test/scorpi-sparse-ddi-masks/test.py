#!/usr/bin/env python3
"""Convert every nonempty DDI component mask without inventing gap components.

Usage: test.py <DXC build directory> [native fixture output directory].
Exercises both arbitrary inputs/outputs,
DDI component types and separately typed packed records using generated DXBC.
"""
import ctypes as C
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

build = Path(sys.argv[1]).resolve()
dxc = next(p for p in (build / 'bin/dxc', build / 'bin/dxc-3.7') if p.exists())
lib = C.CDLL(str(build / 'lib/libScorpiDxbcToDxil.dylib'))
convert = lib.ScorpiCreateDxilBinary
convert.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t,
                    C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.c_void_p, C.c_size_t]
lib.ScorpiFreeDxilBinary.argtypes = [C.c_void_p]


def shader(mask, inputs):
    operand = 0x00102002 | (mask << 4)
    words = [0x10050, 0]
    if inputs:
        words += [0x0300005f, 0x00101002 | (mask << 4), 0]
    words += [0x03000065, operand, 0]
    if inputs:
        words += [0x05000036, operand, 0, 0x00101e46, 0]
    else:
        words += [0x08000036, operand, 0, 0x00004002, 17, 31, 53, 97]
    words += [0x0100003e]
    words[1] = len(words)
    return struct.pack('<%dI' % len(words), *words)


def signature(mask, inputs, component_type, split):
    masks = [1 << c for c in range(4) if mask & (1 << c)] if split else [mask]
    entries = b''.join(struct.pack('<6I', 0, 0, m, 0, component_type, 0) for m in masks)
    count = len(masks)
    return (struct.pack('<10I', 2, 0, 24, count if inputs else 0, count, 0,
                        40, 40 + len(entries) if inputs else 40, 0, 0) +
            (entries if inputs else b'') + entries)


with tempfile.TemporaryDirectory(prefix='scorpi-sparse-ddi-') as name:
    tmp = Path(name)
    cases = 0
    for component_type in (1, 2, 3):  # uint, sint, float
        for inputs in (False, True):
            for split in (False, True):
                for mask in range(1, 16):
                    code = shader(mask, inputs)
                    sig = signature(mask, inputs, component_type, split)
                    output, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(2048)
                    rc = convert(code, len(code), sig, len(sig), C.byref(output), C.byref(size), error, len(error))
                    assert rc == 0 and output.value, (mask, inputs, component_type, split, error.value)
                    try:
                        binary = C.string_at(output, size.value)
                        (tmp / 'shader.dxil').write_bytes(binary)
                        if len(sys.argv) > 2 and component_type == 1 and not inputs and not split:
                            fixtures = Path(sys.argv[2]); fixtures.mkdir(parents=True, exist_ok=True)
                            (fixtures / f'mask-{mask}.dxil').write_bytes(binary)
                    finally:
                        lib.ScorpiFreeDxilBinary(output)
                    ir = subprocess.check_output([str(dxc), '-dumpbin', str(tmp / 'shader.dxil')], text=True)
                    section = ir.split('; Output signature:', 1)[1].split('; Pipeline Runtime Information:', 1)[0]
                    owners = {}
                    for index, components, register in re.findall(r'; SCORPI\s+(\d+)\s+([xyzw]+)\s+(\d+)\s+', section):
                        for c in components:
                            key = (int(register), 'xyzw'.index(c))
                            assert key not in owners, 'overlapping owners'
                            owners[key] = int(index)
                    assert set(owners) == {(0, c) for c in range(4) if mask & (1 << c)}, (mask, section)
                    assert len(re.findall(r'call void @dx.op.storeOutput\.(?:i32|f32)\(', ir)) == bin(mask).count('1')
                    assert len(re.findall(r'call (?:i32|float) @dx.op.loadInput\.(?:i32|f32)\(', ir)) == (bin(mask).count('1') if inputs else 0)
                    cases += 1
    print('PASS:', cases, 'typed input/output DDI mask cases')
