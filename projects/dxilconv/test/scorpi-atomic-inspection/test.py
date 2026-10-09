#!/usr/bin/env python3
"""Inspect opcodes without requiring DXIL resource metadata deserialization.

Usage: test.py <DXC build directory> (dxc and llvm-as targets must be built).
"""
import ctypes as C
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

build = Path(sys.argv[1]).resolve()
lib = C.CDLL(str(build / 'lib/libScorpiDxbcToDxil.dylib'))
inspect = lib.ScorpiInspectAtomic64
inspect.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(C.c_int)]
inspect.restype = C.c_int


def check(code, expected):
    required = C.c_int(-1)
    status = inspect(code, len(code), C.byref(required))
    assert status == 0 and required.value == expected, (status, required.value, expected)


def container(bitcode):
    program = struct.pack('<II4sIII', 0x50066, (24 + len(bitcode)) // 4,
                          b'DXIL', 0x106, 16, len(bitcode)) + bitcode
    return (struct.pack('<4s16sHHIII', b'DXBC', bytes(16), 1, 0,
                        44 + len(program), 1, 36) +
            struct.pack('<4sI', b'DXIL', len(program)) + program)


with tempfile.TemporaryDirectory(prefix='scorpi-atomic-inspection-') as tmp:
    tmp = Path(tmp)
    for name, scalar, operation, expected in [
        ('add64', 'uint64_t', 'InterlockedAdd(output[id.x], uint64_t(7), old);', 1),
        ('compare64', 'uint64_t', 'InterlockedCompareExchange(output[id.x], uint64_t(7), uint64_t(9), old);', 1),
        ('arithmetic64', 'uint64_t', 'output[id.x] = output[id.x] + uint64_t(id.y);', 0),
        ('add32', 'uint', 'InterlockedAdd(output[id.x], 7, old);', 0),
    ]:
        src = tmp / 'shader.hlsl'
        src.write_text(f'RWStructuredBuffer<{scalar}> output : register(u0);\n'
                       '[numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID) {\n'
                       f'{scalar} old; {operation}\n}}')
        subprocess.run([str(build / 'bin/dxc'), '-T', 'cs_6_6', '-E', 'main',
                        '-Fo', str(tmp / 'shader.dxil'), '-Fc', str(tmp / 'shader.ll'), str(src)], check=True)
        check((tmp / 'shader.dxil').read_bytes(), expected)
        # A null range-size metadata operand reproduces the crashing loader
        # dereference. Opcode inspection has no need to interpret this field.
        ir = (tmp / 'shader.ll').read_text()
        ir, count = re.subn(r'(!\d+ = !\{i32 0, [^\n]+, !"[^\n]*", i32 0, i32 0, )i32 1(, i32 12,)',
                            r'\g<1>null\2', ir)
        assert count == 1, ir
        (tmp / 'incomplete.ll').write_text(ir)
        subprocess.run([str(build / 'bin/llvm-as'), str(tmp / 'incomplete.ll'),
                        '-o', str(tmp / 'incomplete.bc')], check=True)
        check(container((tmp / 'incomplete.bc').read_bytes()), expected)
        print('PASS', name, 'original and incomplete resource metadata')
