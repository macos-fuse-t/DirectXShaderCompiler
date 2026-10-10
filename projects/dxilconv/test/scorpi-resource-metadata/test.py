#!/usr/bin/env python3
"""Reject malformed required DXIL metadata without crashing a compiler worker.

Usage: test.py <DXC build directory> (dxc and llvm-as must be built).
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
lower = lib.ScorpiLowerTextureLayouts
lower.argtypes = [C.c_void_p, C.c_size_t, C.c_int, C.c_void_p, C.c_size_t,
                  C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.c_void_p, C.c_size_t]
lower.restype = C.c_int


def container(bitcode):
    program = struct.pack('<II4sIII', 0x50066, (24 + len(bitcode)) // 4,
                          b'DXIL', 0x106, 16, len(bitcode)) + bitcode
    return (struct.pack('<4s16sHHIII', b'DXBC', bytes(16), 1, 0,
                        44 + len(program), 1, 36) +
            struct.pack('<4sI', b'DXIL', len(program)) + program)


def check(code, valid):
    binary, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(1024)
    status = lower(code, len(code), 1, None, 0, C.byref(binary), C.byref(size), error, len(error))
    assert (status == 0) == valid, (status, error.value)
    assert not binary.value and size.value == 0
    if not valid:
        assert error.value


with tempfile.TemporaryDirectory(prefix='scorpi-resource-metadata-') as tmp:
    tmp = Path(tmp)
    src = tmp / 'shader.hlsl'
    src.write_text('RWStructuredBuffer<uint> output : register(u0);\n'
                   '[numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID) { output[id.x] = id.y; }')
    subprocess.run([str(build / 'bin/dxc'), '-T', 'cs_6_6', '-E', 'main',
                    '-Fo', str(tmp / 'shader.dxil'), '-Fc', str(tmp / 'shader.ll'), str(src)], check=True)
    check((tmp / 'shader.dxil').read_bytes(), True)
    ir = (tmp / 'shader.ll').read_text()
    match = re.search(r'!\d+ = !\{i32 0, [^\n]+, !"[^\n]*", i32 0, i32 0, i32 1, i32 12,', ir)
    assert match, ir
    original = match.group(0)
    for name, pattern, replacement in [
        ('null-range-size', r'i32 1, i32 12,$', 'null, i32 12,'),
        ('wrong-range-size-type', r'i32 1, i32 12,$', '!"not an integer", i32 12,'),
        ('null-space', r'i32 0, i32 0, i32 1,', 'null, i32 0, i32 1,'),
        ('null-lower-bound', r'i32 0, i32 0, i32 1,', 'i32 0, null, i32 1,'),
        ('null-resource-id', r'^(!\d+ = !\{)i32 0,', r'\1null,'),
        ('null-resource-name', r'!"[^\n]*"', 'null'),
        ('wrong-resource-name-type', r'!"[^\n]*"', 'i32 1'),
    ]:
        changed, count = re.subn(pattern, replacement, original)
        assert count == 1 and changed != original
        (tmp / 'incomplete.ll').write_text(ir[:match.start()] + changed + ir[match.end():])
        subprocess.run([str(build / 'bin/llvm-as'), str(tmp / 'incomplete.ll'),
                        '-o', str(tmp / 'incomplete.bc')], check=True)
        check(container((tmp / 'incomplete.bc').read_bytes()), False)
        print('PASS', name, 'clean rejection, no output, no worker crash')
