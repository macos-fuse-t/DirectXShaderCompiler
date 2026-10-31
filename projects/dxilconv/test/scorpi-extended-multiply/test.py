#!/usr/bin/env python3
"""Check extended-multiply lowering and its low-only fast path.

Usage: test.py <DXC build directory>; numerical semantics have native coverage.
"""
import ctypes as C
from pathlib import Path
import struct
import re
import subprocess
import sys
import tempfile

build = Path(sys.argv[1]).resolve()
lib = C.CDLL(str(build / 'lib/libScorpiDxbcToDxil.dylib'))
lower = lib.ScorpiLowerTypedBufferViewsWithSamplingPolicy
lower.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t,
                  C.c_uint32, C.c_uint32, C.c_uint32, C.c_void_p, C.c_size_t,
                  C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.c_void_p, C.c_size_t]
lib.ScorpiFreeDxilBinary.argtypes = [C.c_void_p]


def adapt(code):
    output, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(2048)
    rc = lower(code, len(code), None, 0, 0xffffffff, 0, 0, None, 0,
               C.byref(output), C.byref(size), error, len(error))
    assert rc == 0, (rc, error.value)
    try:
        return C.string_at(output, size.value) if output.value else None
    finally:
        lib.ScorpiFreeDxilBinary(output)


def shader(signed, half):
    code = [0x50050, 0, 0x04000059, 0x00208e46, 0, 1,
            0x0300009d, 0x0011e000, 0, 0x02000068, 2,
            0x0400009b, 1, 1, 1]
    start = len(code)
    code.append(38 if signed else 81)
    for i in range(2):
        code += [0xd000] if half == (2 if i == 0 else 1) else [0x001000f2, i]
    code += [0x0020800a, 0, 0, 0x0020801a, 0, 0]
    code[start] |= (len(code) - start) << 24
    for i in range(2):
        if half != (2 if i == 0 else 1):
            code += [0x070000a6, 0x0011e0f2, 0, 0x00004001, i * 16, 0x00100e46, i]
    code.append(0x0100003e)
    code[1] = len(code)
    return struct.pack('<%dI' % len(code), *code)


with tempfile.TemporaryDirectory(prefix='scorpi-extended-multiply-') as name:
    tmp = Path(name)
    for signed in (False, True):
        for half in range(3):
            result = adapt(shader(signed, half))
            assert result is not None
            assert adapt(result) is None, 'lowering must be idempotent'
            (tmp / 'shader.dxil').write_bytes(result)
            ir = subprocess.check_output([str(build / 'bin/dxc'), '-dumpbin',
                                          str(tmp / 'shader.dxil')], text=True)
            assert 'call %dx.types.twoi32' not in ir
            wide = re.search(r'\bmul (?:nuw |nsw )*i64\b', ir)
            if half == 2:
                assert re.search(r'\bmul (?:nuw |nsw )*i32\b', ir) and not wide, 'low-only multiply must stay i32'
                assert 'sext i32' not in ir and 'zext i32' not in ir
            else:
                assert wide
                assert ('sext i32' if signed else 'zext i32') in ir
            print('PASS', 'signed' if signed else 'unsigned', ['both', 'high', 'low'][half])
