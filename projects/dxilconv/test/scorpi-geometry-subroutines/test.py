#!/usr/bin/env python3
"""Exercise legacy CALL/LABEL normalization before geometry capture.

Usage: test.py <DXC build directory>.
"""
import ctypes as C
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

build = Path(sys.argv[1]).resolve()
fixture = Path(__file__).resolve().parent.parent / 'scorpi-stream-output-only'
lib = C.CDLL(str(build / 'lib/libScorpiDxbcToDxil.dylib'))
normalize = lib.ScorpiCreateDxilBinary
normalize.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t,
                      C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.c_void_p, C.c_size_t]
capture = lib.ScorpiCreateGeometryShaderCapture
capture.argtypes = [C.c_void_p, C.c_size_t, C.c_uint32, C.c_int,
                    C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t),
                    C.c_void_p, C.c_size_t]
lib.ScorpiFreeDxilBinary.argtypes = [C.c_void_p]


def shader(depth):
    data = (fixture / 'geometry.tokens').read_bytes()
    words = list(struct.unpack('<%dI' % (len(data) // 4), data))
    positions = []
    at = 2
    while at < len(words):
        positions.append(at)
        length = (words[at] >> 24) & 127
        assert length
        at += length
    assert at == len(words) and not any(words[i] & 2047 == 44 for i in positions)
    # The captured fixture starts execution with an atomic counter operation.
    first = next(i for i in positions if words[i] & 2047 == 180)
    if depth:
        words[first:first] = [0x03000004, 0x0010a000, 0]
        for n in range(depth):
            words += [0x0300002c, 0x0010a000, n,
                      0x05000036, 0x00100012, 0, 0x00004001, 13]
            if n + 1 < depth:
                words += [0x03000004, 0x0010a000, n + 1]
            words += [0x0100003e]
    words[1] = len(words)
    return struct.pack('<%dI' % len(words), *words)


def binary_call(fn, *args):
    result, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(2048)
    rc = fn(*args, C.byref(result), C.byref(size), error, len(error))
    assert rc == 0, (rc, error.value)
    try:
        return C.string_at(result, size.value)
    finally:
        lib.ScorpiFreeDxilBinary(result)


signature = bytearray((fixture / 'geometry.signatures').read_bytes())
struct.pack_into('<I', signature, 4, 1)  # Explicit stream-output-only signature.
with tempfile.TemporaryDirectory(prefix='scorpi-geometry-subroutines-') as name:
    tmp = Path(name)
    for depth in [0, 1, 32, 33]:
        code = shader(depth)
        dxil = binary_call(normalize, code, len(code), bytes(signature), len(signature))
        for mode in [1, 2, 3, 4]:
            layout = (C.c_uint32 * 5)()
            result = binary_call(capture, dxil, len(dxil), 0x7fff0000, mode, layout)
            path = tmp / 'capture.dxil'
            path.write_bytes(result)
            ir = subprocess.check_output([str(build / 'bin/dxc'), '-dumpbin', str(path)], text=True)
            assert 'dx.label.' not in ir
            assert layout[0] > 0 and layout[1] > 0
            print('PASS depth', depth, 'capture mode', mode)
