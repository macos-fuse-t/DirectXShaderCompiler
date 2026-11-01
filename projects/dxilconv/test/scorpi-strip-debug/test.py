#!/usr/bin/env python3
"""Source-debug DXIL preserves executable code, resources and no-debug fast path.
Usage: test.py <DXC build directory>
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
strip = lib.ScorpiStripShaderDebugInfo
strip.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(C.c_void_p),
                 C.POINTER(C.c_size_t), C.c_void_p, C.c_size_t]
lib.ScorpiFreeDxilBinary.argtypes = [C.c_void_p]

def parts(code):
    count = struct.unpack_from('<I', code, 28)[0]
    return {code[o:o+4]: code[o+8:o+8+struct.unpack_from('<I', code, o+4)[0]]
            for o in struct.unpack_from('<%dI' % count, code, 32)}

def bitcode(program):
    offset, size = struct.unpack_from('<II', program, 16)
    return program[8+offset:8+offset+size]

def transform(code):
    result, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(2048)
    rc = strip(code, len(code), C.byref(result), C.byref(size), error, len(error))
    assert rc == 0, (rc, error.value)
    if not result.value:
        assert size.value == 0
        return None
    try:
        return C.string_at(result, size.value)
    finally:
        lib.ScorpiFreeDxilBinary(result)

with tempfile.TemporaryDirectory(prefix='scorpi-strip-debug-') as name:
    tmp = Path(name)
    subprocess.run([str(build/'bin/dxc-3.7'), '-T', 'vs_6_0', '-E', 'main',
                    '-Od', '-Zi', '-Qembed_debug', str(Path(__file__).with_name('shader.hlsl')),
                    '-Fo', str(tmp/'shader.dxil')], check=True)
    original = parts((tmp/'shader.dxil').read_bytes())
    # DDI callers can supply the debug program directly, without an ILDB container tag.
    debug = original[b'ILDB']
    stripped = transform(debug)
    assert stripped is not None
    assert transform(stripped) is None
    assert transform(original[b'DXIL']) is None
    (tmp/'debug.bc').write_bytes(bitcode(debug))
    (tmp/'stripped.bc').write_bytes(bitcode(parts(stripped)[b'DXIL']))
    subprocess.run([str(build/'bin/opt'), '-strip-debug', '-S', str(tmp/'debug.bc'),
                    '-o', str(tmp/'expected.ll')], check=True)
    subprocess.run([str(build/'bin/llvm-dis'), str(tmp/'stripped.bc'),
                    '-o', str(tmp/'actual.ll')], check=True)
    def executable(text):
        # Debug names and metadata numbering may change during container serialization.
        text = re.sub(r'!\w+ !\d+', '', text)
        return '\n'.join(line.strip() for line in text.splitlines()
                         if not line.startswith((';', '!', 'source_filename')) and
                         'llvm.dbg.' not in line and line.strip())
    assert executable((tmp/'actual.ll').read_text()) == executable((tmp/'expected.ll').read_text())
    actual = (tmp/'actual.ll').read_text()
    assert '!dx.resources =' in actual and '!dx.entryPoints =' in actual
    assert '!llvm.dbg.' not in actual and '!DICompileUnit' not in actual
    print('PASS executable preservation, resource metadata, raw debug program, idempotence, no-debug input')
    result, size, error = C.c_void_p(123), C.c_size_t(123), C.create_string_buffer(256)
    assert strip(None, 0, C.byref(result), C.byref(size), error, len(error)) != 0
    assert not result.value and size.value == 0 and error.value
    assert strip(debug, len(debug), None, C.byref(size), error, len(error)) != 0
    print('PASS invalid input and output guards')
