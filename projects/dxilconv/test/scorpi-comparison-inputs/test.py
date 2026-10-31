#!/usr/bin/env python3
"""Check opt-in FP32 comparison adaptation; GPU semantics have native coverage.

Usage: test.py <DXC build directory> (dxc and ScorpiDxbcToDxil targets built).
"""
import ctypes as C
from pathlib import Path
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


def adapt(code, options):
    output, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(1024)
    rc = lower(code, len(code), None, 0, 0xffffffff, 0, options, None, 0,
               C.byref(output), C.byref(size), error, len(error))
    assert rc == 0, (rc, error.value)
    try:
        return C.string_at(output, size.value) if output.value else None
    finally:
        lib.ScorpiFreeDxilBinary(output)


source = '''
cbuffer Inputs : register(b0) { uint4 bits; };
RWByteAddressBuffer output : register(u0);
[numthreads(1,1,1)] void main() {
  float a = asfloat(bits.x), b = asfloat(bits.y);
  output.Store4(0, uint4(a == b, a != b, a >= b, a < b));
  output.Store4(16, bits);
}
'''
with tempfile.TemporaryDirectory(prefix='scorpi-comparison-inputs-') as name:
    tmp = Path(name)
    def compile_shader(text, flags):
        (tmp / 'shader.hlsl').write_text(text)
        subprocess.run([str(build / 'bin/dxc'), *flags, '-E', 'main',
                        '-Fo', str(tmp / 'shader.dxil'), str(tmp / 'shader.hlsl')], check=True)
        return (tmp / 'shader.dxil').read_bytes()

    for mode, flags in [('default', ['-T', 'cs_6_0']),
                        ('ftz', ['-T', 'cs_6_2', '-denorm', 'ftz']),
                        ('preserve', ['-T', 'cs_6_2', '-denorm', 'preserve'])]:
        code = compile_shader(source, flags)
        assert adapt(code, 0) is None, 'disabled policy must leave code unchanged'
        result = adapt(code, 32)
        if mode == 'preserve':
            assert result is None, 'explicit preservation must remain unchanged'
        else:
            assert result is not None
            (tmp / 'adapted.dxil').write_bytes(result)
            ir = subprocess.check_output([str(build / 'bin/dxc'), '-dumpbin',
                                          str(tmp / 'adapted.dxil')], text=True)
            # Four comparisons share exactly two operand canonicalizations.
            assert len(re.findall(r'icmp ult i32 [^\n]+, 8388608', ir)) == 2, ir
            assert len(re.findall(r'fcmp ', ir)) == 4, ir
        print('PASS', mode, 'policy selection and shared comparison inputs')

    for scalar, setup, flags in [
        ('uint', 'uint a = bits.x, b = bits.y;', ['-T', 'cs_6_0']),
        ('double', 'double a = asdouble(bits.x, bits.y), b = asdouble(bits.z, bits.w);', ['-T', 'cs_6_0']),
        ('half', 'float16_t a = float16_t(asfloat(bits.x)), b = float16_t(asfloat(bits.y));',
         ['-T', 'cs_6_2', '-enable-16bit-types']),
    ]:
        text = source.replace('float a = asfloat(bits.x), b = asfloat(bits.y);', setup)
        assert adapt(compile_shader(text, flags), 32) is None, scalar
        print('PASS', scalar, 'comparisons unchanged')
    text = source.replace('uint4(a == b, a != b, a >= b, a < b)', 'bits')
    assert adapt(compile_shader(text, ['-T', 'cs_6_0']), 32) is None
    print('PASS shader without floating comparisons unchanged')
