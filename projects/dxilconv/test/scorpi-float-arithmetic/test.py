#!/usr/bin/env python3
"""Check opt-in arithmetic policy and unchanged preserve/raw/FP16/FP64 paths.

Usage: test.py <DXC build directory>; numerical semantics have native coverage.
"""
import ctypes as C
from pathlib import Path
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
    output, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(2048)
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
  output.Store4(0, asuint(float4(a + b, a * b, a / b, floor(a))));
  output.Store4(16, asuint(float4(ceil(a), frac(a), log2(a), rsqrt(a))));
  output.Store4(32, bits);
}
'''
with tempfile.TemporaryDirectory(prefix='scorpi-float-arithmetic-') as name:
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
        assert adapt(code, 0) is None
        assert adapt(code, 32) is None, 'comparison-only policy must not change arithmetic'
        result = adapt(code, 64)
        if mode == 'preserve':
            assert result is None, 'explicit preservation must remain unchanged'
        else:
            assert result is not None
            (tmp / 'adapted.dxil').write_bytes(result)
            subprocess.run([str(build / 'bin/dxc'), '-dumpbin', str(tmp / 'adapted.dxil')],
                           stdout=subprocess.DEVNULL, check=True)
        print('PASS', mode, 'policy and valid output module')

    for setup, values, flags in [
        ('uint a=bits.x,b=bits.y;', 'uint4(a+b,a*b,a&b,a^b)', ['-T', 'cs_6_0']),
        ('double a=asdouble(bits.x,bits.y),b=asdouble(bits.z,bits.w);',
         'uint4(a+b,a*b,a/b,a)', ['-T', 'cs_6_0']),
        ('float16_t a=float16_t(asfloat(bits.x)),b=float16_t(asfloat(bits.y));',
         'uint4(asuint16(a+b),asuint16(a*b),asuint16(a/b),0)',
         ['-T', 'cs_6_2', '-enable-16bit-types']),
        ('float a=asfloat(bits.x),b=asfloat(bits.y);', 'bits', ['-T', 'cs_6_0']),
    ]:
        text = '''cbuffer Inputs:register(b0){uint4 bits;};
RWByteAddressBuffer output:register(u0);
[numthreads(1,1,1)]void main(){ SETUP output.Store4(0,VALUES); }'''
        assert adapt(compile_shader(text.replace('SETUP', setup).replace('VALUES', values), flags), 64) is None
        print('PASS', setup.split()[0], 'unrelated operations unchanged')
