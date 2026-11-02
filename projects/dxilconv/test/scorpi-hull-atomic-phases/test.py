#!/usr/bin/env python3
"""Check hull CAS phase inlining, DXIL validity, and unchanged atomic operations.

Usage: test.py <DXC build directory>
"""
import ctypes as C
from pathlib import Path
import re
import subprocess
import sys
import tempfile

build = Path(sys.argv[1]).resolve()
here = Path(__file__).resolve().parent
lib = C.CDLL(str(build / 'lib/libScorpiDxbcToDxil.dylib'))
lower = lib.ScorpiLowerTypedBufferViewsWithSamplingPolicy
lower.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t,
                  C.c_uint32, C.c_uint32, C.c_uint32, C.c_void_p,
                  C.c_size_t, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t),
                  C.c_void_p, C.c_size_t]
lower.restype = C.c_int
lib.ScorpiFreeDxilBinary.argtypes = [C.c_void_p]

source = '''RWByteAddressBuffer output : register(u0);
struct CP { float4 p:SV_Position; };
struct PC { float edge[4]:SV_TessFactor; float inside[2]:SV_InsideTessFactor; };
PC Constants(InputPatch<CP,4> patch,uint id:SV_PrimitiveID) {
  PATCH
  PC o; [unroll] for(uint i=0;i<4;i++)o.edge[i]=1;
  o.inside[0]=1;o.inside[1]=1;return o;
}
[domain("quad")][partitioning("integer")][outputtopology("triangle_cw")]
[outputcontrolpoints(4)][patchconstantfunc("Constants")][maxtessfactor(1.0)]
CP HS(InputPatch<CP,4> patch,uint i:SV_OutputControlPointID,uint id:SV_PrimitiveID) {
  CONTROL
  return patch[i];
}
'''
cas = 'uint old;output.InterlockedCompareExchange(0,0,id,old);'
add = 'uint old;output.InterlockedAdd(0,id,old);'
cases = [('none', '', '', False), ('patch', cas, '', False),
         ('control', '', cas, False), ('both-add', add, add, False),
         ('both-unused', cas, cas, True),
         ('both-used', cas + 'output.Store(4,old);',
          cas + 'output.Store(8+i*4,old);', True)]

def lowered(code):
    binary, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(4096)
    status = lower(code, len(code), None, 0, 0xffffffff, 0, 4, None, 0,
                   C.byref(binary), C.byref(size), error, len(error))
    assert status == 0, (status, error.value)
    if not binary.value:
        return None
    result = C.string_at(binary, size.value)
    lib.ScorpiFreeDxilBinary(binary)
    return result

with tempfile.TemporaryDirectory(prefix='scorpi-hull-atomics-') as directory:
    directory = Path(directory)
    validator = directory / 'validate'
    subprocess.run(['clang++', '-std=c++17', '-I' + str(here.parents[3] / 'include'),
                    str(here.parent / 'scorpi-descriptor-metadata-arrays/validate.cpp'),
                    '-o', str(validator)], check=True)
    for name, patch, control, changed in cases:
        hlsl, original, output = [directory / (name + suffix)
                                  for suffix in ('.hlsl', '.dxil', '.lowered.dxil')]
        hlsl.write_text(source.replace('PATCH', patch).replace('CONTROL', control))
        subprocess.run([str(build / 'bin/dxc'), '-T', 'hs_6_0', '-E', 'HS',
                        '-Fo', str(original), str(hlsl)], check=True)
        code = original.read_bytes()
        result = lowered(code)
        assert (result is not None) == changed, name
        output.write_bytes(result or code)
        subprocess.run([str(validator), str(build / 'lib/libdxcompiler.dylib'),
                        str(output)], check=True)
        before = subprocess.check_output([str(build / 'bin/dxc'), '-dumpbin', str(original)], text=True)
        after = subprocess.check_output([str(build / 'bin/dxc'), '-dumpbin', str(output)], text=True)
        assert ('alwaysinline' in after) == changed, name
        atomics = lambda ir: re.findall(r'call i32 @dx\.op\.atomic\w+\.i32\([^\n]+', ir)
        assert atomics(before) == atomics(after), name
        assert lowered(result or code) is None, (name, 'not idempotent')
        print('PASS', name, 'validated; atomic operands preserved; phase scope and idempotence checked')
