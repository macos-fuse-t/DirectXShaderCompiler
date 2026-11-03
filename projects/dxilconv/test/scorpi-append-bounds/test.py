#!/usr/bin/env python3
"""Validate append-store bounds without changing counters or non-append shaders.

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
lib.ScorpiFreeDxilBinary.argtypes = [C.c_void_p]
cases = [
    ('append', 'AppendStructuredBuffer<uint4>', 'data.Append(uint4(id.x,2,3,4));', True),
    ('increment', 'RWStructuredBuffer<uint4>', 'data[data.IncrementCounter()]=uint4(id.x,2,3,4);', True),
    ('consume', 'ConsumeStructuredBuffer<uint4>', 'output[id.x]=data.Consume();', False),
    ('ordinary', 'RWStructuredBuffer<uint4>', 'data[id.x]=uint4(id.x,2,3,4);', False),
]
with tempfile.TemporaryDirectory(prefix='scorpi-append-bounds-') as folder:
    folder = Path(folder)
    validator = folder / 'validate'
    subprocess.run(['clang++', '-std=c++17', '-I' + str(here.parents[3] / 'include'),
                    str(here.parent / 'scorpi-descriptor-metadata-arrays/validate.cpp'),
                    '-o', str(validator)], check=True)
    for model in ('6_0', '6_6'):
        for name, resource, body, changed in cases:
            source, original, rewritten = [folder / (name + suffix)
                                          for suffix in ('.hlsl', '.dxil', '.lowered.dxil')]
            source.write_text(resource + ' data:register(u0);\n'
                              'RWStructuredBuffer<uint4> output:register(u1);\n'
                              '[numthreads(1,1,1)] void Main(uint3 id:SV_DispatchThreadID){' + body + '}')
            subprocess.run([str(build / 'bin/dxc'), '-T', 'cs_' + model, '-E', 'Main',
                            '-Fo', str(original), str(source)], check=True)
            code = original.read_bytes()
            binary, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(4096)
            status = lower(code, len(code), None, 0, 0xffffffff, 0, 4, None, 0,
                           C.byref(binary), C.byref(size), error, len(error))
            assert status == 0, (status, error.value)
            assert bool(binary.value) == changed, (model, name)
            if not changed:
                print('PASS', model, name, 'unchanged')
                continue
            rewritten.write_bytes(C.string_at(binary, size.value))
            lib.ScorpiFreeDxilBinary(binary)
            subprocess.run([str(validator), str(build / 'lib/libdxcompiler.dylib'),
                            str(rewritten)], check=True)
            before = subprocess.check_output([str(build / 'bin/dxc'), '-dumpbin', str(original)], text=True)
            after = subprocess.check_output([str(build / 'bin/dxc'), '-dumpbin', str(rewritten)], text=True)
            counters = lambda ir: re.findall(r'call i32 @dx\.op\.bufferUpdateCounter[^\n]*i8 (-?\d+)\)', ir)
            assert counters(before) == counters(after), (model, name, 'counter changed')
            assert 'icmp ult i32' in after and 'br i1' in after, after
            assert after.index('call i32 @dx.op.bufferUpdateCounter') < after.index('br i1'), after
            assert 'ScorpiTypedBufferMetadata' not in after, after
            print('PASS', model, name, 'validated; store guarded; counter unconditional; no metadata')
