#!/usr/bin/env python3
"""Reject unsupported root-buffer size queries before native compilation.

Usage: test.py <DXC build> [fixture output directory]
The same query must remain native for table bindings; ordinary root-buffer
loads must remain byte-for-byte unchanged by the adaptation pass.
"""
import ctypes as C
from pathlib import Path
import subprocess
import sys
import tempfile

build = Path(sys.argv[1]).resolve()
lib = C.CDLL(str(build / 'lib/libScorpiDxbcToDxil.dylib'))
lib.ScorpiFreeDxilBinary.argtypes = [C.c_void_p]

class Binding(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in
        ('resource_class', 'space', 'base_register', 'count',
         'root_offset', 'table_offset', 'feedback_offset')]

lower = lib.ScorpiLowerTypedBufferViewsWithSamplingPolicy
lower.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(Binding), C.c_size_t,
    C.c_uint32, C.c_uint32, C.c_uint32, C.c_void_p, C.c_size_t,
    C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.c_void_p, C.c_size_t]
lower.restype = C.c_int
dxc = build / 'bin/dxc'
if not dxc.exists(): dxc = build / 'bin/dxc-3.7'

def adapt(code, cls, table, reject):
    bindings = (Binding * 2)(Binding(cls, 7, 3, 1, 0, 0 if table else 0xffffffff, 64),
                            Binding(1, 1, 0, 1, 8, 0xffffffff, 96))
    binary, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(1024)
    status = lower(code, len(code), bindings, len(bindings), 100, 0, 8,
                   None, 0, C.byref(binary), C.byref(size), error, len(error))
    try:
        if reject:
            assert status == 4, (status, error.value)
            assert b'root-buffer GetDimensions' in error.value, error.value
            assert not binary.value and size.value == 0
        else:
            assert status == 0, (status, error.value)
            assert not error.value, error.value
            # No extra instructions, metadata reads or GPU branches are added.
            assert not binary.value and size.value == 0
    finally:
        if binary.value: lib.ScorpiFreeDxilBinary(binary)

with tempfile.TemporaryDirectory(prefix='scorpi-root-dimensions-') as directory:
    directory = Path(directory)
    output = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
    if output: output.mkdir(parents=True, exist_ok=True)
    for model in ['6_0', '6_6']:
        for structured in [False, True]:
            for writable in [False, True]:
                kind = 'StructuredBuffer<uint>' if structured else 'ByteAddressBuffer'
                kind = ('RW' if writable else '') + kind
                register = 'u3' if writable else 't3'
                for query in [False, True]:
                    name = f'{model}-{int(structured)}-{int(writable)}-{int(query)}'
                    operation = ('uint count, stride; source.GetDimensions(count, stride);'
                                 if structured else 'uint count; source.GetDimensions(count);') if query else (
                                     'uint count = source[0];' if structured else 'uint count = source.Load(0);')
                    source = directory / (name + '.hlsl')
                    source.write_text(f'{kind} source : register({register}, space7);\n'
                        'RWByteAddressBuffer output : register(u0, space1);\n'
                        '[numthreads(1,1,1)] void main() {\n' + operation + '\noutput.Store(0,count);\n}\n')
                    bytecode = directory / (name + '.dxil')
                    subprocess.run([str(dxc), '-T', 'cs_' + model, '-E', 'main',
                                    '-Fo', str(bytecode), str(source)], check=True)
                    code = bytecode.read_bytes()
                    adapt(code, int(writable), table=True, reject=False)
                    adapt(code, int(writable), table=False, reject=query)
                    if output:
                        (output / source.name).write_text(source.read_text())
                        (output / bytecode.name).write_bytes(code)
                    print('PASS', name, 'table query/load unchanged; root query rejected or load unchanged')
