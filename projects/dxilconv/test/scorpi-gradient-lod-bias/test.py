#!/usr/bin/env python3
"""Check explicit-gradient bias lowering and unchanged ordinary shader paths.

Usage: test.py <DXC build directory>
"""
import ctypes as C
from pathlib import Path
import subprocess
import sys
import tempfile

build = Path(sys.argv[1]).resolve()
lib = C.CDLL(str(build / 'lib/libScorpiDxbcToDxil.dylib'))

class Binding(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('resource_class', 'space',
        'base_register', 'count', 'root_offset', 'table_offset', 'feedback_offset')]

class Requirement(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('resource_class', 'space',
        'base_register', 'count', 'requirements')]

lower = lib.ScorpiLowerTypedBufferViewsWithSamplingPolicy
lower.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(Binding), C.c_size_t,
    C.c_uint32, C.c_uint32, C.c_uint32, C.POINTER(Requirement), C.c_size_t,
    C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.c_void_p, C.c_size_t]
lib.ScorpiFreeDxilBinary.argtypes = [C.c_void_p]
bindings = (Binding * 2)(Binding(0, 0, 0, 1, 0, 0, 32),
                         Binding(3, 0, 0, 2, 8, 0, 36))
with tempfile.TemporaryDirectory(prefix='scorpi-gradient-bias-') as folder:
    folder = Path(folder)
    for model in ['6_0', '6_6']:
        for stage in ['cs', 'ps']:
            for shape, coord, grad in [('1D', '.5', '.25'),
                ('2D', 'float2(.5,.5)', 'float2(.25,0)'),
                ('3D', 'float3(.5,.5,.5)', 'float3(.25,0,0)'),
                ('Cube', 'float3(0,0,1)', 'float3(.25,0,0)')]:
                for operation in ['SampleLevel(s[0],COORD,1)',
                                  'SampleGrad(s[0],COORD,GRAD,GRAD)']:
                    operation = operation.replace('COORD', coord).replace('GRAD', grad)
                    body = 'return image.' + operation + ';'
                    output = ''
                    entry = 'float4 main():SV_Target'
                    if stage == 'cs':
                        output = 'RWStructuredBuffer<float4> output:register(u0);'
                        entry = '[numthreads(1,1,1)] void main()'
                        body = 'output[0]=image.' + operation + ';'
                    source = folder / 'test.hlsl'
                    source.write_text(f'Texture{shape}<float4> image:register(t0);\n'
                        f'SamplerState s[2]:register(s0);\n{output}\n{entry}{{{body}}}\n')
                    original = folder / 'original.dxil'
                    subprocess.run([str(build/'bin/dxc'), '-T', stage+'_'+model,
                        '-E', 'main', '-Fo', str(original), str(source)], check=True)
                    code = original.read_bytes()
                    for reg, flags in [(0, 0), (1, 256), (0, 256)]:
                        policy = (Requirement * 1)(Requirement(3, 0, reg, 1, flags))
                        binary, size = C.c_void_p(), C.c_size_t()
                        error = C.create_string_buffer(4096)
                        status = lower(code, len(code), bindings, 2, 99, 16,
                            1 | 2 | 4 | 8, policy, 1, C.byref(binary), C.byref(size),
                            error, len(error))
                        assert status == 0 and not error.value, (status, error.value)
                        expected = reg == 0 and flags == 256 and 'SampleGrad' in operation
                        try:
                            assert bool(binary.value) == expected
                            if expected:
                                lowered = folder/'lowered.dxil'
                                lowered.write_bytes(C.string_at(binary, size.value))
                                ir = subprocess.check_output([str(build/'bin/dxc'),
                                    '-dumpbin', str(lowered)], text=True)
                                assert 'dx.op.sampleGrad' in ir and 'Exp' in ir
                                assert 'dx.op.textureLoad' not in ir
                        finally:
                            if binary.value: lib.ScorpiFreeDxilBinary(binary)
                    print('PASS', stage+'_'+model, shape, operation.split('(')[0],
                          'only the used biased gradient sampler is lowered')
