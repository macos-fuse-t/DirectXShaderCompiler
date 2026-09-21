#!/usr/bin/env python3
"""Exercise production lowering; ordinary sampling retains a native branch."""
import ctypes as C
from pathlib import Path
import re
import subprocess
import sys
import tempfile

build = Path(sys.argv[1]).resolve()
lib = C.CDLL(str(build / 'lib/libScorpiDxbcToDxil.dylib'))
dxc = str(build / 'bin/dxc')

fields = [
        'resource_class', 'space', 'base_register', 'count',
        'root_offset', 'table_offset']
# Also usable alongside the pending buffer-feedback ABI extension.
header = Path(__file__).resolve().parents[2] / 'include/ScorpiDxbcToDxil.h'
if 'uint32_t feedback_offset;' in header.read_text():
    fields.append('feedback_offset')
class Binding(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in fields]

lower = lib.ScorpiLowerTypedBufferViews
lower.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(Binding), C.c_size_t,
                  C.c_uint32, C.c_uint32, C.POINTER(C.c_void_p),
                  C.POINTER(C.c_size_t), C.c_char_p, C.c_size_t]
lib.ScorpiFreeDxilBinary.argtypes = [C.c_void_p]
bindings = (Binding * 2)(Binding(*(0, 0, 0, 1, 0, 0, 32)[:len(fields)]),
                         Binding(*(3, 0, 0, 1, 8, 0, 36)[:len(fields)]))
with tempfile.TemporaryDirectory(prefix='scorpi-sampling-native-') as folder:
    folder = Path(folder)
    for model in ('6_0', '6_6'):
        for cube in (False, True):
            operations = ['Sample', 'SampleLevel', 'SampleGrad']
            if '--comparison-gather' in sys.argv:
                operations += ['SampleCmp', 'SampleCmpLevelZero', 'Gather', 'GatherCmp']
            for operation in operations:
                comparison = 'Cmp' in operation
                coord = 'float3(p.xy * .01, 1)' if cube else 'p.xy * .01'
                extra = ', .5' if comparison else ''
                if operation == 'SampleGrad':
                    extra = ', float3(.01,0,0), float3(0,.01,0)' if cube else ', float2(.01,0), float2(0,.01)'
                if operation == 'SampleLevel':
                    # No arithmetic before the sample in this basic block:
                    # the instruction immediately follows the branch label.
                    coord = 'float3(.3,.4,1)' if cube else 'float2(.3,.4)'
                    extra = ', 0'
                branch = '[branch] if(p.x > 0) ' if operation == 'SampleLevel' else ''
                fallback = 'return 0;' if branch else ''
                source = (f'Texture{"Cube" if cube else "2D"}<float> t:register(t0);\n'
                          f'Sampler{"Comparison" if comparison else ""}State s:register(s0);\n'
                          f'float4 main(float4 p:SV_Position):SV_Target {{ {branch}return t.{operation}(s,{coord}{extra}); {fallback} }}\n')
                (folder / 'input.hlsl').write_text(source)
                subprocess.run([dxc, '-T', 'ps_' + model, '-E', 'main', '-Fo',
                                str(folder / 'input.dxil'), str(folder / 'input.hlsl')], check=True)
                code = (folder / 'input.dxil').read_bytes()
                output, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(4096)
                status = lower(code, len(code), bindings, len(bindings), 99, 16,
                               C.byref(output), C.byref(size), error, len(error))
                assert status == 0, error.value
                try:
                    (folder / 'output.dxil').write_bytes(C.string_at(output, size.value))
                finally:
                    lib.ScorpiFreeDxilBinary(output)
                ir = subprocess.check_output([dxc, '-dumpbin', str(folder / 'output.dxil')], text=True)
                body = ir.split('define void @main()', 1)[1].split('\n}', 1)[0]
                prefix = 'scorpi.cube' if cube else 'scorpi.sample'
                # Both paths survive optimization. In particular a comparison
                # or gather must not unconditionally enter software filtering.
                assert prefix + '.native' in body, (model, cube, operation)
                assert prefix + '.reduce' in body, (model, cube, operation)
                entry = re.split(r'^[-a-zA-Z0-9_.]+:', body, flags=re.M)[0]
                # Derivative ops 83/84 stay quad-uniform; sqrt/log and dimension
                # queries belonging to fallback footprints must not precede it.
                assert not re.search(r'call.*@dx.op.unary.f32\(i32 (23|24)', entry), operation
                print('PASS', model, 'cube' if cube else '2D', operation)
