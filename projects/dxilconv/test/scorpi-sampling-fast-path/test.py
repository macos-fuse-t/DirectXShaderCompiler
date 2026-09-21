#!/usr/bin/env python3
"""Exercise production lowering and binding-proven, unmodified native sampling."""
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
native = lib.ScorpiLowerTypedBufferViewsWithOptions
native.argtypes = lower.argtypes[:6] + [C.c_uint32] + lower.argtypes[6:]
lib.ScorpiFreeDxilBinary.argtypes = [C.c_void_p]
bindings = (Binding * 2)(Binding(*(0, 0, 0, 1, 0, 0, 32)[:len(fields)]),
                         Binding(*(3, 0, 0, 1, 8, 0, 36)[:len(fields)]))
with tempfile.TemporaryDirectory(prefix='scorpi-sampling-native-') as folder:
    folder = Path(folder)
    for model in ('6_0', '6_6'):
        for cube in (False, True):
            operations = ['Sample', 'SampleLevel', 'SampleGrad', 'SampleBias']
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
                if operation == 'SampleBias':
                    extra = ', .25'
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
                # Pure texture shaders must be a true no-op: no metadata root,
                # new branches, software loops, or even serialized replacement.
                for policy in (1, 3):
                    status = native(code, len(code), None, 0, 0xffffffff, 0, policy,
                                    C.byref(output), C.byref(size), error, len(error))
                    assert status == 0 and not output.value and size.value == 0, error.value
                if operation == 'SampleLevel':
                    status = native(code, len(code), bindings, len(bindings), 99, 16, 2,
                                    C.byref(output), C.byref(size), error, len(error))
                    assert status == 0 and output.value, error.value
                    (folder / 'no-minmax.dxil').write_bytes(C.string_at(output, size.value))
                    lib.ScorpiFreeDxilBinary(output)
                    no_minmax = subprocess.check_output([dxc, '-dumpbin', str(folder / 'no-minmax.dxil')], text=True)
                    assert not re.search(r'and i32 [^\n]*, (256|384)\b', no_minmax), 'min/max discriminator survived opt-out'
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
                # Gather without a feedback consumer may eliminate the entire
                # software status loop; it must still retain its native sample.
                if not operation.startswith('Gather'):
                    assert prefix + '.reduce' in body, (model, cube, operation)
                    assert 'scorpi.tap.loop' in body, (model, cube, operation)
                    assert 'llvm.loop.unroll.disable' in ir
                    assert len(body.splitlines()) < (2600 if cube else 1400), (model, cube, operation)
                entry = re.split(r'^[-a-zA-Z0-9_.]+:', body, flags=re.M)[0]
                # Derivative ops 83/84 stay quad-uniform; sqrt/log and dimension
                # queries belonging to fallback footprints must not precede it.
                assert not re.search(r'call.*@dx.op.unary.f32\(i32 (23|24)', entry), operation
                print('PASS', model, 'cube' if cube else '2D', operation,
                      'compat IR lines', len(body.splitlines()), 'DXIL bytes', size.value,
                      'native original bytes', len(code), 'native added instructions 0')

        # A binding hint must not suppress application-visible residency status.
        source = '''Texture2D<float4> t:register(t0); SamplerState s:register(s0);
float4 main(float4 p:SV_Position):SV_Target {
 uint status; float4 v=t.SampleLevel(s,p.xy,0,int2(0,0),status);
 return CheckAccessFullyMapped(status) ? v : 0;
}'''
        (folder / 'input.hlsl').write_text(source)
        subprocess.run([dxc, '-T', 'ps_' + model, '-E', 'main', '-Fo',
                        str(folder / 'input.dxil'), str(folder / 'input.hlsl')], check=True)
        code = (folder / 'input.dxil').read_bytes()
        status = native(code, len(code), bindings, len(bindings), 99, 16, 1,
                        C.byref(output), C.byref(size), error, len(error))
        assert status == 0 and output.value and size.value, error.value
        (folder / 'native-feedback.dxil').write_bytes(C.string_at(output, size.value))
        native_ir = subprocess.check_output([dxc, '-dumpbin', str(folder / 'native-feedback.dxil')], text=True)
        assert 'scorpi.tap.loop' not in native_ir, 'fully resident native sample acquired a software filter'
        native_body = native_ir.split('define void @main()', 1)[1].split('\n}', 1)[0]
        lib.ScorpiFreeDxilBinary(output)
        print('PASS', model, 'resident feedback uses native sampling, IR lines', len(native_body.splitlines()))
        # Explicit/dynamic instruction clamps need the bounded emulation path,
        # even with ordinary descriptors and no residency-status consumer.
        source = '''Texture2D<float4> t:register(t0); SamplerState s:register(s0);
float4 main(float4 p:SV_Position):SV_Target {
 return t.Sample(s,p.xy,int2(0,0),p.z);
}'''
        (folder / 'input.hlsl').write_text(source)
        subprocess.run([dxc, '-T', 'ps_' + model, '-E', 'main', '-Fo',
                        str(folder / 'input.dxil'), str(folder / 'input.hlsl')], check=True)
        code = (folder / 'input.dxil').read_bytes()
        status = native(code, len(code), bindings, len(bindings), 99, 16, 1,
                        C.byref(output), C.byref(size), error, len(error))
        assert status == 0 and output.value and size.value, error.value
        lib.ScorpiFreeDxilBinary(output)
        print('PASS', model, 'instruction clamp retains compatibility lowering')

        # A clamp on t0 must not expand independent ordinary accesses to t1.
        # Deliberately supply metadata only for t0: t1 stays entirely native.
        for cube in (False, True):
            for clamp in ('0.75', 'p.z'):
                texture = 'TextureCube' if cube else 'Texture2D'
                coord = 'float3(p.xy, 1)' if cube else 'p.xy'
                source = (f'{texture}<float4> t:register(t0);\n'
                          f'{texture}<float4> ordinary:register(t1);\n'
                          'SamplerState s:register(s0);\n'
                          'float4 main(float4 p:SV_Position):SV_Target {\n'
                          f'float4 a=t.Sample(s,{coord},' +
                          ('' if cube else 'int2(0,0),') + f'{clamp});\n'
                          f'return a + ordinary.SampleLevel(s,{coord},0); }}\n')
                (folder / 'input.hlsl').write_text(source)
                subprocess.run([dxc, '-T', 'ps_' + model, '-E', 'main', '-Fo',
                                str(folder / 'input.dxil'), str(folder / 'input.hlsl')], check=True)
                code = (folder / 'input.dxil').read_bytes()
                status = native(code, len(code), bindings, len(bindings), 99, 16, 3,
                                C.byref(output), C.byref(size), error, len(error))
                assert status == 0 and output.value, error.value
                (folder / 'mixed-clamp.dxil').write_bytes(C.string_at(output, size.value))
                lib.ScorpiFreeDxilBinary(output)
                ir = subprocess.check_output([dxc, '-dumpbin', str(folder / 'mixed-clamp.dxil')], text=True)
                assert not re.search(r'and i32 [^\n]*, (256|384)\b', ir)
                assert 'scorpi.tap.loop' not in ir, 'native instruction clamp acquired a software filter'
                assert not re.search(r'call.*@dx.op.unary.f32\(i32 (23|24)', ir), 'software footprint algebra survived native proof'
                body = ir.split('define void @main()', 1)[1].split('\n}', 1)[0]
                print('PASS', model, texture, clamp, 'clamp does not expand independent sample; IR lines', len(body.splitlines()))

        # Likewise, consuming t0 residency must not expand t1 sampling/loads.
        for operation in ('SampleLevel(s,p.xy,0)', 'Load(int3(p.xy,0))'):
            source = '''Texture2D<float4> t:register(t0);
Texture2D<float4> ordinary:register(t1); SamplerState s:register(s0);
float4 main(float4 p:SV_Position):SV_Target {
 uint status; float4 v=t.SampleLevel(s,p.xy,0,int2(0,0),status);
 return (CheckAccessFullyMapped(status) ? v : 0) + ordinary.''' + operation + '; }'
            (folder / 'input.hlsl').write_text(source)
            subprocess.run([dxc, '-T', 'ps_' + model, '-E', 'main', '-Fo',
                            str(folder / 'input.dxil'), str(folder / 'input.hlsl')], check=True)
            code = (folder / 'input.dxil').read_bytes()
            status = native(code, len(code), bindings, len(bindings), 99, 16, 3,
                            C.byref(output), C.byref(size), error, len(error))
            assert status == 0 and output.value, error.value
            lib.ScorpiFreeDxilBinary(output)
            print('PASS', model, operation, 'residency consumer does not expand independent access')
