#!/usr/bin/env python3
"""Validate dynamic descriptor-metadata indexing through the DXC validator.

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
here = Path(__file__).resolve().parent
lib = C.CDLL(str(build / 'lib/libScorpiDxbcToDxil.dylib'))

class Binding(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in
                ('resource_class', 'space', 'base_register', 'count',
                 'root_offset', 'table_offset', 'feedback_offset')]

class Requirement(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in
                ('resource_class', 'space', 'base_register', 'count', 'requirements')]

lower = lib.ScorpiLowerTypedBufferViewsWithSamplingPolicy
lower.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(Binding), C.c_size_t,
                  C.c_uint32, C.c_uint32, C.c_uint32, C.POINTER(Requirement),
                  C.c_size_t, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t),
                  C.c_void_p, C.c_size_t]
lower.restype = C.c_int
lib.ScorpiFreeDxilBinary.argtypes = [C.c_void_p]
strip = lib.ScorpiStripShaderDebugInfo
strip.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(C.c_void_p),
                  C.POINTER(C.c_size_t), C.c_void_p, C.c_size_t]

def flags(path):
    ir = subprocess.check_output([str(build / 'bin/dxc'), '-dumpbin', str(path)], text=True)
    nodes = dict(re.findall(r'^(!\d+) = (.*)$', ir, re.M))
    entry = re.search(r'!dx.entryPoints = !\{(!\d+)\}', ir)[1]
    properties = re.findall(r'!\d+', nodes[entry])[-1]
    value = re.search(r'i32 0, i64 (\d+)', nodes[properties])
    return int(value[1]) if value else 0

with tempfile.TemporaryDirectory(prefix='scorpi-metadata-array-') as folder:
    folder = Path(folder)
    validator = folder / 'validate'
    subprocess.run(['clang++', '-std=c++17', '-I' + str(here.parents[3] / 'include'),
                    str(here / 'validate.cpp'), '-o', str(validator)], check=True)
    source = folder / 'input.hlsl'
    resources = 'Texture2D<float4> input[4]:register(t0);\nSamplerState s[4]:register(s0);\n'
    bindings = (Binding * 2)(Binding(0, 0, 0, 4, 0, 0, 32), Binding(3, 0, 0, 4, 8, 0, 36))
    policy = (Requirement * 1)(Requirement(3, 0, 0, 4, 4))
    for stage, model in (('cs', '6_0'), ('cs', '6_6'), ('ps', '6_0'), ('ps', '6_6')):
        body = ('RWStructuredBuffer<float4> output:register(u0);\n'
                '[numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID) {\n'
                'uint i=id.x&3; output[id.x]=input[i].SampleLevel(s[i],float2(id.xy),0); }') if stage == 'cs' else (
                '[earlydepthstencil] float4 main(float4 p:SV_Position):SV_Target {\n'
                'uint i=(uint)p.x&3; return input[i].SampleLevel(s[i],p.xy,0); }')
        source.write_text(resources + body)
        original = folder / 'original.dxil'
        subprocess.run([str(build / 'bin/dxc'), '-T', stage + '_' + model, '-E', 'main',
                        '-Zi', '-Qembed_debug', '-Gis', '-Fo', str(original), str(source)], check=True, cwd=folder)
        original_flags = flags(original)
        code = original.read_bytes()
        binary, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(4096)
        assert strip(code, len(code), C.byref(binary), C.byref(size), error, len(error)) == 0, error.value
        if binary.value:
            code = C.string_at(binary, size.value)
            lib.ScorpiFreeDxilBinary(binary)
        # Model instrumentation/debug stripping leaving an unused global and
        # a derived feature bit after its last instruction has disappeared.
        execution = folder / 'execution.dxil'
        execution.write_bytes(code)
        ir = subprocess.check_output([str(build / 'bin/dxc'), '-dumpbin', str(execution)], text=True)
        ir = ir.replace('define ', '@unused.debug = internal constant [1 x i32] zeroinitializer\n\ndefine ', 1)
        assert not original_flags & 4096  # Typed UAV load additional formats.
        ir, count = re.subn(r'(i32 0, i64 )' + str(original_flags) + r'(?=[,}])',
                            lambda m: m[1] + str(original_flags | 4096), ir)
        assert count == 1
        assembly = folder / 'instrumented.ll'
        assembly.write_text(ir)
        bitcode = folder / 'instrumented.bc'
        subprocess.run([str(build / 'bin/llvm-as'), str(assembly), '-o', str(bitcode)], check=True)
        for index in range(struct.unpack_from('<I', code, 28)[0]):
            at = struct.unpack_from('<I', code, 32 + 4 * index)[0]
            if code[at:at+4] == b'DXIL':
                program = bytearray(code[at+8:at+32])
                break
        else:
            raise AssertionError('Missing executable DXIL')
        bc = bitcode.read_bytes()
        program.extend(bc)
        program.extend(bytes((-len(program)) % 4))
        struct.pack_into('<I', program, 4, len(program) // 4)
        struct.pack_into('<I', program, 20, len(bc))
        code = (struct.pack('<4s16sHHIII', b'DXBC', bytes(16), 1, 0,
                            44 + len(program), 1, 36) +
                struct.pack('<4sI', b'DXIL', len(program)) + program)
        binary, size = C.c_void_p(), C.c_size_t()
        status = lower(code, len(code), bindings, 2, 99, 16, 4, policy, 1,
                       C.byref(binary), C.byref(size), error, len(error))
        assert status == 0 and binary.value, (status, error.value)
        lowered = folder / 'lowered.dxil'
        lowered.write_bytes(C.string_at(binary, size.value))
        lib.ScorpiFreeDxilBinary(binary)
        subprocess.run([str(validator), str(build / 'lib/libdxcompiler.dylib'), str(lowered)], check=True)
        assert original_flags and flags(lowered) & original_flags == original_flags
        assert not flags(lowered) & 4096
        print('PASS', stage + '_' + model, 'dynamic metadata indices validate; strict/early-depth modes preserved')
