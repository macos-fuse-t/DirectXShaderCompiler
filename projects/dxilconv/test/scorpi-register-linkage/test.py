#!/usr/bin/env python3
"""DDI register linkage: sparse producer/consumer masks, GS payload and API guards.
Usage: test.py <DXC build> [fixture output directory]
"""
import ctypes as C
from pathlib import Path
import struct
import re
import subprocess
import sys
import tempfile

build = Path(sys.argv[1]).resolve()
lib = C.CDLL(str(build / 'lib/libScorpiDxbcToDxil.dylib'))
class Element(C.Structure):
    _fields_ = [('semantic_name', C.c_char_p)] + [(s, C.c_uint32) for s in
        ('semantic_index', 'register_index', 'first_component', 'component_count',
         'component_type', 'output', 'flat_float')]
lib.ScorpiFreeDxilBinary.argtypes = [C.c_void_p]
convert = lib.ScorpiCreateDxilBinary
convert.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t,
    C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.c_void_p, C.c_size_t]
inspect = lib.ScorpiInspectRegisterLinkage
inspect.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t,
    C.POINTER(C.POINTER(Element)), C.POINTER(C.c_size_t), C.c_void_p, C.c_size_t]
link = lib.ScorpiLinkRegisterInputs
link.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(Element), C.c_size_t,
    C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.c_void_p, C.c_size_t]

def signature(inputs, outputs, flags=0):
    entries = lambda es: b''.join(struct.pack('<6I', *e) for e in es)
    return struct.pack('<10I', 2, flags, 24, len(inputs), len(outputs), 0,
        40, 40 + len(inputs) * 24, 0, 0) + entries(inputs) + entries(outputs)

def make(stage, mask):
    if stage == 'vs':
        w = [0x10050, 0, 0x0300005f, 0x001010f2, 0, 0x0300005f, 0x001010f2, 1,
             0x04000067, 0x001020f2, 0, 1,
             0x03000065, 0x00102002 | mask << 4, 1,
             0x05000036, 0x001020f2, 0, 0x00101e46, 0,
             0x05000036, 0x00102002 | mask << 4, 1, 0x00101e46, 1, 0x0100003e]
        sig = signature([(0,0,15,0,1,0),(0,1,15,0,1,0)],
                        [(1,0,15,0,3,0),(0,1,mask,0,1,0)])
    else:
        w = ([0x20050, 0, 0x0100085d, 0x0100085c, 0x0200005e, 1,
              0x0400005f, 0x00201002 | mask << 4, 1, 1] if stage == 'gs' else
             [0x50, 0, 0x03000862, 0x00101002 | mask << 4, 1])
        w += [0x03000065, 0x001020f2, 0, 0x08000036, 0x001020f2, 0,
              0x00004002, 0, 0, 0, 0]
        for c in range(4):
            if mask & (1 << c):
                w += ([0x06000036, 0x00102002 | 1 << (4+c), 0,
                       0x0020100a | c << 4, 0, 1] if stage == 'gs' else
                      [0x05000036, 0x00102002 | 1 << (4+c), 0,
                       0x0010100a | c << 4, 1])
        if stage == 'gs': w += [0x01000013, 0x01000009]
        w += [0x0100003e]
        sig = signature([(0,1,mask,0,1,0)], [(0,0,15,0,1,0)], int(stage == 'gs'))
    w[1] = len(w)
    raw = struct.pack('<%dI' % len(w), *w)
    ptr, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(2048)
    assert convert(raw,len(raw),sig,len(sig),C.byref(ptr),C.byref(size),error,len(error)) == 0, error.value
    try: return C.string_at(ptr,size.value)
    finally: lib.ScorpiFreeDxilBinary(ptr)

def reflect(code, output):
    ptr, size, error = C.POINTER(Element)(), C.c_size_t(), C.create_string_buffer(2048)
    assert inspect(code,len(code),None,0,C.byref(ptr),C.byref(size),error,len(error)) == 0, error.value
    try:
        # Own the semantic strings after releasing the C allocation.
        return [Element(bytes(e.semantic_name),e.semantic_index,e.register_index,e.first_component,
            e.component_count,e.component_type,e.output,e.flat_float)
            for e in ptr[:size.value] if bool(e.output & 1) == output]
    finally: lib.ScorpiFreeDxilBinary(ptr)

def remap(code, upstream, fail=False):
    layout = (Element * len(upstream))(*upstream)
    ptr, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(2048)
    rc = link(code,len(code),layout,len(layout),C.byref(ptr),C.byref(size),error,len(error))
    if fail:
        assert rc != 0 and not ptr.value and size.value == 0, (rc,error.value)
        return
    assert rc == 0 and ptr.value, (rc,error.value)
    try: return C.string_at(ptr,size.value)
    finally: lib.ScorpiFreeDxilBinary(ptr)

def shape(elements):
    return [(e.semantic_name,e.semantic_index,e.register_index,e.first_component,
        e.component_count,e.component_type,bool(e.output & 2)) for e in elements]

vs = {m: make('vs',m) for m in range(1,16)}
consumers = {stage: {m: make(stage,m) for m in range(1,16)} for stage in ('ps','gs')}
cases = 0
for produced, code in vs.items():
    upstream = reflect(code,True)
    assert upstream[0].semantic_name.lower() == b'sv_position'
    for stage, shaders in consumers.items():
        for consumed, consumer in shaders.items():
            fixed = remap(consumer,upstream)
            inputs = reflect(fixed,False)
            assert shape(reflect(fixed,True)) == shape(reflect(consumer,True))
            expected = upstream if stage == 'gs' else [e for e in upstream
                if not e.output & 2 and ((1 << e.component_count)-1) << e.first_component & consumed]
            assert shape(inputs) == shape(expected), (stage,produced,consumed,shape(inputs),shape(expected))
            cases += 1
# Empty producers leave only intrinsic system inputs; undefined user reads are not fabricated.
assert not reflect(remap(consumers['ps'][15],[]),False)
remap(vs[15],reflect(vs[15],True),fail=True)
invalid = Element(b'SCORPI',0,0,0,4,1,2,0)  # system INPUT is not an upstream output
remap(consumers['ps'][15],[invalid],fail=True)
layout = reflect(vs[15],True)
remap(consumers['gs'][15],layout + layout,fail=True)
# Keep rectangular dynamic arrays as one input load; preserve GS vertex-axis
# and system inputs while reshaping user semantics to an independently named VS.
dxc = next(p for p in (build / 'bin/dxc', build / 'bin/dxc-3.7') if p.exists())
with tempfile.TemporaryDirectory(prefix='scorpi-register-arrays-') as name:
    tmp = Path(name)
    source = tmp / 'array.hlsl'
    source.write_text("""
struct Input { float4 position : SV_Position; uint4 value[2] : TEXCOORD; };
struct Output { float4 position : SV_Position; uint4 value : RESULT; };
[maxvertexcount(1)]
void GS(point Input input[1], uint primitive : SV_PrimitiveID,
        inout PointStream<Output> stream) {
    Output output;
    output.position = input[0].position;
    output.value = input[0].value[primitive & 1];
    stream.Append(output);
}
""")
    original_file, fixed_file = tmp / 'array.dxil', tmp / 'linked.dxil'
    subprocess.run([str(dxc),'-T','gs_6_0','-E','GS','-Fo',str(original_file),str(source)],check=True)
    original = original_file.read_bytes()
    upstream = reflect(original,False)
    upstream = [e for e in upstream if e.semantic_name.lower() != b'sv_primitiveid']
    for e in upstream:
        e.output |= 1
        if not e.output & 2: e.semantic_name = b'PRODUCER'
    fixed_file.write_bytes(remap(original,upstream))
    old_ir = subprocess.check_output([str(dxc),'-dumpbin',str(original_file)],text=True)
    fixed_ir = subprocess.check_output([str(dxc),'-dumpbin',str(fixed_file)],text=True)
    loads = lambda ir: re.findall(r'call (?:i32|float) @dx.op.loadInput[^\n]+',ir)
    assert len(loads(old_ir)) == len(loads(fixed_ir)) == 8
    assert any('i32 %' in line for line in loads(fixed_ir)), 'dynamic row index expanded'
    assert 'SV_PrimitiveID' in fixed_ir and 'PRODUCER' in fixed_ir
if len(sys.argv) > 2:
    output = Path(sys.argv[2]); output.mkdir(parents=True,exist_ok=True)
    for m,code in vs.items(): (output / f'vs-{m}.dxil').write_bytes(code)
    for stage,shaders in consumers.items():
        for m,code in shaders.items(): (output / f'{stage}-{m}.dxil').write_bytes(code)
print('PASS:',cases,'producer/consumer mask conversions; complete GS payload, preserved outputs and API guards')
