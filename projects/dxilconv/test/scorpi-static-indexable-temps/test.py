#!/usr/bin/env python3
"""Check SSA promotion of constant-indexed DXBC temporaries before MSC.

Usage: test.py <DXC build directory> [directory for native GPU fixtures].
Generated fixtures contain our own integer workload, not an HLK shader dump.
"""
import ctypes as C
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import time

build = Path(sys.argv[1]).resolve()
dxc = next(p for p in (build / 'bin/dxc', build / 'bin/dxc-3.7') if p.exists())
lib = C.CDLL(str(build / 'lib/libScorpiDxbcToDxil.dylib'))
lower = lib.ScorpiLowerTypedBufferViewsWithOptions
lower.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t,
                  C.c_uint32, C.c_uint32, C.c_uint32,
                  C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.c_void_p, C.c_size_t]
lib.ScorpiFreeDxilBinary.argtypes = [C.c_void_p]


def adapt(code):
    result, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(2048)
    status = lower(code, len(code), None, 0, 0xffffffff, 0, 0,
                   C.byref(result), C.byref(size), error, len(error))
    assert status == 0, (status, error.value)
    try:
        return C.string_at(result, size.value) if result.value else None
    finally:
        lib.ScorpiFreeDxilBinary(result)


def shader(body):
    return '''target triple = "dxil-ms-dx"
define void @main() {
entry:
''' + body + '''
  ret void
}
declare i32 @dx.op.loadInput.i32(i32, i32, i32, i8, i32) nounwind readnone
declare void @dx.op.storeOutput.i32(i32, i32, i32, i8, i32) nounwind
!dx.version = !{!0}
!dx.valver = !{!0}
!dx.shaderModel = !{!1}
!dx.entryPoints = !{!2}
!0 = !{i32 1, i32 0}
!1 = !{!"vs", i32 6, i32 0}
!2 = !{void ()* @main, !"main", !3, null, null}
!3 = !{!4, !7, null}
!4 = !{!5}
!5 = !{i32 0, !"TEXCOORD", i8 5, i8 0, !6, i8 0, i32 1, i8 1, i32 0, i8 0, null}
!6 = !{i32 0}
!7 = !{!8}
!8 = !{i32 0, !"SCORPI", i8 5, i8 0, !6, i8 0, i32 1, i8 4, i32 0, i8 0, null}
'''


def assemble(source, tmp):
    (tmp / 'shader.ll').write_text(shader(source))
    subprocess.run([str(build / 'bin/llvm-as'), str(tmp / 'shader.ll'),
                    '-o', str(tmp / 'shader.bc')], check=True)
    bitcode = (tmp / 'shader.bc').read_bytes()
    payload = struct.pack('<II4sIII', 0x10060, (24 + len(bitcode) + 3) // 4,
                          b'DXIL', 0x100, 16, len(bitcode)) + bitcode
    payload += bytes(-len(payload) % 4)
    return (struct.pack('<4s16sHHIII', b'DXBC', bytes(16), 1, 0,
                        44 + len(payload), 1, 36) +
            struct.pack('<4sI', b'DXIL', len(payload)) + payload)


def straight(registers, seed):
    count = registers * 4
    lines = [f'  %dx.v32.x0 = alloca [{count} x i32], align 4']
    sums = [0] * 4
    for i in range(count):
        value = ((seed + i) * 2654435761) & 0xffffffff
        sums[i % 4] = (sums[i % 4] + value) & 0xffffffff
        lines += [f'  %p{i} = getelementptr [{count} x i32], [{count} x i32]* %dx.v32.x0, i32 0, i32 {i}',
                  f'  store i32 {value}, i32* %p{i}, align 4']
    totals = ['0'] * 4
    for i in range(count):
        component = i % 4
        lines += [f'  %r{i} = load i32, i32* %p{i}, align 4',
                  f'  %s{i} = add i32 {totals[component]}, %r{i}']
        totals[component] = f'%s{i}'
    for i in range(4):
        lines += [f'  call void @dx.op.storeOutput.i32(i32 5, i32 0, i32 0, i8 {i}, i32 {totals[i]})']
    return '\n'.join(lines), sums


def evaluate_straight(ir):
    # Interpret the resulting SSA additions independently; reject unexpected IR.
    body = ir.split('define void @main()', 1)[1].split('\n}', 1)[0]
    values, outputs = {}, {}
    def value(token):
        return values[token] if token.startswith('%') else int(token) & 0xffffffff
    for line in body.splitlines():
        line = line.strip().split(';')[0].strip()
        if not line or line in ('{', 'entry:', 'ret void'):
            continue
        m = re.fullmatch(r'(%[\w.]+) = add i32 ([%\w.-]+), ([%\w.-]+)', line)
        if m:
            values[m[1]] = (value(m[2]) + value(m[3])) & 0xffffffff
            continue
        m = re.fullmatch(r'call void @dx.op.storeOutput.i32\(i32 5, i32 0, i32 0, i8 (\d), i32 ([%\w.-]+)\)', line)
        assert m, line
        outputs[int(m[1])] = value(m[2])
    return [outputs[i] for i in range(4)]


with tempfile.TemporaryDirectory(prefix='scorpi-static-temps-') as name:
    tmp = Path(name)
    for registers, seed in ((1, 17), (64, 0xfffffff0), (4096, 73)):
        source, expected = straight(registers, seed)
        code = assemble(source, tmp)
        start = time.monotonic()
        result = adapt(code)
        elapsed = time.monotonic() - start
        assert result and elapsed < 30, elapsed
        assert adapt(result) is None, 'promotion must be idempotent'
        (tmp / 'out.dxil').write_bytes(result)
        ir = subprocess.check_output([str(dxc), '-dumpbin', str(tmp / 'out.dxil')], text=True)
        assert 'alloca' not in ir and 'getelementptr' not in ir
        assert evaluate_straight(ir) == expected
        if len(sys.argv) > 2:
            output = Path(sys.argv[2]); output.mkdir(parents=True, exist_ok=True)
            # Keep the unlowered input: native pipeline preparation must run the pass.
            (output / f'temps-{registers}.dxil').write_bytes(code)
            (output / f'temps-{registers}.json').write_text(json.dumps(expected))
        print('PASS static', registers, 'registers', 'seconds', round(elapsed, 3), flush=True)

    base = '''  %dx.v32.x0 = alloca [4 x i32], align 4
  %index = call i32 @dx.op.loadInput.i32(i32 4, i32 0, i32 0, i8 0, i32 undef)
  %p = getelementptr [4 x i32], [4 x i32]* %dx.v32.x0, i32 0, INDEX
  store QUALIFIER i32 7, i32* %p, align 4
  %v = load QUALIFIER i32, i32* %p, align 4
  call void @dx.op.storeOutput.i32(i32 5, i32 0, i32 0, i8 0, i32 %v)'''
    for label, index, qualifier in [('dynamic', '%index', ''), ('negative', '-1', ''),
                                     ('out-of-bounds', '4', ''), ('volatile', '0', 'volatile')]:
        source = base.replace('INDEX', 'i32 ' + index).replace('QUALIFIER ', qualifier + ' ' if qualifier else '')
        assert adapt(assemble(source, tmp)) is None, label
        print('PASS unchanged', label)

    branch = '''  %dx.v32.x0 = alloca [4 x i32], align 4
  %p = getelementptr [4 x i32], [4 x i32]* %dx.v32.x0, i32 0, i32 0
  %index = call i32 @dx.op.loadInput.i32(i32 4, i32 0, i32 0, i8 0, i32 undef)
  %choose = icmp eq i32 %index, 0
  br i1 %choose, label %left, label %right
left:
  store i32 7, i32* %p, align 4
  br label %join
right:
  store i32 11, i32* %p, align 4
  br label %join
join:
  %v = load i32, i32* %p, align 4
  call void @dx.op.storeOutput.i32(i32 5, i32 0, i32 0, i8 0, i32 %v)'''
    result = adapt(assemble(branch, tmp)); assert result
    (tmp / 'out.dxil').write_bytes(result)
    ir = subprocess.check_output([str(dxc), '-dumpbin', str(tmp / 'out.dxil')], text=True)
    phi = re.search(r'phi i32 ([^\n]+)', ir)
    assert 'alloca' not in ir and phi, ir
    assert set(re.findall(r'\[ (\d+), %(\w+) \]', phi[1])) == {('7', 'left'), ('11', 'right')}, ir
    assert adapt(result) is None
    print('PASS branch-dependent stores')
