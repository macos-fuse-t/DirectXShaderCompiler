#!/usr/bin/env python3
"""Check minimum-precision interface conversion without changing native 16-bit IR.

Usage: test.py <DXC build directory> [LLVM disassembler].
"""
import ctypes as C
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

build = Path(sys.argv[1]).resolve()
dis = Path(sys.argv[2]) if len(sys.argv) > 2 else build / 'bin/llvm-dis'
lib = C.CDLL(str(build / 'lib/libScorpiDxbcToDxil.dylib'))
lower = lib.ScorpiLowerTypedBufferViewsWithSamplingPolicy
lower.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t,
                   C.c_uint32, C.c_uint32, C.c_uint32, C.c_void_p, C.c_size_t,
                   C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.c_void_p, C.c_size_t]
lib.ScorpiFreeDxilBinary.argtypes = [C.c_void_p]
fixture = Path(__file__).with_name('interface.ll').read_text()


def program(bitcode, minor=0):
    payload = struct.pack('<II4sIII', 0x60 + minor, (24 + len(bitcode) + 3) // 4,
                          b'DXIL', 0x100, 16, len(bitcode)) + bitcode
    payload += bytes(-len(payload) % 4)
    return (struct.pack('<4s16sHHIII', b'DXBC', bytes(16), 1, 0,
                        44 + len(payload), 1, 36) +
            struct.pack('<4sI', b'DXIL', len(payload)) + payload)


def ir(code, tmp):
    parts = struct.unpack_from('<I', code, 28)[0]
    for offset in struct.unpack_from('<%dI' % parts, code, 32):
        if code[offset:offset + 4] == b'DXIL':
            start = offset + 8
            bc_offset, bc_size = struct.unpack_from('<II', code, start + 16)
            (tmp / 'out.bc').write_bytes(code[start + 8 + bc_offset:start + 8 + bc_offset + bc_size])
            return subprocess.check_output([str(dis), str(tmp / 'out.bc'), '-o', '-'], text=True)
    raise AssertionError('No DXIL program')


with tempfile.TemporaryDirectory(prefix='scorpi-minimum-precision-') as name:
    tmp = Path(name)
    for kind in ['float', 'internal', 'signed', 'unsigned', 'default', 'native16',
                 'nonfinite', 'finite-fast', 'native16-nonfinite']:
        source = fixture
        if kind in ['signed', 'unsigned']:
            source = source.replace('half', 'i16').replace('.f16', '.i16')
            source = source.replace('fadd i16 %input, 0xH3C00', 'add i16 %input, 1')
            source = source.replace('i8 8, i8', 'i8 %d, i8' % (2 if kind == 'signed' else 3))
        elif kind == 'default':
            source = source.replace('half', 'float').replace('.f16', '.f32')
            source = source.replace('0xH3C00', '1.0').replace('i8 8, i8', 'i8 9, i8').replace('i64 288', 'i64 256')
        elif kind in ['internal', 'native16']:
            source = source.replace('  %input = call half @dx.op.loadInput.f16', '  %wide = call float @dx.op.loadInput.f32')
            source = source.replace('  %sum = fadd half', '  %input = fptrunc float %wide to half\n  %sum = fadd half')
            source = source.replace('  call void @dx.op.storeOutput.f16', '  %output = fpext half %sum to float\n  call void @dx.op.storeOutput.f32').replace('half %sum)', 'float %output)')
            source = source.replace('declare half @dx.op.loadInput.f16', 'declare float @dx.op.loadInput.f32').replace('declare void @dx.op.storeOutput.f16(i32, i32, i32, i8, half)', 'declare void @dx.op.storeOutput.f32(i32, i32, i32, i8, float)').replace('i8 8, i8', 'i8 9, i8')
        if kind in ['nonfinite', 'finite-fast', 'native16-nonfinite']:
            source = source.replace('fadd half', 'fadd fast half')
            if kind != 'finite-fast': source = source.replace('0xH3C00', '0xHFC00')
        if kind in ['native16', 'native16-nonfinite']:
            source = source.replace('i32 6, i32 0', 'i32 6, i32 2').replace('i64 288', 'i64 8388896')
        (tmp / 'shader.ll').write_text(source)
        subprocess.run([str(build / 'bin/llvm-as'), str(tmp / 'shader.ll'), '-o', str(tmp / 'shader.bc')], check=True)
        code = program((tmp / 'shader.bc').read_bytes(), 2 if kind.startswith('native16') else 0)
        result, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(2048)
        rc = lower(code, len(code), None, 0, 0xffffffff, 0, 0, None, 0,
                   C.byref(result), C.byref(size), error, len(error))
        assert rc == 0, (kind, rc, error.value)
        if kind in ['default', 'native16', 'unsigned', 'native16-nonfinite']:
            assert not result.value and size.value == 0, kind
        else:
            assert result.value, kind
            try:
                output = C.string_at(result, size.value)
            finally:
                lib.ScorpiFreeDxilBinary(result)
            text = ir(output, tmp)
            if kind in ['float', 'internal', 'nonfinite', 'finite-fast']:
                assert '@dx.op.loadInput.f32' in text and '@dx.op.storeOutput.f32' in text
                assert 'dx.op.legacyF32ToF16' in text and 'fpext half' in text
                expected_add = 'fadd nsz arcp half' if kind == 'nonfinite' else 'fadd fast half' if kind == 'finite-fast' else 'fadd half'
                assert expected_add in text, (kind, text)
            else:
                assert '@dx.op.loadInput.i32' in text and '@dx.op.storeOutput.i32' in text
                assert 'add i16' in text and 'trunc i32' in text
                assert 'sext i16' in text
            # The converted interface is stable when prepared a second time.
            result2, size2 = C.c_void_p(), C.c_size_t()
            assert lower(output, len(output), None, 0, 0xffffffff, 0, 0, None, 0,
                         C.byref(result2), C.byref(size2), error, len(error)) == 0
            assert not result2.value and size2.value == 0
        print('PASS', kind)
