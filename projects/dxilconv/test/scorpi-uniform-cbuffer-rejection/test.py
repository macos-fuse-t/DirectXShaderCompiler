#!/usr/bin/env python3
"""Check rejection of ordinary SM5 uniform CB binding indices without an AV.

Usage: test.py <DXC build directory>. This verifies failure handling, not
functional support for D3D11.3 specification section 7.11 uniform indexing.
"""
import ctypes as C
import json
from pathlib import Path
import resource
import struct
import subprocess
import sys
import tempfile

build = Path(sys.argv[1]).resolve()
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
if len(sys.argv) > 2 and sys.argv[2] == 'child':
    lib = C.CDLL(str(build / 'lib/libScorpiDxbcToDxil.dylib'))
    convert = lib.ScorpiCreateDxilBinary
    convert.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t,
                        C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.c_void_p, C.c_size_t]
    lib.ScorpiFreeDxilBinary.argtypes = [C.c_void_p]
    code = Path(sys.argv[3]).read_bytes()
    binary, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(2048)
    rc = convert(code, len(code), None, 0, C.byref(binary), C.byref(size), error, len(error))
    print(json.dumps({'Status': rc, 'Error': error.value.decode(), 'Bytes': size.value}))
    lib.ScorpiFreeDxilBinary(binary)
    raise SystemExit(0)

fixture = (Path(__file__).resolve().parent.parent / 'dxbc2dxil/cbuffer1.50.dxbc').read_bytes()
chunks = []
for index in range(struct.unpack_from('<I', fixture, 28)[0]):
    offset = struct.unpack_from('<I', fixture, 32 + index * 4)[0]
    size = struct.unpack_from('<I', fixture, offset + 4)[0]
    chunks.append((fixture[offset:offset + 4], fixture[offset + 8:offset + 8 + size]))


def variant(mode):
    parts = []
    for tag, payload in chunks:
        if tag in (b'SHEX', b'SHDR') and mode != 'static':
            words = list(struct.unpack('<%dI' % (len(payload) // 4), payload))
            # The fixture reads cb5[0] directly. Replace only that load with
            # cb[r0.x][0] or cb[r0.x+3][0], initializing r0.x to 5 or 2.
            assert words[10:] == [0x06000036, 0x001020f2, 0, 0x00208e46, 5, 0, 0x0100003e]
            relative = [0x0010000a, 0] if mode == 'relative' else [3, 0x0010000a, 0]
            representation = 2 if mode == 'relative' else 3
            load = [((5 + len(relative)) << 24) | 54, 0x001020f2, 0,
                    0x00208e46 | (representation << 22), *relative, 0]
            declare_temp = [0x02000068, 1]
            initialize_temp = [0x05000036, 0x00100012, 0, 0x00004001,
                               5 if mode == 'relative' else 2]
            words = words[:10] + declare_temp + initialize_temp + load + [0x0100003e]
            words[1] = len(words)
            payload = struct.pack('<%dI' % len(words), *words)
        parts.append(tag + struct.pack('<I', len(payload)) + payload)
    offset = 32 + len(parts) * 4
    offsets = []
    for part in parts:
        offsets.append(offset)
        offset += len(part)
    return (b'DXBC' + bytes(16) + struct.pack('<III', 1, offset, len(parts)) +
            struct.pack('<%dI' % len(parts), *offsets) + b''.join(parts))


with tempfile.TemporaryDirectory(prefix='scorpi-uniform-cbuffer-') as name:
    for mode in ('static', 'relative', 'offset-relative'):
        path = Path(name) / (mode + '.dxbc')
        path.write_bytes(variant(mode))
        # Isolate the native call: the regression originally killed the process.
        result = subprocess.run([sys.executable, __file__, str(build), 'child', str(path)],
                                capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, (mode, result.returncode, result.stderr)
        data = json.loads(result.stdout)
        if mode == 'static':
            assert data['Status'] == 0 and data['Bytes'] > 0, data
        else:
            assert data['Status'] != 0 and data['Bytes'] == 0 and data['Error'], data
        print('PASS', mode, data)
