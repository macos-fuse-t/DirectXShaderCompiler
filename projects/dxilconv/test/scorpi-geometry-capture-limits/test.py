#!/usr/bin/env python3
"""Check capture layout and disabled-raster ABI at the GS scalar-output limit.

Usage: test.py <DXC build directory>. GPU output/termination coverage lives in
the renderer's dx12-metal-geometry-capture-limits test.
"""
import ctypes as C
from pathlib import Path
import re
import subprocess
import sys
import tempfile

build = Path(sys.argv[1]).resolve()
dxc = next(p for p in (build / 'bin/dxc', build / 'bin/dxc-3.7') if p.exists())
lib = C.CDLL(str(build / 'lib/libScorpiDxbcToDxil.dylib'))
capture = lib.ScorpiCreateGeometryShaderCapture
capture.argtypes = [C.c_void_p, C.c_size_t, C.c_uint32, C.c_int,
                    C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t),
                    C.c_void_p, C.c_size_t]
lib.ScorpiFreeDxilBinary.argtypes = [C.c_void_p]

with tempfile.TemporaryDirectory(prefix='scorpi-geometry-limits-') as name:
    tmp = Path(name)
    for width in (3, 4):
        source = tmp / 'shader.hlsl'
        source.write_text('''
struct Output { uint%d value : TEXCOORD0; };
[maxvertexcount(%d)]
void main(point uint input[1] : TEXCOORD0, inout PointStream<Output> stream) {
    for (uint i = 0; i < input[0]; ++i) {
        Output o; o.value = i; stream.Append(o);
    }
}
''' % (width, 1024 // width))
        path = tmp / 'original.dxil'
        subprocess.run([str(dxc), '-T', 'gs_6_0', '-E', 'main', '-Fo', str(path), str(source)], check=True)
        code = path.read_bytes()
        for mode in (1, 2, 3, 4):
            layout = (C.c_uint32 * 5)()
            binary, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(2048)
            rc = capture(code, len(code), 1234, mode, layout, C.byref(binary), C.byref(size), error, len(error))
            assert rc == 0, (rc, error.value)
            try:
                path = tmp / 'capture.dxil'
                path.write_bytes(C.string_at(binary, size.value))
            finally:
                lib.ScorpiFreeDxilBinary(binary)
            assert list(layout) == [width * 4, 1024 // width, 1, 1, 1]
            ir = subprocess.check_output([str(dxc), '-dumpbin', str(path)], text=True)
            assert re.search(r'icmp uge i32 [^,]+, ' + str(1024 // width), ir)
            if mode in (3, 4):
                assert 'SV_Position' in ir
                # The application bound stays in layout; Metal exports only
                # one dummy point, so padding a uint3 to float4 cannot reduce
                # or invalidate the application's 341-vertex capture capacity.
                assert re.search(r'!\{i32 1, i32 1, i32 1, i32 1, i32 1\}', ir)
            print('PASS width', width, 'capture mode', mode)
