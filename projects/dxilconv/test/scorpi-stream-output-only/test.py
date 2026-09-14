#!/usr/bin/env python3
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
build=Path(sys.argv[1]).resolve();fixture=Path(__file__).resolve().parent
suffix=".exe" if sys.platform=="win32" else ""
probe=build/"bin"/("scorpi-dxbc-probe"+suffix);dxc=build/"bin"/("dxc"+suffix)
signature=bytearray((fixture/"geometry.signatures").read_bytes())
shader=bytearray((fixture/"geometry.tokens").read_bytes())
with tempfile.TemporaryDirectory(prefix="scorpi-so-conversion-") as tmp:
    tmp=Path(tmp);code=tmp/"shader.bin";sig=tmp/"signature.bin";out=tmp/"out.dxil"
    def convert(flags, stage=2):
        struct.pack_into("<I",signature,4,flags)
        struct.pack_into("<I",shader,0,(stage<<16)|0x50)
        code.write_bytes(shader);sig.write_bytes(signature)
        return subprocess.run([str(probe),str(code),str(sig),str(out)],capture_output=True,text=True)
    result=convert(0)
    assert result.returncode!=0 and "0x80aa0010" in result.stderr,result
    result=convert(1)
    assert result.returncode==0,result.stderr
    ir=subprocess.run([str(dxc),"-dumpbin",str(out)],check=True,capture_output=True,text=True).stdout
    assert "storeOutput.i32" in ir and "storeOutput.f32" in ir
    assert "SCORPI" in ir and "SV_Position" in ir
    assert "uitofp" not in ir, "Raw stream output must not numerically convert uint payloads"
    assert convert(16).returncode!=0, "Unknown signature flags must fail"
    assert convert(1,1).returncode!=0, "Stream-output-only flag must require GS tokens"
    # Captured D3D11 DDI input: only the rasterized stream is typed.
    shader = bytearray((fixture / "multistream.tokens").read_bytes())
    signature = bytearray((fixture / "multistream.signatures").read_bytes())
    original = bytearray(signature)
    assert convert(0).returncode != 0, "Missing rasterization context must not guess output types"
    for stream in range(2):
        signature = bytearray(original)
        output_count, output_offset = struct.unpack_from("<I", signature, 16)[0], struct.unpack_from("<I", signature, 28)[0]
        for index in range(output_count):
            struct.pack_into("<I", signature, output_offset + index * 24 + 12, stream)
        result = convert(2 | (stream << 2))
        assert result.returncode == 0, result.stderr
        ir = subprocess.run([str(dxc), "-dumpbin", str(out)], check=True, capture_output=True, text=True).stdout
        assert "storeOutput.i32" in ir and "storeOutput.f32" in ir
        assert "uitofp" not in ir, "Non-rasterized stream values must retain their bits"
        # Removing an arbitrary record from the rasterized stream is still an error.
        struct.pack_into("<I", signature, 16, 1)
        assert convert(2 | (stream << 2)).returncode != 0
    signature = bytearray(original)
    assert convert(3).returncode != 0, "SO-only and rasterized-stream flags conflict"
    assert convert(4).returncode != 0, "Stream index requires the rasterized-stream flag"
    assert convert(2, 1).returncode != 0, "Rasterized-stream flag requires GS tokens"
print("stream-output DDI conversion: PASS")
