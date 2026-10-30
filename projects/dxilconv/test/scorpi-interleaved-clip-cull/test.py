#!/usr/bin/env python3
"""Exercise the real converter with packed DDI clip/cull signature inputs."""
from pathlib import Path
import itertools
import re
import struct
import subprocess
import sys
import tempfile

build = Path(sys.argv[1]).resolve()
fixture = Path(__file__).resolve().parent
suffix = ".exe" if sys.platform == "win32" else ""
probe = build / "bin" / ("scorpi-dxbc-probe" + suffix)
dxc = build / "bin" / ("dxc" + suffix)
original_shader = (fixture / "vertex.tokens").read_bytes()
original_signature = (fixture / "vertex.signatures").read_bytes()
header = struct.unpack_from("<10I", original_signature)
inputs = original_signature[header[6]:header[6] + header[3] * 24]
outputs = [struct.unpack_from("<6I", original_signature, header[7] + i * 24)
           for i in range(header[4])]

with tempfile.TemporaryDirectory(prefix="scorpi-clip-cull-") as directory:
    directory = Path(directory)
    code, sig, result = [directory / name for name in ("shader.bin", "signature.bin", "shader.dxil")]

    def convert(shader, entries, expected):
        h = list(header)
        h[4] = len(entries)
        code.write_bytes(shader)
        sig.write_bytes(struct.pack("<10I", *h) + inputs + b"".join(struct.pack("<6I", *e) for e in entries))
        run = subprocess.run([str(probe), str(code), str(sig), str(result)], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        ir = subprocess.run([str(dxc), "-dumpbin", str(result)], check=True, capture_output=True, text=True).stdout
        # Read the serialized output signature, independently of element order.
        section = ir.split("; Output signature:", 1)[1].split("; Pipeline Runtime Information:", 1)[0]
        actual, identities = {}, set()
        for name, index, mask, register in re.findall(r"; (SV_\w+)\s+(\d+)\s+([xyzw]+)\s+(\d+)\s+", section):
            assert (name, index) not in identities, "Duplicate semantic identity"
            identities.add((name, index))
            for c in mask:
                key = (int(register), "xyzw".index(c))
                assert key not in actual, "Overlapping signature owners"
                actual[key] = name
        assert actual == expected, (actual, expected)
        assert "@dx.op.tempReg" not in ir, "Dead temporary-register declarations must be removed"
        # Every captured output component must still have a real shader write.
        assert ir.count("call void @dx.op.storeOutput.f32(") == 8, ir

    expected = {(0, c): "SV_Position" for c in range(4)}
    expected.update({(1, c): "SV_ClipDistance" if c % 2 == 0 else "SV_CullDistance" for c in range(4)})
    for permutation in itertools.permutations(outputs[1:]):
        convert(original_shader, [outputs[0], *permutation], expected)
    # A single DDI entry can itself have a noncontiguous mask.
    combined = [outputs[0], (2, 1, 5, 0, 3, 0), (3, 1, 10, 0, 3, 0)]
    convert(original_shader, combined, expected)
    convert(original_shader, [combined[0], combined[2], combined[1]], expected)

    # Contiguous components must still coalesce regardless of DDI entry order.
    words = list(struct.unpack("<%dI" % (len(original_shader) // 4), original_shader))
    i = 2
    while i < len(words):
        count = (words[i] >> 24) & 0x7f
        assert count > 0
        if words[i] & 0x7ff == 103 and words[i + count - 1] == 3:
            words[i + count - 1] = 2
        i += count
    shader = struct.pack("<%dI" % len(words), *words)
    scalar = [(2, 1, 1 << c, 0, 3, 0) for c in range(4)]
    contiguous = {(0, c): "SV_Position" for c in range(4)}
    contiguous.update({(1, c): "SV_ClipDistance" for c in range(4)})
    for permutation in itertools.permutations(scalar):
        convert(shader, [outputs[0], *permutation], contiguous)
    convert(shader, [outputs[0], (2, 1, 15, 0, 3, 0)], contiguous)
    # Conflicting structural records must return an error and unwind normally,
    # rather than asserting and escaping the converter through Windows SEH.
    sig.write_bytes(struct.pack("<10I", header[0], header[1], 24, 2, 3, 0, 40, 88, 0, 0) + inputs + b"".join(struct.pack("<6I", *e) for e in [outputs[0], (2, 1, 5, 0, 3, 0), (3, 1, 14, 0, 3, 0)]))
    code.write_bytes(original_shader)
    run = subprocess.run([str(probe), str(code), str(sig), str(result)], capture_output=True, text=True)
    assert run.returncode == 1 and "0x80aa0010" in run.stderr, run
print("interleaved clip/cull DDI conversion: PASS (51 valid cases, overlap rejected)")
