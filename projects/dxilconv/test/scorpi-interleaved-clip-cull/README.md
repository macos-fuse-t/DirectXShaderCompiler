# Interleaved clip/cull DDI signatures

The vertex shader and runtime signature were captured at `ScorpiCreateDxilBinary`
from native ARM64 WGF11 Clipping (`Clipping/Line/ClipDist:1`). The position is
in output register 0; register 1 contains clip distances in x/z and cull distances
in y/w. The converter must preserve those four owners without introducing holes,
overlap, duplicate semantic indices, or extra clip planes.

Run `python3 test.py <dxc-build-directory>`. Requires the `scorpi-dxbc-probe` and
`dxc` targets. Tests the captured data, reordered and combined DDI records, and
contiguous scalar clip records in every order. It does not execute a GPU workload
or establish support for cull-distance rasterization.
