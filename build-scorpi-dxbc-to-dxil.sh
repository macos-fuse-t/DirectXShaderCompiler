#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${1:-"$script_dir/build"}"

d3d_headers="$script_dir/external/DirectX-Headers/include/directx"
if [[ ! -d "$d3d_headers" ]]; then
  echo "missing DirectX-Headers at $d3d_headers" >&2
  exit 1
fi

cmake \
  -S "$script_dir" \
  -B "$build_dir" \
  -DHLSL_INCLUDE_TESTS=OFF \
  -DD3D12_INCLUDE_DIR="$d3d_headers" \
  -DDXGI_INCLUDE_DIR="$d3d_headers"

cmake \
  --build "$build_dir" \
  --target ScorpiDxbcToDxil ScorpiDxbcToDxilStatic scorpi-dxbc-probe \
  -j "${SCORPI_DXIL_JOBS:-$(sysctl -n hw.ncpu)}"

echo "built:"
echo "  $build_dir/lib/libScorpiDxbcToDxil.dylib"
echo "  $build_dir/lib/libScorpiDxbcToDxil.a"
echo "  $build_dir/bin/scorpi-dxbc-probe"
