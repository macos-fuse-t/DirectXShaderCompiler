#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include "ScorpiDxbcToDxil.h"
namespace llvm { class Module; }
namespace hlsl { class DxilModule; }
namespace scorpi {
bool LowerGeometryPrimitiveId(hlsl::DxilModule &source, std::string &error);
bool PrepareRasterReplayVertexShader(hlsl::DxilModule &source,
    uint32_t registerSpace, std::string &error);
// Capture an assembled input primitive after the application's VS. The replay
// VS reads the same packed output signature; no application code is replayed.
std::unique_ptr<llvm::Module> CreateGeometryCaptureModule(
    const hlsl::DxilModule &source, uint32_t registerSpace,
    uint32_t verticesPerPrimitive, bool capture, uint32_t &vertexStride,
    std::string &error, const ScorpiGeometryCaptureLayout *generated = nullptr, bool compacted = false);
bool GetGeometryCaptureLayout(const hlsl::DxilModule &source,
    ScorpiGeometryCaptureLayout &layout, std::string &error);
bool PrepareGeometryShaderCapture(hlsl::DxilModule &source,
    uint32_t registerSpace, const ScorpiGeometryCaptureLayout &layout,
    std::string &error);
}
