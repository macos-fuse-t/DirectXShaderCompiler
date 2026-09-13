#pragma once

#include <cstdint>
#include <string>

namespace hlsl {
class DxilModule;
}
namespace scorpi {
// Adds a private cbuffer b0 and raw SRV t0 in registerSpace. The caller must
// provide a per-generated-primitive coverage slice and matching root bindings.
bool PrepareRasterReplayPixelShader(hlsl::DxilModule &module,
                                    uint32_t registerSpace,
                                    uint32_t sampleCount, uint32_t sampleMask,
                                    std::string &error, bool generated = false);
} // namespace scorpi
