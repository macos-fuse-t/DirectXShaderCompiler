#pragma once

#include <cstdint>
#include <string>
#include "ScorpiDxbcToDxil.h"

namespace hlsl {
class DxilModule;
}
namespace scorpi {
bool PrepareMsaa8TextureLoads(hlsl::DxilModule &module, std::string &error, bool &changed,
    const ScorpiSamplingRequirement *policy = nullptr, size_t policy_count = 0);
// Adds a private cbuffer b0, raw SRV t0 and visibility UAV u1 in registerSpace. The caller must
// provide a per-generated-primitive coverage slice and matching root bindings.
bool PrepareRasterReplayPixelShader(hlsl::DxilModule &module,
                                    uint32_t registerSpace,
                                    uint32_t sampleCount, uint32_t sampleMask,
                                    std::string &error, bool generated = false,
                                    bool attachments = false, bool alphaToCoverage = false,
                                    uint32_t *requirements = nullptr);
} // namespace scorpi
