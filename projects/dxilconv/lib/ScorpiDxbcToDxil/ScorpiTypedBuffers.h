#pragma once
#include "ScorpiDxbcToDxil.h"
namespace hlsl { class DxilModule; }
bool ScorpiLowerTypedBuffers(hlsl::DxilModule &module,
    const ScorpiTypedBufferBinding *bindings, size_t binding_count,
    uint32_t metadata_space, uint32_t self_offset, uint32_t options,
    const ScorpiSamplingRequirement *policy, size_t policy_count);
