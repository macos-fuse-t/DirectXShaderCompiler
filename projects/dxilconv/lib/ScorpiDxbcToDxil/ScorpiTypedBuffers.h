#pragma once
#include "ScorpiDxbcToDxil.h"
namespace llvm { class Value; }
namespace hlsl { class DxilModule; }
bool ScorpiInlineDxbcSubroutines(hlsl::DxilModule &module);
bool ScorpiWidenMinimumPrecisionIO(hlsl::DxilModule &module);
bool ScorpiLowerTypedBuffers(hlsl::DxilModule &module,
    const ScorpiTypedBufferBinding *bindings, size_t binding_count,
    uint32_t metadata_space, uint32_t self_offset, uint32_t options,
    const ScorpiSamplingRequirement *policy, size_t policy_count);

llvm::Value *ScorpiAtomic64Origin(hlsl::DxilModule &, llvm::Value *,
    const ScorpiTypedBufferBinding *, size_t, uint32_t);
