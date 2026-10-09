#pragma once
#include "ScorpiDxbcToDxil.h"
#include <cstdint>
namespace hlsl {
class DxilModule;
}
namespace llvm {
class Module;
}
// Returns false without changing modules that have no resource i64 atomics.
bool ScorpiLowerAtomic64(hlsl::DxilModule &module, uint32_t lock_space,
                         const ScorpiTypedBufferBinding *bindings,
                         size_t binding_count);
bool ScorpiHasAtomic64(llvm::Module &module);
