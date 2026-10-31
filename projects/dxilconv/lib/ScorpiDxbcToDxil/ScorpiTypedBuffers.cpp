#include "ScorpiTypedBuffers.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilResource.h"
#include "dxc/DXIL/DxilSampler.h"
#include "dxc/DXIL/DxilResourceBinding.h"
#include "dxc/DXIL/DxilResourceProperties.h"
#include "dxc/DXIL/DxilShaderModel.h"
#include "dxc/DXIL/DxilSignature.h"
#include "dxc/DXIL/DxilSignatureElement.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <functional>
#include <set>
#include <stdexcept>
#include <vector>

using namespace llvm;
using namespace hlsl;
using namespace hlsl::DXIL;

namespace {
OpCode Opcode(Value *value) {
  if (!value) return OpCode::NumOpCodes;
  auto *call = dyn_cast<CallInst>(value);
  if (call && call->getCalledFunction() &&
      call->getCalledFunction()->getName().startswith("dx.op.") && call->getNumArgOperands())
    if (auto *op = dyn_cast<ConstantInt>(call->getArgOperand(0)))
      return static_cast<OpCode>(op->getZExtValue());
  return OpCode::NumOpCodes;
}

ResourceKind Kind(Value *handle, DxilModule &dm, std::set<Value *> &seen) {
  if (!seen.insert(handle).second) return ResourceKind::Invalid;
  if (auto *call = dyn_cast<CallInst>(handle)) {
    if (Opcode(call) == OpCode::AnnotateHandle)
      return resource_helper::loadPropsFromConstant(*cast<Constant>(call->getArgOperand(2))).getResourceKind();
    if (Opcode(call) == OpCode::CreateHandle) {
      auto cls = cast<ConstantInt>(call->getArgOperand(1))->getZExtValue();
      auto id = cast<ConstantInt>(call->getArgOperand(2))->getZExtValue();
      if (cls == unsigned(ResourceClass::SRV) && id < dm.GetSRVs().size()) return dm.GetSRV(id).GetKind();
      if (cls == unsigned(ResourceClass::UAV) && id < dm.GetUAVs().size()) return dm.GetUAV(id).GetKind();
    }
  }
  ResourceKind result = ResourceKind::Invalid;
  auto merge = [&](Value *v) {
    auto next = Kind(v, dm, seen);
    if (result != ResourceKind::Invalid && next != ResourceKind::Invalid && result != next)
      throw std::runtime_error("mixed typed-buffer handle kinds");
    if (next != ResourceKind::Invalid) result = next;
  };
  if (auto *phi = dyn_cast<PHINode>(handle)) for (auto &v : phi->incoming_values()) merge(v);
  if (auto *select = dyn_cast<SelectInst>(handle)) { merge(select->getTrueValue()); merge(select->getFalseValue()); }
  return result;
}
bool Gather(OpCode op){return op==OpCode::TextureGather||op==OpCode::TextureGatherCmp;}
bool Comparison(OpCode op){return op==OpCode::SampleCmp||op==OpCode::SampleCmpLevelZero||op==OpCode::TextureGatherCmp;}
bool Implicit(OpCode op){return op==OpCode::Sample||op==OpCode::SampleBias||op==OpCode::SampleCmp;}
bool ClampOperand(OpCode op){return op==OpCode::Sample||op==OpCode::SampleBias||op==OpCode::SampleGrad||op==OpCode::SampleCmp;}

// This GPU's native minimum-LOD operand is quantized to sixteenths. Exact
// native clamps stay native; only instructions requesting finer precision
// need a fallback. Dynamic clamps make this decision at the access itself.
bool NeedsClampPrecision(CallInst *call) {
  auto op = Opcode(call);
  if (!ClampOperand(op)) return false;
  unsigned at = op == OpCode::SampleGrad ? 16 : (op == OpCode::SampleBias || op == OpCode::SampleCmp) ? 11 : 10;
  auto *value = call->getArgOperand(at);
  if (isa<UndefValue>(value)) return false;
  if (auto *constant = dyn_cast<ConstantFP>(value)) {
    double scaled = constant->getValueAPF().convertToDouble() * 16;
    return scaled > 0 && std::isfinite(scaled) && scaled != std::floor(scaled);
  }
  return true;
}

// Only derivatives need quad-uniform execution before the mode branch. Keep
// dimensions, cube projection and anisotropic footprint algebra off the native
// path; a regular hardware sample must not pay for its software fallback.
std::vector<Instruction *> FootprintInstructions(Instruction *anchor, Instruction *end) {
  std::vector<Instruction *> result;
  // A sample can be the first instruction of a conditional block; its
  // operands/handles may all dominate it from another block.
  for (auto *i = anchor ? anchor->getNextNode() : &end->getParent()->front();
       i != end; i = i->getNextNode())
    if (Opcode(i) != OpCode::DerivCoarseX && Opcode(i) != OpCode::DerivCoarseY)
      result.push_back(i);
  return result;
}
void SinkFootprint(const std::vector<Instruction *> &instructions, BasicBlock *destination) {
  for (auto *i : instructions) {
    i->removeFromParent();
    destination->getInstList().push_back(i);
  }
}

// Keep a single copy of the expensive addressing/residency/seam machinery per
// mip. Emitting it once per tap balloons ordinary shaders even when runtime
// descriptors select the native path. Scalar PHIs also avoid AIR aggregate-PHI
// serialization limitations. The loop preserves the original accumulation order.
class SampleTapLoop {
  BasicBlock *body, *exit;
  PHINode *values[5];
  unsigned count;
public:
  PHINode *index;
  SampleTapLoop(IRBuilder<> &b, Value *&result, unsigned taps, BasicBlock *after)
      : count(taps) {
    auto *before = b.GetInsertBlock();
    body = BasicBlock::Create(before->getContext(), "scorpi.tap.loop", before->getParent(), after);
    exit = BasicBlock::Create(before->getContext(), "scorpi.tap.end", before->getParent(), after);
    Value *initial[5];
    for (unsigned c = 0; c < 5; ++c)
      initial[c] = isa<UndefValue>(result) ?
          (c == 4 ? static_cast<Value *>(b.getInt32(1)) : Constant::getNullValue(result->getType()->getStructElementType(c))) :
          b.CreateExtractValue(result, c);
    b.CreateBr(body); b.SetInsertPoint(body);
    index = b.CreatePHI(b.getInt32Ty(), 2, "scorpi.tap.index");
    index->addIncoming(b.getInt32(0), before);
    for (unsigned c = 0; c < 5; ++c) {
      values[c] = b.CreatePHI(initial[c]->getType(), 2, "scorpi.tap.value");
      values[c]->addIncoming(initial[c], before);
    }
    for (unsigned c = 0; c < 5; ++c) result = b.CreateInsertValue(result, values[c], c);
  }
  void End(IRBuilder<> &b, Value *result) {
    auto *latch = b.GetInsertBlock();
    auto *next = b.CreateAdd(index, b.getInt32(1));
    index->addIncoming(next, latch);
    for (unsigned c = 0; c < 5; ++c) values[c]->addIncoming(b.CreateExtractValue(result, c), latch);
    auto *branch = b.CreateCondBr(b.CreateICmpULT(next, b.getInt32(count)), body, exit);
    auto &context = branch->getContext();
    auto *disable = MDNode::get(context, MDString::get(context, "llvm.loop.unroll.disable"));
    auto *loop = MDNode::get(context, {nullptr, disable});
    loop->replaceOperandWith(0, loop);
    branch->setMetadata("llvm.loop", loop);
    b.SetInsertPoint(exit);
  }
};

unsigned StructuredStride(Value *handle, DxilModule &dm, std::set<Value *> &seen) {
  if (!seen.insert(handle).second) return 0;
  if (auto *call = dyn_cast<CallInst>(handle)) {
    if (Opcode(call) == OpCode::AnnotateHandle)
      return resource_helper::loadPropsFromConstant(*cast<Constant>(call->getArgOperand(2))).getElementStride();
    if (Opcode(call) == OpCode::CreateHandle) {
      auto cls = cast<ConstantInt>(call->getArgOperand(1))->getZExtValue();
      auto id = cast<ConstantInt>(call->getArgOperand(2))->getZExtValue();
      if (cls == unsigned(ResourceClass::SRV) && id < dm.GetSRVs().size()) return dm.GetSRV(id).GetElementStride();
      if (cls == unsigned(ResourceClass::UAV) && id < dm.GetUAVs().size()) return dm.GetUAV(id).GetElementStride();
    }
  }
  unsigned result = 0;
  auto merge = [&](Value *v) {
    unsigned next = StructuredStride(v, dm, seen);
    if (result && next && result != next) throw std::runtime_error("mixed structured-buffer strides");
    if (next) result = next;
  };
  if (auto *phi = dyn_cast<PHINode>(handle)) for (auto &v : phi->incoming_values()) merge(v);
  if (auto *select = dyn_cast<SelectInst>(handle)) { merge(select->getTrueValue()); merge(select->getFalseValue()); }
  return result;
}

// Resolve the union of semantic requirements reachable by this handle. A
// dynamic array index affects its declared range, never an unrelated range.
// PHI cycles contribute no new binding; unknown producers are conservative.
uint32_t SamplingRequirements(DxilModule &dm, Value *handle,
    const ScorpiSamplingRequirement *policy, size_t count, std::set<Value *> &seen) {
  if (!seen.insert(handle).second) return 0;
  if (auto *phi = dyn_cast<PHINode>(handle)) {
    uint32_t result = 0;
    for (auto &v : phi->incoming_values()) result |= SamplingRequirements(dm, v, policy, count, seen);
    return result;
  }
  if (auto *s = dyn_cast<SelectInst>(handle))
    return SamplingRequirements(dm, s->getTrueValue(), policy, count, seen) |
        SamplingRequirements(dm, s->getFalseValue(), policy, count, seen);
  auto *call = dyn_cast<CallInst>(handle);
  if (!call) return UINT32_MAX;
  if (Opcode(call) == OpCode::AnnotateHandle)
    return SamplingRequirements(dm, call->getArgOperand(1), policy, count, seen);
  uint32_t cls, space;
  uint64_t first, end;
  Value *index;
  if (Opcode(call) == OpCode::CreateHandle) {
    cls = cast<ConstantInt>(call->getArgOperand(1))->getZExtValue();
    auto id = cast<ConstantInt>(call->getArgOperand(2))->getZExtValue();
    if (cls != unsigned(ResourceClass::SRV) && cls != unsigned(ResourceClass::UAV) && cls != unsigned(ResourceClass::Sampler)) return UINT32_MAX;
    const DxilResourceBase &r = cls == unsigned(ResourceClass::Sampler) ?
        static_cast<const DxilResourceBase &>(dm.GetSampler(id)) : cls == unsigned(ResourceClass::SRV) ? dm.GetSRV(id) : dm.GetUAV(id);
    space = r.GetSpaceID(); first = r.GetLowerBound(); end = first + r.GetRangeSize();
    index = call->getArgOperand(3);
  } else if (Opcode(call) == OpCode::CreateHandleFromBinding) {
    auto r = resource_helper::loadBindingFromConstant(*cast<Constant>(call->getArgOperand(1)));
    cls = r.resourceClass; space = r.spaceID; first = r.rangeLowerBound;
    end = uint64_t(r.rangeUpperBound) + 1; index = call->getArgOperand(2);
  } else return UINT32_MAX;
  if (auto *constant = dyn_cast<ConstantInt>(index)) {
    auto reg = constant->getZExtValue();
    if (reg < first || reg >= end) return UINT32_MAX;
    first = reg; end = reg + 1;
  }
  uint32_t result = 0;
  for (size_t i = 0; i < count; ++i) {
    const auto &r = policy[i];
    if (r.resource_class == cls && r.space == space &&
        uint64_t(r.base_register) < end && first < uint64_t(r.base_register) + r.count)
      result |= r.requirements;
  }
  return result;
}

class Lowering {
  DxilModule &dm;
  const ScorpiTypedBufferBinding *bindings;
  size_t count;
  uint32_t space, resourceID;
  uint32_t descriptorResourceIDs[2];
  bool minmaxFiltering;
  bool nativeTextures;
  bool bindingPolicy;
  const ScorpiSamplingRequirement *policy;
  size_t policyCount;
  std::map<Function *, Value *> metadataHandles;
  std::map<Function *, Value *> pointHandles;
  uint32_t pointID = UINT32_MAX;
  std::map<std::pair<Value *, unsigned>, Value *> words;

  CallInst *Call(IRBuilder<> &b, OpCode opcode, Type *type, std::vector<Value *> args) {
    args.insert(args.begin(), b.getInt32(unsigned(opcode)));
    return b.CreateCall(dm.GetOP()->GetOpFunc(opcode, type), args);
  }
  Value *MetadataHandle(Function *f) {
    auto found = metadataHandles.find(f);
    if (found != metadataHandles.end()) return found->second;
    IRBuilder<> b(&*f->getEntryBlock().getFirstInsertionPt());
    Value *handle;
    auto &resource = dm.GetSRV(resourceID);
    if (dm.GetShaderModel()->GetMinor() >= 6) {
      auto binding = resource_helper::loadBindingFromResourceBase(&resource);
      auto *value = resource_helper::getAsConstant(binding, dm.GetOP()->GetResourceBindingType(), *dm.GetShaderModel());
      auto *raw = Call(b, OpCode::CreateHandleFromBinding, b.getVoidTy(), {value, b.getInt32(0), b.getFalse()});
      auto props = resource_helper::loadPropsFromResourceBase(&resource);
      auto *properties = resource_helper::getAsConstant(props, dm.GetOP()->GetResourcePropertiesType(), *dm.GetShaderModel());
      handle = Call(b, OpCode::AnnotateHandle, b.getVoidTy(), {raw, properties});
    } else {
      handle = Call(b, OpCode::CreateHandle, b.getVoidTy(),
          {b.getInt8(unsigned(ResourceClass::SRV)), b.getInt32(resourceID), b.getInt32(0), b.getFalse()});
    }
    metadataHandles[f] = handle;
    return handle;
  }
  Value *Load(IRBuilder<> &b, Value *handle, Value *offset) {
    return b.CreateExtractValue(Call(b, OpCode::BufferLoad, b.getInt32Ty(),
        {handle, offset, UndefValue::get(b.getInt32Ty())}), 0);
  }
  Value *PointSampler(Function *fn) {
    if(pointHandles.count(fn))return pointHandles[fn];
    if(pointID==UINT32_MAX){
      auto sampler=std::make_unique<DxilSampler>();
      sampler->SetID(dm.GetSamplers().size());sampler->SetKind(ResourceKind::Sampler);
      sampler->SetSamplerKind(SamplerKind::Default);sampler->SetSpaceID(space);
      sampler->SetLowerBound(0);sampler->SetRangeSize(1);sampler->SetGlobalName("ScorpiCubePointSampler");
      sampler->SetGlobalSymbol(UndefValue::get(StructType::create(fn->getContext(),{Type::getInt32Ty(fn->getContext())},"ScorpiCubePointSampler")->getPointerTo()));
      pointID=dm.AddSampler(std::move(sampler));
    }
    IRBuilder<> b(&*fn->getEntryBlock().getFirstInsertionPt());
    auto &sampler=dm.GetSampler(pointID);Value *handle;
    if(dm.GetShaderModel()->GetMinor()>=6){
      auto binding=resource_helper::loadBindingFromResourceBase(&sampler);
      auto *value=resource_helper::getAsConstant(binding,dm.GetOP()->GetResourceBindingType(),*dm.GetShaderModel());
      auto *raw=Call(b,OpCode::CreateHandleFromBinding,b.getVoidTy(),{value,b.getInt32(0),b.getFalse()});
      auto props=resource_helper::loadPropsFromResourceBase(&sampler);
      handle=Call(b,OpCode::AnnotateHandle,b.getVoidTy(),{raw,resource_helper::getAsConstant(props,dm.GetOP()->GetResourcePropertiesType(),*dm.GetShaderModel())});
    }else handle=Call(b,OpCode::CreateHandle,b.getVoidTy(),{b.getInt8(unsigned(ResourceClass::Sampler)),b.getInt32(pointID),b.getInt32(0),b.getFalse()});
    return pointHandles[fn]=handle;
  }
public:
  Lowering(DxilModule &module, const ScorpiTypedBufferBinding *ranges,
      size_t n, uint32_t s, uint32_t self, uint32_t options,
      const ScorpiSamplingRequirement *p, size_t pn) : dm(module), bindings(ranges), count(n), space(s),
      minmaxFiltering(!(options & (SCORPI_SHADER_NO_MINMAX_FILTERING | SCORPI_SHADER_NATIVE_TEXTURES))),
      nativeTextures(options & SCORPI_SHADER_NATIVE_TEXTURES),
      bindingPolicy(options & SCORPI_SHADER_BINDING_POLICY), policy(p), policyCount(pn) {
    (void)self; // ABI v3 never subtracts GPU addresses.
    auto &ctx = dm.GetModule()->getContext();
    auto resource = std::make_unique<DxilResource>();
    resource->SetID(dm.GetSRVs().size()); resource->SetRW(false);
    resource->SetKind(ResourceKind::RawBuffer); resource->SetSpaceID(space);
    resource->SetLowerBound(0); resource->SetRangeSize(1);
    resource->SetGlobalName("ScorpiTypedBufferMetadata");
    resource->SetGlobalSymbol(UndefValue::get(StructType::create(ctx,
        {Type::getInt32Ty(ctx)}, "ScorpiTypedBufferMetadata")->getPointerTo()));
    resourceID = dm.AddSRV(std::move(resource));
    for (unsigned i = 0; i != 2; ++i) {
      auto descriptors = std::make_unique<DxilResource>();
      descriptors->SetID(dm.GetSRVs().size()); descriptors->SetRW(false);
      descriptors->SetKind(ResourceKind::RawBuffer); descriptors->SetSpaceID(space);
      descriptors->SetLowerBound(i ? SCORPI_METADATA_SAMPLER_REGISTER : SCORPI_METADATA_RESOURCE_REGISTER);
      descriptors->SetRangeSize(i ? SCORPI_METADATA_SAMPLER_DESCRIPTORS : SCORPI_METADATA_RESOURCE_DESCRIPTORS);
      descriptors->SetGlobalName(i ? "ScorpiSamplerMetadata" : "ScorpiResourceMetadata");
      descriptors->SetGlobalSymbol(UndefValue::get(StructType::create(ctx,
          {Type::getInt32Ty(ctx)}, "ScorpiDescriptorMetadata")->getPointerTo()));
      descriptorResourceIDs[i] = dm.AddSRV(std::move(descriptors));
    }
  }

  Value *DescriptorMetadataHandle(IRBuilder<> &b, Value *slot, bool sampler) {
    auto &resource = dm.GetSRV(descriptorResourceIDs[sampler]);
    auto *index = b.CreateAdd(slot, b.getInt32(resource.GetLowerBound()));
    if (dm.GetShaderModel()->GetMinor() >= 6) {
      auto binding = resource_helper::loadBindingFromResourceBase(&resource);
      auto *value = resource_helper::getAsConstant(binding, dm.GetOP()->GetResourceBindingType(), *dm.GetShaderModel());
      auto *raw = Call(b, OpCode::CreateHandleFromBinding, b.getVoidTy(), {value, index, b.getTrue()});
      auto props = resource_helper::loadPropsFromResourceBase(&resource);
      return Call(b, OpCode::AnnotateHandle, b.getVoidTy(), {raw,
          resource_helper::getAsConstant(props, dm.GetOP()->GetResourcePropertiesType(), *dm.GetShaderModel())});
    }
    return Call(b, OpCode::CreateHandle, b.getVoidTy(), {b.getInt8(unsigned(ResourceClass::SRV)),
        b.getInt32(resource.GetID()), index, b.getTrue()});
  }

  // Word 7 returns the metadata buffer handle, used for view-local feedback.
  Value *Word(Value *handle, unsigned word) {
    auto key = std::make_pair(handle, word);
    auto found = words.find(key);
    if (found != words.end()) return found->second;
    if (auto *phi = dyn_cast<PHINode>(handle)) {
      auto *value = PHINode::Create(word == 7 ? dm.GetOP()->GetHandleType() : Type::getInt32Ty(phi->getContext()), phi->getNumIncomingValues(),
          "scorpi.view.prefix", phi);
      words[key] = value;
      for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i)
        value->addIncoming(Word(phi->getIncomingValue(i), word), phi->getIncomingBlock(i));
      return value;
    }
    if (auto *select = dyn_cast<SelectInst>(handle)) {
      auto *a = Word(select->getTrueValue(), word), *b = Word(select->getFalseValue(), word);
      IRBuilder<> builder(select->getNextNode());
      return words[key] = builder.CreateSelect(select->getCondition(), a, b);
    }
    auto *call = dyn_cast<CallInst>(handle);
    if (!call) throw std::runtime_error("unsupported typed-buffer handle");
    if (Opcode(call) == OpCode::AnnotateHandle) return words[key] = Word(call->getArgOperand(1), word);
    uint32_t cls, resourceSpace, lower;
    Value *index;
    if (Opcode(call) == OpCode::CreateHandle) {
      cls = cast<ConstantInt>(call->getArgOperand(1))->getZExtValue();
      unsigned id = cast<ConstantInt>(call->getArgOperand(2))->getZExtValue();
      const DxilResourceBase &resource = cls == unsigned(ResourceClass::Sampler) ?
          static_cast<const DxilResourceBase &>(dm.GetSampler(id)) :
          cls == unsigned(ResourceClass::SRV) ? dm.GetSRV(id) : dm.GetUAV(id);
      resourceSpace = resource.GetSpaceID(); lower = resource.GetLowerBound(); index = call->getArgOperand(3);
    } else if (Opcode(call) == OpCode::CreateHandleFromBinding) {
      auto binding = resource_helper::loadBindingFromConstant(*cast<Constant>(call->getArgOperand(1)));
      cls = binding.resourceClass; resourceSpace = binding.spaceID; lower = binding.rangeLowerBound;
      index = call->getArgOperand(2);
    } else throw std::runtime_error("unsupported typed-buffer binding (direct heaps require a separate ABI)");
    // The selected native variant proves that all reachable SRV textures are
    // non-sparse and have zero resource MinLOD. Carry that proof into accesses
    // retained for instruction clamps or status consumers: otherwise their
    // dead software filter bodies survive into the Metal compiler. Do not
    // apply this to UAVs (not covered by the native SRV binding policy).
    if (cls == unsigned(ResourceClass::SRV) && (word == 0 || word == 4)) {
      std::set<Value *> seen;
      const uint32_t needed = word == 0 ? SCORPI_SAMPLING_SPARSE : SCORPI_SAMPLING_RESOURCE_MIN_LOD;
      if (bindingPolicy ? !(SamplingRequirements(dm, handle, policy, policyCount, seen) & needed) : nativeTextures)
        return words[key] = ConstantInt::get(Type::getInt32Ty(call->getContext()), 0);
    }
    const ScorpiTypedBufferBinding *range = nullptr;
    for (size_t i = 0; i < count; ++i) {
      const auto &r = bindings[i];
      if (cls == r.resource_class && resourceSpace == r.space && lower >= r.base_register &&
          uint64_t(lower) - r.base_register < r.count) { range = &r; break; }
    }
    if (!range) throw std::runtime_error("typed buffer not covered by a visible descriptor-table range");
    auto *metadata = MetadataHandle(call->getParent()->getParent());
    IRBuilder<> b(call->getNextNode());
    Value *offset = b.getInt32(range->feedback_offset);
    if (range->table_offset != UINT32_MAX) {
      auto *base = Load(b, metadata, offset);
      auto *slot = b.CreateAdd(base, b.CreateAdd(b.CreateSub(index, b.getInt32(range->base_register)),
          b.getInt32(range->table_offset)));
      metadata = DescriptorMetadataHandle(b, slot, cls == unsigned(ResourceClass::Sampler));
      offset = b.getInt32(0);
    }
    if (word == 7) return words[key] = metadata;
    return words[key] = Load(b, metadata, b.CreateAdd(offset, b.getInt32(word * 4)));
  }

  void Rewrite(CallInst *call, bool mayBeRGB32, bool extendedPrefix, bool nativeAddressing) {
    auto *handle = call->getArgOperand(1);
    auto *word = Word(handle, 5);
    IRBuilder<> b(call);
    Value *prefix = b.CreateAnd(word, b.getInt32(255), "scorpi.view.prefix");
    Value *scale = b.getInt32(1), *extra = b.getInt32(0);
    if (mayBeRGB32) scale = b.CreateSelect(b.CreateICmpNE(b.CreateAnd(word,b.getInt32(1u<<30)),b.getInt32(0)),b.getInt32(3),b.getInt32(1));
    if (extendedPrefix) {
      auto *high = b.CreateAnd(b.CreateLShr(word,b.getInt32(16)),b.getInt32(0x3fff));
      extra = b.CreateMul(high,b.CreateSelect(b.CreateICmpEQ(scale,b.getInt32(3)),b.getInt32(1),b.getInt32(256)));
      prefix = b.CreateAdd(prefix,b.CreateMul(extra,scale));
    }
    if (Opcode(call) == OpCode::GetDimensions) {
      b.SetInsertPoint(call->getNextNode());
      auto *width = cast<Instruction>(b.CreateExtractValue(call, 0));
      // RGB32 is represented by three scalar texture-buffer texels. The
      // private native addressing switch does not fix GetDimensions.
      Value *length = b.CreateSub(width, prefix);
      if (mayBeRGB32) {
        length = b.CreateUDiv(length, scale);
      }
      auto *fixed = cast<Instruction>(b.CreateInsertValue(call, length, 0));
      call->replaceAllUsesWith(fixed);
      width->setOperand(0, call); fixed->setOperand(0, call);
      return;
    }
    unsigned indexOperand = Opcode(call) == OpCode::AtomicBinOp ? 3 : 2;
    auto *index = call->getArgOperand(indexOperand);
    auto *dimensions = Call(b, OpCode::GetDimensions, b.getInt32Ty(), {handle, b.getInt32(0)});
    auto *length = b.CreateUDiv(b.CreateSub(b.CreateExtractValue(dimensions, 0), prefix),scale);
    auto *valid = b.CreateICmpULT(index, length);
    // Only a reachable extended sparse prefix adds this small index fixup.
    // Ordinary accesses and native loads/stores/atomics retain their bodies.
    auto *bounded = b.CreateSelect(valid, index, length);
    call->setArgOperand(indexOperand, b.CreateAdd(bounded, nativeAddressing ? extra : prefix));
  }

  Value *TextureResident(IRBuilder<> &b,Value *texture,Value *mip,Value *x,Value *y,Value *slice) {
    auto *offset=Word(texture,0);auto *sparse=b.CreateICmpNE(offset,b.getInt32(0));
    auto *handle=Word(texture,7);
    auto read=[&](Value *word)->Value*{
      auto *address=b.CreateSelect(sparse,b.CreateAdd(offset,b.CreateMul(word,b.getInt32(4))),b.getInt32(0));
      return Load(b,handle,address);
    };
    auto constant=[&](unsigned word){return read(b.getInt32(word));};
    auto safeDivisor=[&](Value *v){return b.CreateSelect(b.CreateICmpEQ(v,b.getInt32(0)),b.getInt32(1),v);};
    auto *validMip=b.CreateICmpULT(mip,constant(0));
    auto *record=b.CreateAdd(b.getInt32(8),b.CreateMul(b.CreateSelect(validMip,mip,b.getInt32(0)),b.getInt32(5)));
    auto field=[&](unsigned n){return read(b.CreateAdd(record,b.getInt32(n)));};
    auto *valid=b.CreateAnd(validMip,b.CreateAnd(b.CreateICmpULT(x,field(2)),b.CreateAnd(b.CreateICmpULT(y,field(3)),b.CreateICmpULT(slice,constant(5)))));
    auto *tile=b.CreateAdd(field(0),b.CreateAdd(b.CreateUDiv(x,safeDivisor(constant(1))),b.CreateMul(b.CreateUDiv(y,safeDivisor(constant(2))),field(1))));
    tile=b.CreateAdd(tile,b.CreateMul(b.CreateAdd(slice,constant(6)),constant(3)));
    tile=b.CreateSelect(b.CreateAnd(sparse,valid),tile,b.getInt32(0));
    auto *bits=read(b.CreateAdd(constant(4),b.CreateUDiv(tile,b.getInt32(32))));
    auto *mapped=b.CreateICmpNE(b.CreateAnd(b.CreateLShr(bits,b.CreateAnd(tile,b.getInt32(31))),b.getInt32(1)),b.getInt32(0));
    mapped=b.CreateSelect(b.CreateICmpEQ(field(4),b.getInt32(UINT32_MAX)),mapped,b.CreateICmpNE(field(4),b.getInt32(0)));
    return b.CreateOr(b.CreateNot(sparse),b.CreateOr(b.CreateNot(valid),mapped));
  }

  static void ReplaceStatus(CallInst *call, Value *mapped) {
    IRBuilder<> b(call);
    std::vector<ExtractValueInst*> statuses;
    for(auto *user:call->users())if(auto *extract=dyn_cast<ExtractValueInst>(user))
      if(extract->getNumIndices()==1 && *extract->idx_begin()==4)statuses.push_back(extract);
    for(auto *extract:statuses){extract->replaceAllUsesWith(b.CreateZExt(mapped,b.getInt32Ty()));extract->eraseFromParent();}
  }

  void RewriteBufferStatus(CallInst *call) {
    bool used=false;for(auto *u:call->users())if(auto *e=dyn_cast<ExtractValueInst>(u))used|=e->getNumIndices()==1&&*e->idx_begin()==4;
    if(!used)return;
    auto *buffer=call->getArgOperand(1);auto *offset=Word(buffer,6);IRBuilder<> b(call);
    auto *sparse=b.CreateICmpNE(offset,b.getInt32(0));auto *metadata=Word(buffer,7);
    auto read=[&](Value *word){return Load(b,metadata,b.CreateSelect(sparse,b.CreateAdd(offset,b.CreateMul(word,b.getInt32(4))),b.getInt32(0)));};
    auto word=[&](unsigned n){return read(b.getInt32(n));};
    auto u64=[&](Value *v){return b.CreateZExt(v,b.getInt64Ty());};
    auto pair=[&](unsigned n){return b.CreateOr(u64(word(n)),b.CreateShl(u64(word(n+1)),b.getInt64(32)));};
    auto *size=pair(0),*viewOffset=pair(2);std::set<Value*> seen;auto kind=Kind(buffer,dm,seen);
    Value *position=u64(call->getArgOperand(2));
    if(kind==ResourceKind::TypedBuffer)position=b.CreateMul(position,u64(word(4)));
    if(kind==ResourceKind::StructuredBuffer){
      // Root descriptors have no stride in their descriptor metadata. DXIL
      // carries the structured element size for both table and root handles.
      seen.clear();unsigned stride=StructuredStride(buffer,dm,seen);
      if(!stride)throw std::runtime_error("missing structured-buffer feedback stride");
      position=b.CreateMul(position,b.getInt64(stride));
    }
    if(kind==ResourceKind::StructuredBuffer&&!isa<UndefValue>(call->getArgOperand(3)))position=b.CreateAdd(position,u64(call->getArgOperand(3)));
    unsigned mask=0;
    if(Opcode(call)==OpCode::RawBufferLoad)mask=cast<ConstantInt>(call->getArgOperand(4))->getZExtValue();
    else for(auto *u:call->users())if(auto *e=dyn_cast<ExtractValueInst>(u))if(e->getNumIndices()==1&&*e->idx_begin()<4)mask|=1u<<*e->idx_begin();
    if(!mask)mask=1;
    const unsigned componentBytes=call->getType()->getStructElementType(0)->getPrimitiveSizeInBits()/8;
    Value *mapped=b.getTrue();
    for(unsigned c=0;c<(kind==ResourceKind::TypedBuffer?1u:4u);++c){
      if(kind!=ResourceKind::TypedBuffer&&!(mask&(1u<<c)))continue;
      auto *start=b.CreateAdd(position,b.getInt64(c*componentBytes));
      Value *bytes=kind==ResourceKind::TypedBuffer?u64(word(4)):b.getInt64(componentBytes);
      auto *valid=b.CreateAnd(b.CreateICmpULT(start,size),b.CreateICmpULE(bytes,b.CreateSub(size,start)));
      auto *absolute=b.CreateAdd(viewOffset,start),*end=b.CreateAdd(absolute,b.CreateSub(bytes,b.getInt64(1)));
      // Each returned component/typed texel spans at most two 64-KiB pages.
      for(auto *address:{absolute,end}){
        auto *tile=b.CreateTrunc(b.CreateLShr(address,b.getInt64(16)),b.getInt32Ty());
        tile=b.CreateSelect(b.CreateAnd(sparse,valid),tile,b.getInt32(0));
        auto *bits=read(b.CreateAdd(b.getInt32(8),b.CreateLShr(tile,b.getInt32(5))));
        auto *present=b.CreateICmpNE(b.CreateAnd(b.CreateLShr(bits,b.CreateAnd(tile,b.getInt32(31))),b.getInt32(1)),b.getInt32(0));
        mapped=b.CreateAnd(mapped,b.CreateOr(b.CreateNot(b.CreateAnd(sparse,valid)),present));
      }
    }
    ReplaceStatus(call,mapped);
  }

  Value *Compare(IRBuilder<> &b,Value *state,Value *reference,Value *value) {
    auto *function=b.CreateAnd(b.CreateLShr(state,b.getInt32(24)),b.getInt32(15));
    Value *tests[]={b.getFalse(),b.getFalse(),b.CreateFCmpOLT(reference,value),b.CreateFCmpOEQ(reference,value),
      b.CreateFCmpOLE(reference,value),b.CreateFCmpOGT(reference,value),b.CreateFCmpUNE(reference,value),b.CreateFCmpOGE(reference,value),b.getTrue()};
    Value *result=b.getFalse();for(unsigned i=2;i<=8;++i)result=b.CreateSelect(b.CreateICmpEQ(function,b.getInt32(i)),tests[i],result);
    return b.CreateUIToFP(result,value->getType());
  }

  void RewriteTextureLoad(CallInst *call) {
    auto *texture = call->getArgOperand(1);
    std::vector<ExtractValueInst*> statuses;
    for (auto *user : call->users())
      if (auto *extract = dyn_cast<ExtractValueInst>(user))
        if (extract->getNumIndices() == 1 && *extract->idx_begin() == 4 && !extract->use_empty())
          statuses.push_back(extract);
    std::set<Value*> seen;
    const auto requirements = bindingPolicy ?
        SamplingRequirements(dm, texture, policy, policyCount, seen) :
        (nativeTextures ? 0u : UINT32_MAX);
    if(isa<UndefValue>(call->getArgOperand(2))){
      if (statuses.empty()) return;
      IRBuilder<> b(call);seen.clear();auto kind=Kind(texture,dm,seen);
      ReplaceStatus(call,TextureResident(b,texture,b.getInt32(0),call->getArgOperand(3),call->getArgOperand(4),
          kind==ResourceKind::Texture2DArray?call->getArgOperand(5):b.getInt32(0)));
      return;
    }
    IRBuilder<> b(call);
    // Sparse residency and resource MinLOD are independent requirements.
    // A status-only rewrite must not alter the application's texture address.
    if (requirements & SCORPI_SAMPLING_RESOURCE_MIN_LOD) {
      auto *minimum = Word(texture,4);
      auto *clamp = b.CreateBitCast(minimum,b.getFloatTy());
      auto *dims = Call(b,OpCode::GetDimensions,b.getInt32Ty(),{texture,b.getInt32(0)});
      auto *last = b.CreateUIToFP(b.CreateSub(b.CreateExtractValue(dims,3),b.getInt32(1)),b.getFloatTy());
      auto *first = Call(b,OpCode::Round_ni,b.getFloatTy(),{clamp});
      auto *mip = b.CreateUIToFP(call->getArgOperand(2),b.getFloatTy());
      auto *valid = b.CreateAnd(b.CreateFCmpOLE(clamp,last),b.CreateFCmpOGE(mip,first));
      // Preserve the native out-of-bounds format defaults and SRV swizzle.
      Value *outside = b.getInt32(UINT32_MAX);
      if (!isa<UndefValue>(call->getArgOperand(6)))
        outside = b.CreateSub(outside,call->getArgOperand(6));
      call->setArgOperand(3,b.CreateSelect(valid,call->getArgOperand(3),outside));
    }
    // Do not generate page-table reads whose result the shader never consumes.
    if (statuses.empty()) return;
    seen.clear();auto kind=Kind(texture,dm,seen);
    Value *coord[3];for(unsigned i=0;i<3;++i){coord[i]=call->getArgOperand(3+i);if(isa<UndefValue>(coord[i]))coord[i]=b.getInt32(0);if(!isa<UndefValue>(call->getArgOperand(6+i)))coord[i]=b.CreateAdd(coord[i],call->getArgOperand(6+i));}
    auto *resident=TextureResident(b,texture,call->getArgOperand(2),coord[0],coord[1],kind==ResourceKind::Texture2DArray?coord[2]:b.getInt32(0));
    for(auto *extract:statuses){extract->replaceAllUsesWith(b.CreateZExt(resident,b.getInt32Ty()));extract->eraseFromParent();}
  }

  void RewriteCubeSample(CallInst *call,ResourceKind kind) {
    std::set<Value *> seen;
    const bool minmaxFiltering = bindingPolicy ?
        (SamplingRequirements(dm, call->getArgOperand(2), policy, policyCount, seen) & SCORPI_SAMPLING_REDUCTION) != 0 : this->minmaxFiltering;
    std::vector<User*> originalUsers(call->user_begin(),call->user_end());
    const auto opcode=Opcode(call);
    const bool implicit=Implicit(opcode),gather=Gather(opcode),comparison=Comparison(opcode);
    auto *texture=call->getArgOperand(1),*sampler=call->getArgOperand(2);
    auto *state=Word(sampler,2),*minimum=Word(sampler,5),*maximum=Word(sampler,3),*bias=Word(sampler,4);
    auto *resourceMinimum=Word(texture,4),*mapping=Word(texture,5);
    // Metal's comparison-cube corner ordering differs from D3D when one of
    // the contributing faces is unmapped. Retain only that required fallback.
    const bool comparisonSample = opcode == OpCode::SampleCmp || opcode == OpCode::SampleCmpLevelZero;
    const bool preciseClamp = NeedsClampPrecision(call);
    const bool nativeColor = !preciseClamp && !comparisonSample && !minmaxFiltering && isa<ConstantInt>(resourceMinimum) && cast<ConstantInt>(resourceMinimum)->isZero();
    auto *low=Word(texture,2),*high=Word(texture,3);
    auto *fn=call->getParent()->getParent();auto *point=PointSampler(fn);
    auto *footprintAnchor=call->getPrevNode();
    IRBuilder<> b(call);
    auto f=[&](double n){return ConstantFP::get(b.getFloatTy(),n);};
    auto unary=[&](OpCode op,Value *v){return Call(b,op,b.getFloatTy(),{v});};
    auto min=[&](Value *a,Value *z){return Call(b,OpCode::FMin,b.getFloatTy(),{a,z});};
    auto max=[&](Value *a,Value *z){return Call(b,OpCode::FMax,b.getFloatTy(),{a,z});};
    // Face basis: normal, horizontal and vertical, in D3D's +X,-X,+Y,-Y,+Z,-Z order.
    const int basis[6][3][3]={{{1,0,0},{0,0,-1},{0,-1,0}},{{-1,0,0},{0,0,1},{0,-1,0}},
      {{0,1,0},{1,0,0},{0,0,1}},{{0,-1,0},{1,0,0},{0,0,-1}},
      {{0,0,1},{1,0,0},{0,-1,0}},{{0,0,-1},{-1,0,0},{0,-1,0}}};
    auto basisValue=[&](Value *face,unsigned vector,unsigned axis)->Value*{
      Value *v=f(basis[5][vector][axis]);for(int i=4;i>=0;--i)v=b.CreateSelect(b.CreateICmpEQ(face,b.getInt32(i)),f(basis[i][vector][axis]),v);return v;
    };
    auto faceOf=[&](Value **d)->Value*{
      Value *ax=unary(OpCode::FAbs,d[0]),*ay=unary(OpCode::FAbs,d[1]),*az=unary(OpCode::FAbs,d[2]);
      auto *z=b.CreateAnd(b.CreateFCmpOGE(az,ax),b.CreateFCmpOGE(az,ay));auto *y=b.CreateFCmpOGE(ay,ax);
      auto *axis=b.CreateSelect(z,b.getInt32(2),b.CreateSelect(y,b.getInt32(1),b.getInt32(0)));
      auto *major=b.CreateSelect(z,d[2],b.CreateSelect(y,d[1],d[0]));
      return b.CreateAdd(b.CreateMul(axis,b.getInt32(2)),b.CreateZExt(b.CreateFCmpOLT(major,f(0)),b.getInt32Ty()));
    };
    auto dot=[&](Value *face,unsigned vector,Value **d)->Value*{
      Value *v=f(0);for(unsigned a=0;a<3;++a)v=b.CreateFAdd(v,b.CreateFMul(basisValue(face,vector,a),d[a]));return v;
    };
    auto direction=[&](Value *face,Value *s,Value *t,Value **d){for(unsigned a=0;a<3;++a)d[a]=b.CreateFAdd(basisValue(face,0,a),b.CreateFAdd(b.CreateFMul(s,basisValue(face,1,a)),b.CreateFMul(t,basisValue(face,2,a))));};
    Value *d[3]={call->getArgOperand(3),call->getArgOperand(4),call->getArgOperand(5)};
    auto *face=faceOf(d);auto *ma=max(dot(face,0,d),f(1e-20));auto *sc=dot(face,1,d),*tc=dot(face,2,d);
    Value *uv[2]={b.CreateFMul(b.CreateFAdd(b.CreateFDiv(sc,ma),f(1)),f(.5)),b.CreateFMul(b.CreateFAdd(b.CreateFDiv(tc,ma),f(1)),f(.5))};
    auto *dims=Call(b,OpCode::GetDimensions,b.getInt32Ty(),{texture,b.getInt32(0)});
    Value *lod=(gather||comparison)?f(0):call->getArgOperand(10);
    if(opcode==OpCode::SampleGrad || implicit){
      Value *length[2];auto *scale=b.CreateFMul(b.CreateUIToFP(b.CreateExtractValue(dims,0),b.getFloatTy()),f(.5));
      for(unsigned axis=0;axis<2;++axis){
        Value *g[3];for(unsigned c=0;c<3;++c)g[c]=implicit?static_cast<Value*>(unary(axis?OpCode::DerivCoarseY:OpCode::DerivCoarseX,d[c])):call->getArgOperand(10+axis*3+c);
        auto *dm=dot(face,0,g);Value *grad[2];Value *numerators[2]={sc,tc};
        for(unsigned c=0;c<2;++c)grad[c]=b.CreateFMul(b.CreateFDiv(b.CreateFSub(b.CreateFMul(dot(face,c+1,g),ma),b.CreateFMul(numerators[c],dm)),b.CreateFMul(ma,ma)),scale);
        length[axis]=b.CreateFAdd(b.CreateFMul(grad[0],grad[0]),b.CreateFMul(grad[1],grad[1]));
      }
      // Cube anisotropy may use trilinear quality (D3D functional spec 7.18.11).
      lod=b.CreateFMul(unary(OpCode::Log,max(max(length[0],length[1]),f(1e-30))),f(.5));
      if(opcode==OpCode::SampleBias)lod=b.CreateFAdd(lod,call->getArgOperand(10));
    }
    const auto footprint=FootprintInstructions(footprintAnchor,call);
    auto *before=call->getParent();auto *after=before->splitBasicBlock(call->getNextNode(),"scorpi.cube.end");
    auto *normal=BasicBlock::Create(call->getContext(),"scorpi.cube.native",fn,after);
    auto *reduce=BasicBlock::Create(call->getContext(),"scorpi.cube.reduce",fn,after);
    auto *present=BasicBlock::Create(call->getContext(),"scorpi.cube.present",fn,after);
    auto *missing=BasicBlock::Create(call->getContext(),"scorpi.cube.null",fn,after);
    before->getTerminator()->eraseFromParent();
    if(!implicit&&!gather&&!nativeColor){call->removeFromParent();normal->getInstList().push_back(call);}
    b.SetInsertPoint(before);b.CreateCondBr(b.CreateICmpNE(b.CreateOr(low,high),b.getInt32(0)),present,missing);
    b.SetInsertPoint(missing);b.CreateBr(after);
    b.SetInsertPoint(normal);b.CreateBr(after);
    b.SetInsertPoint(present);
    Value *ordinary=(!minmaxFiltering||comparison||gather)?b.getTrue():b.CreateICmpEQ(b.CreateAnd(state,b.getInt32(0x180)),b.getInt32(0));
    auto *resMin=b.CreateBitCast(resourceMinimum,b.getFloatTy());Value *accessMin=resMin;
    if(ClampOperand(opcode)){unsigned at=opcode==OpCode::SampleGrad?16:(opcode==OpCode::SampleBias||opcode==OpCode::SampleCmp)?11:10;if(!isa<UndefValue>(call->getArgOperand(at)))accessMin=max(accessMin,call->getArgOperand(at));}
    auto *viewDims=Call(b,OpCode::GetDimensions,b.getInt32Ty(),{texture,b.getInt32(0)});
    auto *last=b.CreateUIToFP(b.CreateSub(b.CreateExtractValue(viewDims,3),b.getInt32(1)),b.getFloatTy());
    auto *empty=b.CreateFCmpOGT(accessMin,last);
    if(gather)empty=b.CreateOr(empty,b.CreateFCmpOGE(accessMin,f(1)));
    Value *precision = b.getFalse();
    if (preciseClamp) { auto *scaled=b.CreateFMul(accessMin,f(16)); precision=b.CreateAnd(b.CreateFCmpOGT(accessMin,f(0)),b.CreateFCmpONE(scaled,unary(OpCode::Round_ni,scaled))); }
    b.CreateCondBr(b.CreateOr(precision,b.CreateOr(b.CreateAnd(ordinary,b.CreateOr(b.CreateFCmpOGT(resMin,f(0)),b.CreateICmpNE(Word(texture,0),b.getInt32(0)))),minmaxFiltering?b.CreateICmpNE(b.CreateAnd(state,b.getInt32(0x100)),b.getInt32(0)):b.getFalse())),reduce,normal);
    SinkFootprint(footprint,reduce);
    b.SetInsertPoint(reduce);
    lod=b.CreateFAdd(lod,b.CreateBitCast(bias,b.getFloatTy()));
    lod=min(max(max(max(min(lod,b.CreateBitCast(maximum,b.getFloatTy())),b.CreateBitCast(minimum,b.getFloatTy())),accessMin),f(0)),last);
    if(gather)lod=f(0);
    Value *mipLinear=gather?b.getFalse():b.CreateICmpNE(b.CreateAnd(state,b.getInt32(1)),b.getInt32(0));
    Value *linear=gather?b.getTrue():b.CreateICmpNE(b.CreateAnd(state,b.CreateSelect(b.CreateFCmpOGT(lod,f(0)),b.getInt32(16),b.getInt32(4))),b.getInt32(0));
    auto *isMin=b.CreateICmpEQ(b.CreateAnd(state,b.getInt32(0x180)),b.getInt32(0x100));
    auto *base=unary(OpCode::Round_ni,lod);auto *mf=b.CreateSelect(mipLinear,b.CreateFSub(lod,base),f(0));
    auto *mip0=b.CreateFPToUI(b.CreateSelect(mipLinear,base,unary(OpCode::Round_ni,b.CreateFAdd(lod,f(.5)))),b.getInt32Ty());
    auto *more=b.CreateFCmpOGT(mf,f(0));
    auto *element=call->getType()->getStructElementType(0);
    auto combine=[&](Value *a,Value *z)->Value*{return b.CreateSelect(isMin,Call(b,OpCode::FMin,element,{a,z}),Call(b,OpCode::FMax,element,{a,z}));};
    Value *result=UndefValue::get(call->getType());
    for(unsigned m=0;m<(gather?1u:2u);++m){
      auto *level=m?b.CreateAdd(mip0,b.CreateZExt(more,b.getInt32Ty())):mip0;
      auto *size=Call(b,OpCode::GetDimensions,b.getInt32Ty(),{texture,level});auto *n=b.CreateUIToFP(b.CreateExtractValue(size,0),b.getFloatTy());
      Value *start[2],*fraction[2];for(unsigned a=0;a<2;++a){auto *coord=b.CreateFSub(b.CreateFMul(uv[a],n),b.CreateSelect(linear,f(.5),f(0)));start[a]=unary(OpCode::Round_ni,coord);fraction[a]=b.CreateSelect(linear,b.CreateFSub(coord,start[a]),f(0));}
      // Read exact texel centers through a private point sampler. Guest
      // addressing, mip bias and clamps must not be applied a second time.
      auto sample=[&](Value *sampleFace,Value **dir)->Value*{
        auto *denom=max(dot(sampleFace,0,dir),f(1e-20));Value *st[2],*indices[2];
        for(unsigned a=0;a<2;++a){auto *coord=b.CreateFMul(b.CreateFAdd(b.CreateFDiv(dot(sampleFace,a+1,dir),denom),f(1)),b.CreateFMul(n,f(.5)));
          auto *index=min(max(unary(OpCode::Round_ni,coord),f(0)),b.CreateFSub(n,f(1)));indices[a]=b.CreateFPToUI(index,b.getInt32Ty());st[a]=b.CreateFSub(b.CreateFDiv(b.CreateFMul(b.CreateFAdd(index,f(.5)),f(2)),n),f(1));}
        Value *center[3];direction(sampleFace,st[0],st[1],center);
        Value *layer=kind==ResourceKind::TextureCubeArray?call->getArgOperand(6):UndefValue::get(b.getFloatTy());
        Value *value=(gather||nativeColor)?static_cast<Value*>(UndefValue::get(call->getType())):Call(b,OpCode::SampleLevel,element,{texture,point,center[0],center[1],center[2],layer,
          UndefValue::get(b.getInt32Ty()),UndefValue::get(b.getInt32Ty()),UndefValue::get(b.getInt32Ty()),b.CreateUIToFP(level,b.getFloatTy())});
        Value *slice=sampleFace;
        if(kind==ResourceKind::TextureCubeArray){auto *lastCube=b.CreateUIToFP(b.CreateSub(b.CreateExtractValue(dims,2),b.getInt32(1)),b.getFloatTy());auto *cube=b.CreateFPToUI(min(max(unary(OpCode::Round_ne,layer),f(0)),lastCube),b.getInt32Ty());slice=b.CreateAdd(slice,b.CreateMul(cube,b.getInt32(6)));}
        return b.CreateInsertValue(value,b.CreateZExt(TextureResident(b,texture,level,indices[0],indices[1],slice),b.getInt32Ty()),4);
      };
      {
        SampleTapLoop taps(b,result,4,after);
        auto *tap=taps.index;
        Value *index[2],*outside[2],*st[2],*sign[2];Value *weight=m?static_cast<Value*>(mf):b.CreateFSub(f(1),mf);
        for(unsigned a=0;a<2;++a){auto *highTap=b.CreateICmpNE(b.CreateAnd(tap,b.getInt32(1u<<a)),b.getInt32(0));index[a]=b.CreateFAdd(start[a],b.CreateSelect(highTap,f(1),f(0)));outside[a]=b.CreateOr(b.CreateFCmpOLT(index[a],f(0)),b.CreateFCmpOGE(index[a],n));
          sign[a]=b.CreateSelect(b.CreateFCmpOLT(index[a],f(0)),f(-1),f(1));
          st[a]=b.CreateSelect(outside[a],sign[a],b.CreateFSub(b.CreateFDiv(b.CreateFMul(b.CreateFAdd(index[a],f(.5)),f(2)),n),f(1)));
          weight=b.CreateFMul(weight,b.CreateSelect(highTap,fraction[a],b.CreateFSub(f(1),fraction[a])));}
        Value *active=gather?b.getTrue():b.CreateFCmpOGT(weight,f(0));auto *corner=b.CreateAnd(outside[0],outside[1]);
        Value *edge[3],*nx[3],*ny[3];direction(face,st[0],st[1],edge);
        for(unsigned a=0;a<3;++a){nx[a]=b.CreateFMul(sign[0],basisValue(face,1,a));ny[a]=b.CreateFMul(sign[1],basisValue(face,2,a));}
        auto *fx=faceOf(nx),*fy=faceOf(ny);auto *selected=b.CreateSelect(corner,face,b.CreateSelect(outside[0],fx,b.CreateSelect(outside[1],fy,face)));
        auto *v=sample(selected,edge),*x=sample(fx,edge),*y=sample(fy,edge);
        for(unsigned c=0;c<4;++c){
          if(gather||nativeColor){result=b.CreateInsertValue(result,b.CreateExtractValue(call,c),c);continue;}
          Value *value=b.CreateExtractValue(v,c),*xc=b.CreateExtractValue(x,c),*yc=b.CreateExtractValue(y,c);
          if(comparison){value=Compare(b,state,call->getArgOperand(10),b.CreateExtractValue(v,0));xc=Compare(b,state,call->getArgOperand(10),b.CreateExtractValue(x,0));yc=Compare(b,state,call->getArgOperand(10),b.CreateExtractValue(y,0));}
          auto *cornerValue=b.CreateSelect(ordinary,b.CreateFDiv(b.CreateFAdd(value,b.CreateFAdd(xc,yc)),f(3)),combine(value,combine(xc,yc)));
          value=b.CreateSelect(corner,cornerValue,value);
          auto *weighted=b.CreateFMul(value,weight);
          auto *first=b.CreateSelect(ordinary,weighted,value);
          auto *old=b.CreateExtractValue(result,c);value=b.CreateSelect(active,b.CreateSelect(ordinary,b.CreateFAdd(old,weighted),combine(old,value)),old);
          if(!m)value=b.CreateSelect(b.CreateICmpEQ(tap,b.getInt32(0)),first,value);
          result=b.CreateInsertValue(result,value,c);
        }
        auto *resident=b.CreateAnd(b.CreateExtractValue(v,4),b.CreateSelect(corner,b.CreateAnd(b.CreateExtractValue(x,4),b.CreateExtractValue(y,4)),b.getInt32(1)));
        auto *old=b.CreateExtractValue(result,4);
        result=b.CreateInsertValue(result,b.CreateAnd(old,b.CreateSelect(active,resident,b.getInt32(1))),4);
        taps.End(b,result);
      }
    }
    auto emptyValue = [&](IRBuilder<> &builder, unsigned component) -> Value * {
      if(gather)component=cast<ConstantInt>(call->getArgOperand(9))->getZExtValue();
      auto *selector = builder.CreateAnd(builder.CreateLShr(mapping,builder.getInt32(component*3)),builder.getInt32(7));
      Value *one = element->isIntegerTy() ? static_cast<Value *>(ConstantInt::get(element,1)) : ConstantFP::get(element,1);
      auto *value = builder.CreateSelect(builder.CreateICmpEQ(selector,builder.getInt32(5)),one,Constant::getNullValue(element));
      return comparison ? Compare(builder,state,call->getArgOperand(10),value) : value;
    };
    Value *emptyComparison = comparison ? emptyValue(b,0) : nullptr;
    for(unsigned c=0;c<4;++c){auto *defaultValue=comparison?emptyComparison:emptyValue(b,c);result=b.CreateInsertValue(result,b.CreateSelect(empty,defaultValue,b.CreateExtractValue(result,c)),c);}
    auto *reducedExit=b.GetInsertBlock();
    b.CreateBr(after);
    b.SetInsertPoint(reducedExit->getTerminator());
    IRBuilder<> normalResult(normal->getTerminator()),merged(&*after->begin());Value *assembled=UndefValue::get(call->getType());
    Value *normalEmptyComparison = comparison ? emptyValue(normalResult,0) : nullptr;
    for(unsigned c=0;c<5;++c){auto *type=call->getType()->getStructElementType(c);auto *phi=PHINode::Create(type,3,"scorpi.cube.component",&*after->begin());
      Value *normalValue=normalResult.getInt32(1);
      if(c<4){auto *defaultValue=comparison?normalEmptyComparison:emptyValue(normalResult,c);
        normalValue=normalResult.CreateSelect(empty,defaultValue,normalResult.CreateExtractValue(call,c));}
      phi->addIncoming(normalValue,normal);phi->addIncoming(c==4?b.CreateSelect(empty,b.getInt32(1),b.CreateExtractValue(result,c)):b.CreateExtractValue(result,c),reducedExit);phi->addIncoming(c==4?static_cast<Value*>(b.getInt32(1)):Constant::getNullValue(type),missing);assembled=merged.CreateInsertValue(assembled,phi,c);}
    for(auto *user:originalUsers)user->replaceUsesOfWith(call,assembled);
  }

  // Software reduction uses integer texture reads, so every addressing mode,
  // zero-weight exclusion and mip clamp is independent of native sampler
  // reduction support. Unclamped ordinary sampling keeps the original instruction.
  void RewriteSample(CallInst *call, ResourceKind kind) {
    std::set<Value *> policySeen;
    const bool minmaxFiltering = bindingPolicy ?
        (SamplingRequirements(dm, call->getArgOperand(2), policy, policyCount, policySeen) & SCORPI_SAMPLING_REDUCTION) != 0 : this->minmaxFiltering;
    if(kind==ResourceKind::TextureCube || kind==ResourceKind::TextureCubeArray){RewriteCubeSample(call,kind);return;}
    std::vector<User*> originalUsers(call->user_begin(),call->user_end());
    const auto opcode = Opcode(call);
    const bool implicit = Implicit(opcode),gather=Gather(opcode),comparison=Comparison(opcode);
    auto *texture = call->getArgOperand(1), *sampler = call->getArgOperand(2);
    auto *state = Word(sampler, 2), *minimum = Word(sampler, 5), *maximum = Word(sampler, 3);
    auto *bias = Word(sampler, 4);
    auto *resourceMinimum = Word(texture, 4);
    const bool preciseClamp = NeedsClampPrecision(call);
    const bool nativeColor = !preciseClamp && !minmaxFiltering && isa<ConstantInt>(resourceMinimum) && cast<ConstantInt>(resourceMinimum)->isZero();
    auto *borderMapping = Word(texture, 5);
    auto *textureLow = Word(texture, 2), *textureHigh = Word(texture, 3);
    auto *footprintAnchor = call->getPrevNode();
    const unsigned axes = kind == ResourceKind::Texture3D ? 3 : 2;
    Value *sourceLod = (gather||comparison)?ConstantFP::get(Type::getFloatTy(call->getContext()),0):call->getArgOperand(10);
    Value *anisoSteps = nullptr, *anisoDirection[2] = {};
    if (opcode == OpCode::SampleGrad || implicit) {
      IRBuilder<> b(call);
      auto *size = Call(b, OpCode::GetDimensions, b.getInt32Ty(), {texture,b.getInt32(0)});
      Value *dx2 = ConstantFP::get(b.getFloatTy(),0), *dy2 = dx2;
      Value *gradientX[3], *gradientY[3];
      for (unsigned axis=0;axis<axes;++axis) {
        auto *n = b.CreateUIToFP(b.CreateExtractValue(size,axis),b.getFloatTy());
        // Derivatives must execute before the sampler-mode branch: descriptor
        // indexing can make that branch non-uniform within a pixel quad.
        auto *gx = implicit ? Call(b,OpCode::DerivCoarseX,b.getFloatTy(),{call->getArgOperand(3+axis)}) : call->getArgOperand(10+axis);
        auto *gy = implicit ? Call(b,OpCode::DerivCoarseY,b.getFloatTy(),{call->getArgOperand(3+axis)}) : call->getArgOperand(13+axis);
        auto *dx = b.CreateFMul(gx,n);
        auto *dy = b.CreateFMul(gy,n);
        gradientX[axis]=dx; gradientY[axis]=dy;
        dx2 = b.CreateFAdd(dx2,b.CreateFMul(dx,dx));
        dy2 = b.CreateFAdd(dy2,b.CreateFMul(dy,dy));
      }
      auto *rho2 = Call(b,OpCode::FMax,b.getFloatTy(),{dx2,dy2});
      sourceLod = b.CreateFMul(Call(b,OpCode::Log,b.getFloatTy(),{rho2}),ConstantFP::get(b.getFloatTy(),.5));
      if (axes == 2) {
        auto f = [&](double v) { return ConstantFP::get(b.getFloatTy(),v); };
        auto min = [&](Value *a,Value *z) -> Value * { return Call(b,OpCode::FMin,b.getFloatTy(),{a,z}); };
        auto max = [&](Value *a,Value *z) -> Value * { return Call(b,OpCode::FMax,b.getFloatTy(),{a,z}); };
        auto sqrt = [&](Value *v) -> Value * { return Call(b,OpCode::Sqrt,b.getFloatTy(),{v}); };
        // Principal axes of J*J^T, not merely the longer screen derivative:
        // skewed footprints need orthogonalization before choosing mip/taps.
        Value *g[4]={gradientX[0],gradientX[1],gradientY[0],gradientY[1]};
        for(auto &v:g)v=min(max(v,f(-65536)),f(65536));
        auto *xx=b.CreateFAdd(b.CreateFMul(g[0],g[0]),b.CreateFMul(g[2],g[2]));
        auto *yy=b.CreateFAdd(b.CreateFMul(g[1],g[1]),b.CreateFMul(g[3],g[3]));
        auto *xy=b.CreateFAdd(b.CreateFMul(g[0],g[1]),b.CreateFMul(g[2],g[3]));
        auto *difference=b.CreateFSub(xx,yy);
        auto *delta=sqrt(b.CreateFAdd(b.CreateFMul(difference,difference),b.CreateFMul(f(4),b.CreateFMul(xy,xy))));
        auto *major2=max(b.CreateFMul(b.CreateFAdd(b.CreateFAdd(xx,yy),delta),f(.5)),f(1e-30));
        auto *major=sqrt(major2);
        auto *limit=b.CreateUIToFP(b.CreateAdd(b.CreateAnd(b.CreateLShr(state,b.getInt32(20)),b.getInt32(15)),b.getInt32(1)),b.getFloatTy());
        auto *minor=max(sqrt(max(b.CreateFMul(b.CreateFSub(b.CreateFAdd(xx,yy),delta),f(.5)),f(0))),b.CreateFDiv(major,limit));
        auto *enabled=b.CreateICmpNE(b.CreateAnd(state,b.getInt32(0x40)),b.getInt32(0));
        auto *ratio=min(limit,max(f(1),b.CreateFDiv(major,max(minor,f(1)))));
        // sqrt/log approximations must not turn an exact 4:1 footprint into
        // five probes. This tolerance is much finer than D3D's 8-bit weights.
        anisoSteps=b.CreateFPToUI(b.CreateSelect(enabled,Call(b,OpCode::Round_pi,b.getFloatTy(),{max(f(1),b.CreateFSub(ratio,f(1e-5)))}),f(1)),b.getInt32Ty());
        sourceLod=b.CreateSelect(enabled,Call(b,OpCode::Log,b.getFloatTy(),{minor}),sourceLod);
        Value *vx=b.CreateFSub(major2,yy), *vy=xy;
        auto *norm=sqrt(b.CreateFAdd(b.CreateFMul(vx,vx),b.CreateFMul(vy,vy)));
        auto *nonzero=b.CreateFCmpOGT(norm,f(1e-15));
        vx=b.CreateSelect(nonzero,b.CreateFDiv(vx,max(norm,f(1e-15))),f(0));
        vy=b.CreateSelect(nonzero,b.CreateFDiv(vy,max(norm,f(1e-15))),f(1));
        Value *direction[2]={vx,vy};
        for(unsigned axis=0;axis<2;++axis)
          anisoDirection[axis]=b.CreateFDiv(b.CreateFMul(direction[axis],major),
              max(b.CreateUIToFP(b.CreateExtractValue(size,axis),b.getFloatTy()),f(1)));
      }
      if (opcode == OpCode::SampleBias) sourceLod = b.CreateFAdd(sourceLod,call->getArgOperand(10));
    }
    const auto footprint = FootprintInstructions(footprintAnchor, call);
    auto *before = call->getParent();
    auto *fn = before->getParent();
    auto *after = before->splitBasicBlock(call->getNextNode(), "scorpi.sample.end");
    auto *normal = BasicBlock::Create(call->getContext(), "scorpi.sample.native", fn, after);
    auto *reduce = BasicBlock::Create(call->getContext(), "scorpi.sample.reduce", fn, after);
    auto *present = BasicBlock::Create(call->getContext(), "scorpi.sample.present", fn, after);
    auto *missing = BasicBlock::Create(call->getContext(), "scorpi.sample.null", fn, after);
    auto *bounded = BasicBlock::Create(call->getContext(), "scorpi.sample.bounded", fn, after);
    auto *empty = BasicBlock::Create(call->getContext(), "scorpi.sample.empty", fn, after);
    before->getTerminator()->eraseFromParent();
    // Keep native implicit derivatives in the original control-flow context.
    if (!implicit&&!gather&&!nativeColor) { call->removeFromParent(); normal->getInstList().push_back(call); }
    IRBuilder<> nb(normal); nb.CreateBr(after);
    IRBuilder<> pre(before);
    // A null SRV must not enter footprint math (dimensions are zero, which
    // would otherwise lead to integer modulo/division by zero).
    pre.CreateCondBr(pre.CreateICmpNE(pre.CreateOr(textureLow,textureHigh),pre.getInt32(0)),present,missing);
    IRBuilder<> zero(missing); zero.CreateBr(after);
    pre.SetInsertPoint(present);
    auto *dims = Call(pre, OpCode::GetDimensions, pre.getInt32Ty(), {texture, pre.getInt32(0)});
    auto *lastMip = pre.CreateUIToFP(pre.CreateSub(pre.CreateExtractValue(dims,3),pre.getInt32(1)),pre.getFloatTy());
    Value *accessMinimum = pre.CreateBitCast(resourceMinimum,pre.getFloatTy());
    if (ClampOperand(opcode)) {
      const unsigned clamp = opcode == OpCode::SampleGrad ? 16 : (opcode == OpCode::SampleBias||opcode==OpCode::SampleCmp) ? 11 : 10;
      if (!isa<UndefValue>(call->getArgOperand(clamp)))
        accessMinimum = Call(pre,OpCode::FMax,pre.getFloatTy(),{accessMinimum,call->getArgOperand(clamp)});
    }
    // Unlike sampler clamps, resource/instruction clamps can empty the view.
    // Even a fractional clamp past its last mip is out of bounds. Use a real
    // out-of-bounds load to retain the view's missing-component defaults.
    auto *emptyView = pre.CreateFCmpOGT(accessMinimum,lastMip);
    // Gathers retain mip 0 for fractional clamps below 1 (D3D11.3 5.8.4),
    // except when the clamp is already beyond the view's last mip (5.8.5).
    if(gather)emptyView=pre.CreateOr(emptyView,pre.CreateFCmpOGE(accessMinimum,ConstantFP::get(pre.getFloatTy(),1)));
    pre.CreateCondBr(emptyView,empty,bounded);
    auto *elementType = call->getType()->getStructElementType(0);
    IRBuilder<> out(empty);
    Value *emptyResult = Call(out,OpCode::TextureLoad,elementType,
        {texture,out.getInt32(0),out.getInt32(UINT32_MAX),out.getInt32(0),out.getInt32(0),
         out.getInt32(0),out.getInt32(0),out.getInt32(0)});
    // Empty-set comparison sampling compares against the OOB texel value;
    // returning that raw value skips the sampler's comparison function.
    // A gather returns its selected channel for all four OOB taps.
    if (comparison || gather) {
      const unsigned component = comparison ? 0 : cast<ConstantInt>(call->getArgOperand(9))->getZExtValue();
      Value *value = out.CreateExtractValue(emptyResult,component);
      if(comparison)value=Compare(out,state,call->getArgOperand(10),value);
      for (unsigned c=0;c<4;++c) emptyResult = out.CreateInsertValue(emptyResult,value,c);
    }
    out.CreateBr(after);
    pre.SetInsertPoint(bounded);
    Value *ordinary = (!minmaxFiltering||gather||comparison)?pre.getTrue():pre.CreateICmpEQ(pre.CreateAnd(state,pre.getInt32(0x180)),pre.getInt32(0));
    // The converter ignores resource MinLOD for SampleLevel. Use the same
    // footprint machinery with weighted taps for clamped sampling.
    // Unclamped ordinary sampling, including its native precision, is intact.
    auto *clampedOrdinary = pre.CreateAnd(ordinary,pre.CreateOr(pre.CreateICmpNE(Word(texture,0),pre.getInt32(0)),
        pre.CreateFCmpOGT(pre.CreateBitCast(resourceMinimum,pre.getFloatTy()),ConstantFP::get(pre.getFloatTy(),0))));
    Value *precision = pre.getFalse();
    if (preciseClamp) {
      auto *scaled = pre.CreateFMul(accessMinimum,ConstantFP::get(pre.getFloatTy(),16));
      precision = pre.CreateAnd(pre.CreateFCmpOGT(accessMinimum,ConstantFP::get(pre.getFloatTy(),0)),
          pre.CreateFCmpONE(scaled,Call(pre,OpCode::Round_ni,pre.getFloatTy(),{scaled})));
    }
    pre.CreateCondBr(pre.CreateOr(precision,pre.CreateOr(clampedOrdinary,
        minmaxFiltering?pre.CreateICmpNE(pre.CreateAnd(state,pre.getInt32(0x100)),pre.getInt32(0)):pre.getFalse())),reduce,normal);
    SinkFootprint(footprint, reduce);
    IRBuilder<> b(reduce);
    auto f = [&](double n) { return ConstantFP::get(b.getFloatTy(), n); };
    auto unary = [&](OpCode op, Value *v) -> Value * { return Call(b, op, b.getFloatTy(), {v}); };
    auto min = [&](Value *a, Value *z) -> Value * { return Call(b, OpCode::FMin, b.getFloatTy(), {a,z}); };
    auto max = [&](Value *a, Value *z) -> Value * { return Call(b, OpCode::FMax, b.getFloatTy(), {a,z}); };
    Value *lod = b.CreateFAdd(sourceLod,b.CreateBitCast(bias,b.getFloatTy()));
    // D3D gives MinLOD precedence even when it exceeds MaxLOD.
    lod = max(min(lod, b.CreateBitCast(maximum, b.getFloatTy())), b.CreateBitCast(minimum, b.getFloatTy()));
    lod = min(max(max(lod, accessMinimum), f(0)), lastMip);
    if(anisoSteps) {
      auto *nearest=unary(OpCode::Round_ne,lod);
      lod=b.CreateSelect(b.CreateFCmpOLT(unary(OpCode::FAbs,b.CreateFSub(lod,nearest)),f(1e-5)),nearest,lod);
    }
    if(gather)lod=f(0);
    Value *mipLinear = gather?b.getFalse():b.CreateICmpNE(b.CreateAnd(state, b.getInt32(1)), b.getInt32(0));
    Value *linear = gather?b.getTrue():b.CreateICmpNE(b.CreateAnd(state,
        b.CreateSelect(b.CreateFCmpOGT(lod, f(0)), b.getInt32(0x10), b.getInt32(4))), b.getInt32(0));
    auto *isMin = b.CreateICmpEQ(b.CreateAnd(state, b.getInt32(0x180)), b.getInt32(0x100));
    auto *base = unary(OpCode::Round_ni, lod);
    auto *mipFraction = b.CreateSelect(mipLinear,b.CreateFSub(lod,base),f(0));
    auto *mip0 = b.CreateFPToUI(b.CreateSelect(mipLinear, base, unary(OpCode::Round_ni, b.CreateFAdd(lod,f(.5)))), b.getInt32Ty());
    auto *useMip1 = b.CreateAnd(mipLinear, b.CreateFCmpONE(lod, base));
    const bool array = kind == ResourceKind::Texture2DArray;
    Value *result = UndefValue::get(call->getType());
    PHINode *iteration=nullptr, *accumulators[5]={};
    BasicBlock *loopHeader=nullptr, *loopExit=nullptr;
    Value *loopResult=nullptr, *anisoPosition=nullptr;
    if (anisoSteps) {
      loopHeader=BasicBlock::Create(call->getContext(),"scorpi.aniso.loop",fn,after);
      auto *body=BasicBlock::Create(call->getContext(),"scorpi.aniso.tap",fn,after);
      loopExit=BasicBlock::Create(call->getContext(),"scorpi.aniso.end",fn,after);
      b.CreateBr(loopHeader);b.SetInsertPoint(loopHeader);
      iteration=b.CreatePHI(b.getInt32Ty(),2,"scorpi.aniso.index");iteration->addIncoming(b.getInt32(0),reduce);
      for(unsigned c=0;c<5;++c) {
        auto *type=call->getType()->getStructElementType(c);
        accumulators[c]=b.CreatePHI(type,2,"scorpi.aniso.value");
        accumulators[c]->addIncoming(c==4?static_cast<Value*>(b.getInt32(1)):Constant::getNullValue(type),reduce);
      }
      for(unsigned c=0;c<5;++c)result=b.CreateInsertValue(result,accumulators[c],c);
      loopResult=result;
      b.CreateCondBr(b.CreateICmpULT(iteration,anisoSteps),body,loopExit);b.SetInsertPoint(body);
      anisoPosition=b.CreateFSub(b.CreateFDiv(b.CreateFAdd(b.CreateUIToFP(iteration,b.getFloatTy()),f(.5)),
          b.CreateUIToFP(anisoSteps,b.getFloatTy())),f(.5));
    }
    auto combine = [&](Value *a, Value *z) -> Value * {
      return b.CreateSelect(isMin, Call(b, OpCode::FMin, elementType, {a,z}), Call(b, OpCode::FMax, elementType, {a,z}));
    };
    for (unsigned mip = 0; mip < (gather?1u:2u); ++mip) {
      auto *level = mip == 0 ? mip0 : b.CreateAdd(mip0, b.CreateZExt(useMip1, b.getInt32Ty()));
      auto *size = Call(b, OpCode::GetDimensions, b.getInt32Ty(), {texture, level});
      Value *length[3], *start[3], *upper[3], *fraction[3];
      for (unsigned axis = 0; axis < axes; ++axis) {
        length[axis] = b.CreateExtractValue(size, axis);
        Value *normalized=call->getArgOperand(3+axis);
        if(anisoPosition)normalized=b.CreateFAdd(normalized,b.CreateFMul(anisoPosition,anisoDirection[axis]));
        auto *coord = b.CreateFMul(normalized, b.CreateUIToFP(length[axis], b.getFloatTy()));
        coord = b.CreateFSub(coord, b.CreateSelect(linear, f(.5), f(0)));
        auto *floor = unary(OpCode::Round_ni, coord);
        fraction[axis] = b.CreateSelect(linear,b.CreateFSub(coord,floor),f(0));
        start[axis] = b.CreateFPToSI(floor, b.getInt32Ty());
        if (!isa<UndefValue>(call->getArgOperand(7+axis)))
          start[axis] = b.CreateAdd(start[axis], call->getArgOperand(7+axis));
        upper[axis] = b.CreateAnd(linear, b.CreateFCmpONE(coord, floor));
      }
      {
        SampleTapLoop taps(b,result,1u << axes,after);
        auto *tap=taps.index;
        Value *active = mip == 0 ? b.getTrue() : useMip1;
        Value *weight = mip == 0 ? b.CreateFSub(f(1),mipFraction) : mipFraction;
        if(anisoSteps)weight=b.CreateFDiv(weight,b.CreateUIToFP(anisoSteps,b.getFloatTy()));
        Value *outside = b.getFalse();
        Value *coords[3] = {UndefValue::get(b.getInt32Ty()),UndefValue::get(b.getInt32Ty()),UndefValue::get(b.getInt32Ty())};
        for (unsigned axis = 0; axis < axes; ++axis) {
          auto *high = b.CreateICmpNE(b.CreateAnd(tap,b.getInt32(1u << axis)),b.getInt32(0));
          weight = b.CreateFMul(weight,b.CreateSelect(high,fraction[axis],b.CreateFSub(f(1),fraction[axis])));
          if (!gather) active = b.CreateAnd(active, b.CreateOr(b.CreateNot(high),upper[axis]));
          auto *index = b.CreateAdd(start[axis],b.CreateZExt(high,b.getInt32Ty()));
          auto *n = length[axis];
          auto *last = b.CreateSub(n,b.getInt32(1));
          auto clamp = [&](Value *v) -> Value * {
            v = b.CreateSelect(b.CreateICmpSLT(v,b.getInt32(0)),b.getInt32(0),v);
            return b.CreateSelect(b.CreateICmpSGT(v,last),last,v);
          };
          auto wrap = [&](Value *v, Value *period) -> Value * {
            auto *r = b.CreateSRem(v,period);
            return b.CreateSelect(b.CreateICmpSLT(r,b.getInt32(0)),b.CreateAdd(r,period),r);
          };
          auto *mode = b.CreateAnd(b.CreateLShr(state,b.getInt32(9+axis*3)),b.getInt32(7));
          auto *period = b.CreateMul(n,b.getInt32(2));
          auto *mirror = wrap(index,period);
          mirror = b.CreateSelect(b.CreateICmpSGE(mirror,n),b.CreateSub(b.CreateSub(period,b.getInt32(1)),mirror),mirror);
          auto *once = clamp(b.CreateSelect(b.CreateICmpSLT(index,b.getInt32(0)),b.CreateSub(b.getInt32(-1),index),index));
          coords[axis] = b.CreateSelect(b.CreateICmpEQ(mode,b.getInt32(1)),wrap(index,n),
              b.CreateSelect(b.CreateICmpEQ(mode,b.getInt32(2)),mirror,
              b.CreateSelect(b.CreateICmpEQ(mode,b.getInt32(5)),once,clamp(index))));
          outside = b.CreateOr(outside,b.CreateAnd(b.CreateICmpEQ(mode,b.getInt32(4)),b.CreateICmpUGE(index,n)));
        }
        if (array) {
          auto *layer = b.CreateFPToSI(unary(OpCode::Round_ne,call->getArgOperand(5)),b.getInt32Ty());
          auto *last = b.CreateSub(b.CreateExtractValue(size,2),b.getInt32(1));
          layer = b.CreateSelect(b.CreateICmpSLT(layer,b.getInt32(0)),b.getInt32(0),layer);
          coords[2] = b.CreateSelect(b.CreateICmpSGT(layer,last),last,layer);
        }
        Value *loaded = (gather || nativeColor) ? static_cast<Value *>(UndefValue::get(call->getType())) : Call(b,OpCode::TextureLoad,elementType,
            {texture,level,coords[0],coords[1],coords[2],b.getInt32(0),b.getInt32(0),b.getInt32(0)});
        auto *border = b.CreateAnd(b.CreateLShr(state,b.getInt32(18)),b.getInt32(3));
        for (unsigned c = 0; c < 4; ++c) {
          if(gather||nativeColor){result=b.CreateInsertValue(result,b.CreateExtractValue(call,c),c);continue;}
          auto *selector = b.CreateAnd(b.CreateLShr(borderMapping,b.getInt32(c*3)),b.getInt32(7));
          auto *colorOne = b.CreateOr(b.CreateICmpEQ(border,b.getInt32(2)),
              b.CreateAnd(b.CreateICmpEQ(border,b.getInt32(1)),b.CreateICmpEQ(selector,b.getInt32(3))));
          auto *isOne = b.CreateOr(b.CreateICmpEQ(selector,b.getInt32(5)),
              b.CreateAnd(b.CreateICmpULT(selector,b.getInt32(4)),colorOne));
          auto *borderValue = b.CreateSelect(isOne,
              ConstantFP::get(elementType,1),ConstantFP::get(elementType,0));
          auto *value = b.CreateSelect(outside,borderValue,b.CreateExtractValue(loaded,c));
          if(comparison)value=Compare(b,state,call->getArgOperand(10),b.CreateSelect(outside,borderValue,b.CreateExtractValue(loaded,0)));
          auto *weighted = b.CreateFMul(value,b.CreateFPCast(weight,elementType));
          auto *first=b.CreateSelect(ordinary,weighted,value);
          auto *old=b.CreateExtractValue(result,c);
          value=b.CreateSelect(active,b.CreateSelect(ordinary,b.CreateFAdd(old,weighted),combine(old,value)),old);
          if (!mip) {
            Value *initial=b.CreateICmpEQ(tap,b.getInt32(0));
            if(iteration) initial=b.CreateAnd(initial,b.CreateICmpEQ(iteration,b.getInt32(0)));
            value=b.CreateSelect(initial,first,value);
          }
          result = b.CreateInsertValue(result,value,c);
        }
        auto *resident=TextureResident(b,texture,level,coords[0],coords[1],array?coords[2]:b.getInt32(0));
        auto *contributes=b.CreateAnd(active,b.CreateNot(outside));
        Value *old=b.CreateExtractValue(result,4);
        result=b.CreateInsertValue(result,b.CreateAnd(old,b.CreateZExt(b.CreateOr(b.CreateNot(contributes),resident),b.getInt32Ty())),4);
        taps.End(b,result);
      }
    }
    if(iteration) {
      auto *body=b.GetInsertBlock();
      iteration->addIncoming(b.CreateAdd(iteration,b.getInt32(1)),body);
      for(unsigned c=0;c<5;++c)accumulators[c]->addIncoming(b.CreateExtractValue(result,c),body);
      b.CreateBr(loopHeader);b.SetInsertPoint(loopExit);result=loopResult;
    }
    auto *reducedExit=b.GetInsertBlock();
    b.CreateBr(after);
    // Keep PHIs scalar. The converter's AIR serializer cannot reliably encode
    // an aggregate sample-result PHI with a zero/undef constant incoming edge.
    IRBuilder<> nativeResult(normal->getTerminator()), reducedResult(reducedExit->getTerminator()), outResult(empty->getTerminator());
    IRBuilder<> merged(&*after->begin());
    Value *assembled = UndefValue::get(call->getType());
    for (unsigned c=0;c<5;++c) {
      auto *type = call->getType()->getStructElementType(c);
      auto *phi = PHINode::Create(type,4,"scorpi.sample.component",&*after->begin());
      phi->addIncoming(c==4?nativeResult.getInt32(1):nativeResult.CreateExtractValue(call,c),normal);
      phi->addIncoming(reducedResult.CreateExtractValue(nativeColor && c < 4 ? static_cast<Value *>(call) : result,c),reducedExit);
      phi->addIncoming(c==4?static_cast<Value*>(b.getInt32(1)):Constant::getNullValue(type),missing);
      phi->addIncoming(c==4?outResult.getInt32(1):outResult.CreateExtractValue(emptyResult,c),empty);
      assembled = merged.CreateInsertValue(assembled,phi,c);
    }
    for (auto *user : originalUsers) user->replaceUsesOfWith(call,assembled);
  }
};
}

// Root buffer descriptors have no view length and cannot use GetDimensions.
// Only table/heap descriptors carry the bounds required by this correction.
static bool RawHandleHasBounds(DxilModule &dm, Value *handle,
    const ScorpiTypedBufferBinding *bindings, size_t count, std::set<Value *> &seen) {
  if (!seen.insert(handle).second) return true;
  if (auto *phi = dyn_cast<PHINode>(handle)) {
    for (auto &v : phi->incoming_values())
      if (!RawHandleHasBounds(dm, v, bindings, count, seen)) return false;
    return true;
  }
  if (auto *select = dyn_cast<SelectInst>(handle))
    return RawHandleHasBounds(dm, select->getTrueValue(), bindings, count, seen) &&
        RawHandleHasBounds(dm, select->getFalseValue(), bindings, count, seen);
  auto *call = dyn_cast<CallInst>(handle);
  if (!call) return false;
  if (Opcode(call) == OpCode::AnnotateHandle)
    return RawHandleHasBounds(dm, call->getArgOperand(1), bindings, count, seen);
  if (Opcode(call) == OpCode::CreateHandleFromHeap) return true;
  unsigned cls, space, lower;
  if (Opcode(call) == OpCode::CreateHandle) {
    cls = cast<ConstantInt>(call->getArgOperand(1))->getZExtValue();
    unsigned id = cast<ConstantInt>(call->getArgOperand(2))->getZExtValue();
    const auto &resource = cls == unsigned(ResourceClass::SRV) ? dm.GetSRV(id) : dm.GetUAV(id);
    space = resource.GetSpaceID(); lower = resource.GetLowerBound();
  } else if (Opcode(call) == OpCode::CreateHandleFromBinding) {
    auto binding = resource_helper::loadBindingFromConstant(*cast<Constant>(call->getArgOperand(1)));
    cls = binding.resourceClass; space = binding.spaceID; lower = binding.rangeLowerBound;
  } else return false;
  for (size_t i = 0; i < count; ++i) {
    const auto &range = bindings[i];
    if (cls == range.resource_class && space == range.space && lower >= range.base_register &&
        uint64_t(lower) - range.base_register < range.count)
      return range.table_offset != UINT32_MAX;
  }
  return false;
}

// Native buffer bounds checks cover the base address, but not the trailing
// components of a vector raw load. Preserve that native load and mask only
// live trailing results; scalar loads and all other resource kinds stay native.
static unsigned RawLoadComponentMask(CallInst *call) {
  unsigned mask = 0;
  std::vector<User *> users(call->user_begin(), call->user_end());
  for (auto *user : users) {
    auto *extract = dyn_cast<ExtractValueInst>(user);
    if (!extract || extract->getNumIndices() != 1) { mask = 14; break; }
    unsigned component = *extract->idx_begin();
    if (component > 0 && component < 4 && !extract->use_empty())
      mask |= 1u << component;
  }
  return mask;
}

static void BoundRawLoadComponents(DxilModule &dm, CallInst *call) {
  const unsigned mask = RawLoadComponentMask(call);
  if (!mask) return;
  std::vector<User *> users(call->user_begin(), call->user_end());
  IRBuilder<> b(call->getNextNode());
  auto *dimensions = b.CreateCall(dm.GetOP()->GetOpFunc(OpCode::GetDimensions, b.getInt32Ty()),
      {b.getInt32(unsigned(OpCode::GetDimensions)), call->getArgOperand(1), b.getInt32(0)});
  auto *size = b.CreateExtractValue(dimensions, 0);
  auto *offset = call->getArgOperand(2);
  auto *inRange = b.CreateICmpULT(offset, size);
  auto *remaining = b.CreateSub(size, offset);
  const unsigned bytes = call->getType()->getStructElementType(0)->getPrimitiveSizeInBits() / 8;
  Value *result = call;
  for (unsigned c = 1; c < 4; ++c) if (mask & (1u << c)) {
    auto *value = b.CreateExtractValue(call, c);
    auto *valid = b.CreateAnd(inRange, b.CreateICmpUGE(remaining, b.getInt32((c + 1) * bytes)));
    result = b.CreateInsertValue(result,
        b.CreateSelect(valid, value, Constant::getNullValue(value->getType())), c);
  }
  for (auto *user : users) user->replaceUsesOfWith(call, result);
}

// Metal permits preserving FP32 subnormals; Direct3D's default math does not.
// The renderer opts in after measuring its device. Preserve raw data users,
// share canonical values, and leave explicit preserve/FP16/FP64 untouched.
static bool FlushFp32Math(DxilModule &dm, uint32_t options) {
  // Converted trig may return a tiny input unchanged, bypassing the GPU's
  // arithmetic FTZ behavior. Its inputs need canonicalization on every device.
  struct Math { Instruction *inst; unsigned first, count; bool output; };
  std::vector<Math> math;
  for (auto &f : *dm.GetModule()) {
    if (f.hasFnAttribute("fp32-denorm-mode") &&
        f.getFnAttribute("fp32-denorm-mode").getValueAsString() == "preserve") continue;
    for (auto &block : f) for (auto &inst : block) {
      if ((options & SCORPI_SHADER_FLUSH_FP32_COMPARISON_INPUTS) &&
          isa<FCmpInst>(inst) &&
          inst.getOperand(0)->getType()->getScalarType()->isFloatTy())
        math.push_back({&inst, 0, 2, false});
      const bool trig = Opcode(&inst) == OpCode::Sin || Opcode(&inst) == OpCode::Cos;
      if ((!(options & SCORPI_SHADER_FLUSH_FP32_ARITHMETIC) && !trig) ||
          !inst.getType()->getScalarType()->isFloatTy()) continue;
      switch (inst.getOpcode()) {
      case Instruction::FSub:
        // LLVM represents sign-only negation as -0 - x. DXBC MOV source
        // modifiers preserve subnormals, so leave that bit operation intact.
        if (auto *zero = dyn_cast<ConstantFP>(inst.getOperand(0)))
          if (zero->isZero() && zero->isNegative()) continue;
        LLVM_FALLTHROUGH;
      case Instruction::FAdd: case Instruction::FMul:
      case Instruction::FDiv: case Instruction::FRem:
        math.push_back({&inst, 0, 2, true});
        break;
      case Instruction::Call: {
        const auto op = Opcode(&inst);
        if (op == OpCode::NumOpCodes || op == OpCode::FAbs) break;
        const auto cls = OP::GetOpCodeClass(op);
        if (cls != OpCodeClass::Unary && cls != OpCodeClass::Binary &&
            cls != OpCodeClass::Tertiary && cls != OpCodeClass::Dot2 &&
            cls != OpCodeClass::Dot3 && cls != OpCodeClass::Dot4) break;
        // With flushed inputs these operations cannot produce subnormals.
        const bool bounded = op == OpCode::Round_ne || op == OpCode::Round_ni ||
            op == OpCode::Round_pi || op == OpCode::Round_z || op == OpCode::Frc ||
            op == OpCode::Log || op == OpCode::Sqrt || op == OpCode::Rsqrt ||
            op == OpCode::Saturate || op == OpCode::FMin || op == OpCode::FMax ||
            op == OpCode::Sin || op == OpCode::Cos;
        math.push_back({&inst, 1, cast<CallInst>(&inst)->getNumArgOperands() - 1, !bounded});
        break;
      }
      default: break;
      }
    }
  }
  bool changed = false;
  std::map<Value *, Value *> canonical;
  auto flush = [&](Value *value) -> Value * {
    auto found = canonical.find(value);
    if (found != canonical.end()) return found->second;
    Instruction *position = nullptr;
    if (auto *definition = dyn_cast<Instruction>(value))
      position = isa<PHINode>(definition) ? &*definition->getParent()->getFirstInsertionPt() :
          definition->getNextNode();
    else if (auto *argument = dyn_cast<Argument>(value))
      position = &*argument->getParent()->getEntryBlock().getFirstInsertionPt();
    IRBuilder<> b(dm.GetModule()->getContext());
    if (position) b.SetInsertPoint(position);
    auto *type = value->getType()->isVectorTy() ?
        static_cast<Type *>(VectorType::get(b.getInt32Ty(), value->getType()->getVectorNumElements())) :
        b.getInt32Ty();
    auto *bits = b.CreateBitCast(value, type);
    auto *magnitude = b.CreateAnd(bits, ConstantInt::get(type, 0x7fffffff));
    auto *subnormal = b.CreateICmpULT(magnitude, ConstantInt::get(type, 0x00800000));
    auto *sign = b.CreateAnd(bits, ConstantInt::get(type, 0x80000000));
    auto *result = b.CreateBitCast(b.CreateSelect(subnormal, sign, bits), value->getType());
    canonical.emplace(value, result);
    canonical.emplace(result, result);
    changed |= result != value;
    return result;
  };
  // Canonicalize results first so consumers reuse them, including across PHIs
  // and backedges. Snapshot users before creating the canonicalization itself.
  for (auto &item : math) {
    if (!item.inst->getType()->getScalarType()->isFloatTy()) continue;
    if (!item.output) { canonical.emplace(item.inst, item.inst); continue; }
    std::vector<User *> users(item.inst->user_begin(), item.inst->user_end());
    auto *result = flush(item.inst);
    for (auto *user : users) user->replaceUsesOfWith(item.inst, result);
  }
  for (auto &item : math) for (unsigned i = item.first; i < item.first + item.count; ++i) {
    auto *value = item.inst->getOperand(i);
    if (value->getType()->getScalarType()->isFloatTy())
      item.inst->setOperand(i, flush(value));
  }
  return changed;
}

// Native converted Sin/Cos return zero for nonfinite inputs, including with
// DisableNanInfOptimization. Preserve native finite arithmetic; share one
// integer classification across Sin/Cos using the same SSA input. Integer
// classification remains valid even when the original shader permits fast math.
static bool FixNonfiniteTrig(DxilModule &dm) {
  std::vector<CallInst *> calls;
  for (auto &f : *dm.GetModule()) for (auto &block : f) for (auto &inst : block)
    if (auto *call = dyn_cast<CallInst>(&inst))
      if (call->getType()->isFloatTy() &&
          (Opcode(call) == OpCode::Sin || Opcode(call) == OpCode::Cos))
        calls.push_back(call);
  std::map<Value *, Value *> nonfinite;
  for (auto *call : calls) {
    auto *value = call->getArgOperand(1);
    auto found = nonfinite.find(value);
    if (found == nonfinite.end()) {
      Instruction *position = call;
      if (auto *definition = dyn_cast<Instruction>(value))
        position = isa<PHINode>(definition) ? &*definition->getParent()->getFirstInsertionPt() :
            definition->getNextNode();
      else if (auto *argument = dyn_cast<Argument>(value))
        position = &*argument->getParent()->getEntryBlock().getFirstInsertionPt();
      IRBuilder<> b(position);
      auto *bits = b.CreateBitCast(value, b.getInt32Ty());
      auto *magnitude = b.CreateAnd(bits, b.getInt32(0x7fffffff));
      found = nonfinite.emplace(value, b.CreateICmpUGE(magnitude, b.getInt32(0x7f800000))).first;
    }
    std::vector<User *> users(call->user_begin(), call->user_end());
    IRBuilder<> b(call->getNextNode());
    auto *nan = ConstantFP::get(dm.GetModule()->getContext(), APFloat::getQNaN(APFloat::IEEEsingle));
    auto *result = b.CreateSelect(found->second, nan, call);
    for (auto *user : users) user->replaceUsesOfWith(call, result);
  }
  return !calls.empty();
}

// The native converter treats IMul's high half as unsigned and aborts on UMul.
// Carry/borrow instructions are also unsupported; use add/sub and comparisons.
// Express the exact signed/unsigned product using native integer arithmetic.
// DXIL's two-result order is high, low (unlike SPIR-V's low, high).
static bool LowerExtendedIntegerArithmetic(DxilModule &dm) {
  std::vector<CallInst *> calls;
  for (auto &f : *dm.GetModule()) for (auto &block : f) for (auto &inst : block)
    if (auto *call = dyn_cast<CallInst>(&inst))
      if (Opcode(call) == OpCode::IMul || Opcode(call) == OpCode::UMul ||
          Opcode(call) == OpCode::UAddc || Opcode(call) == OpCode::USubb)
        calls.push_back(call);
  for (auto *call : calls) {
    IRBuilder<> b(call);
    auto *a = call->getArgOperand(1), *c = call->getArgOperand(2);
    if (Opcode(call) == OpCode::UAddc || Opcode(call) == OpCode::USubb) {
      const bool add = Opcode(call) == OpCode::UAddc;
      auto *value = add ? b.CreateAdd(a, c) : b.CreateSub(a, c);
      auto *flag = b.CreateICmpULT(add ? value : a, add ? a : c);
      auto *pair = b.CreateInsertValue(UndefValue::get(call->getType()), value, 0);
      call->replaceAllUsesWith(b.CreateInsertValue(pair, flag, 1));
      call->eraseFromParent();
      continue;
    }
    const bool isSigned = Opcode(call) == OpCode::IMul;
    auto extend = [&](Value *value) {
      return isSigned ? b.CreateSExt(value, b.getInt64Ty()) :
          b.CreateZExt(value, b.getInt64Ty());
    };
    auto *product = b.CreateMul(extend(a), extend(c));
    auto *high = b.CreateTrunc(b.CreateLShr(product, b.getInt64(32)), b.getInt32Ty());
    auto *low = b.CreateTrunc(product, b.getInt32Ty());
    auto *pair = b.CreateInsertValue(UndefValue::get(call->getType()), high, 0);
    call->replaceAllUsesWith(b.CreateInsertValue(pair, low, 1));
    call->eraseFromParent();
  }
  if (!calls.empty()) {
    // Eliminate unused halves and fold low-only products back to i32.
    legacy::PassManager cleanup;
    cleanup.add(createInstructionCombiningPass());
    cleanup.run(*dm.GetModule());
    dm.CollectShaderFlagsForModule();
  }
  return !calls.empty();
}

bool ScorpiInlineDxbcSubroutines(DxilModule &dm) {
  std::vector<Function *> subroutines;
  for (auto &function : *dm.GetModule())
    if (function.hasLocalLinkage() && !function.isDeclaration() &&
        function.getName().startswith("dx.label."))
      subroutines.push_back(&function);
  const bool inlinedSubroutines = !subroutines.empty();
  if (inlinedSubroutines) {
    auto legacyCall = [](Value *value) {
      auto *call = dyn_cast_or_null<CallInst>(value);
      auto *function = call ? call->getCalledFunction() : nullptr;
      return function && function->hasLocalLinkage() && !function->isDeclaration() &&
          function->getName().startswith("dx.label.") ? call : nullptr;
    };
    {
      CallGraph graph(*dm.GetModule());
      std::vector<std::pair<WeakTrackingVH, unsigned>> pending;
      for (auto &function : *dm.GetModule())
        if (std::find(subroutines.begin(), subroutines.end(), &function) == subroutines.end())
          for (auto &block : function) for (auto &inst : block)
            if (auto *call = legacyCall(&inst)) pending.emplace_back(call, 0);
      while (!pending.empty()) {
        auto item = pending.back(); pending.pop_back();
        auto *call = legacyCall(item.first);
        if (!call) continue;
        // SM4/5 CALL is a no-op when its 32-entry return stack is full.
        // Expand per call site: a shared helper can be reached at both the
        // boundary and a shallower depth. No runtime depth tracking is needed.
        if (item.second == 32) {
          graph[call->getParent()->getParent()]->removeCallEdgeFor(call);
          call->eraseFromParent();
          continue;
        }
        InlineFunctionInfo info(&graph);
        if (!InlineFunction(call, info, false))
          throw std::runtime_error("cannot inline DXBC subroutine");
        for (auto &nested : info.InlinedCalls)
          if (auto *next = legacyCall(nested)) pending.emplace_back(next, item.second + 1);
      }
    }
    // All reachable call sites are expanded. Drop the original helper bodies
    // together, including references from one now-unused helper to another.
    for (auto *function : subroutines) function->dropAllReferences();
    for (auto *function : subroutines) {
      dm.RemoveFunction(function);
      function->eraseFromParent();
    }
    // DXBC call/label helpers share the thread's temporary-register globals.
    // Inline them before native conversion, then promote that storage to SSA.
    // Ordinary shaders never run these passes.
    legacy::PassManager cleanup;
    cleanup.add(createGlobalOptimizerPass());
    cleanup.add(createPromoteMemoryToRegisterPass());
    cleanup.add(createInstructionCombiningPass());
    cleanup.add(createCFGSimplificationPass());
    cleanup.run(*dm.GetModule());
  }
  return inlinedSubroutines;
}

namespace {
bool WidenMinimumPrecisionIO(DxilModule &dm) {
  // Minimum precision permits a wider interface. MSC's stage-in/position
  // lowering emits invalid AIR for f16 interfaces and invalid pipelines for
  // signed i16 interfaces. Keep the shader's arithmetic precision and convert only at
  // the interface; explicit native 16-bit shaders retain their exact ABI.
  if (!dm.m_ShaderFlags.GetLowPrecisionPresent() ||
      dm.m_ShaderFlags.GetUseNativeLowPrecision()) return false;
  using Types = std::map<unsigned, CompType>;
  Types inputs, outputs, patches;
  auto widen = [](DxilSignature &signature, Types &types) {
    for (auto &element : signature.GetElements()) {
      auto type = element->GetCompType();
      if (!(type == CompType::getF16()) && !(type == CompType::getI16())) continue;
      types.emplace(element->GetID(), type);
      element->SetCompType(type.GetBaseCompType());
    }
  };
  widen(dm.GetInputSignature(), inputs);
  widen(dm.GetOutputSignature(), outputs);
  widen(dm.GetPatchConstOrPrimSignature(), patches);
  if (inputs.empty() && outputs.empty() && patches.empty()) return false;
  std::vector<CallInst *> calls;
  for (auto &f : *dm.GetModule()) for (auto &block : f) for (auto &inst : block)
    if (auto *call = dyn_cast<CallInst>(&inst)) {
      auto op = Opcode(call);
      if (op == OpCode::LoadInput || op == OpCode::LoadOutputControlPoint ||
          op == OpCode::LoadPatchConstant || op == OpCode::StoreOutput ||
          op == OpCode::StorePatchConstant) calls.push_back(call);
    }
  for (auto *call : calls) {
    auto op = Opcode(call);
    const auto &types = op == OpCode::LoadInput ? inputs :
        (op == OpCode::LoadPatchConstant || op == OpCode::StorePatchConstant) ? patches : outputs;
    auto *id = dyn_cast<ConstantInt>(call->getArgOperand(1));
    if (!id) throw std::runtime_error("dynamic minimum-precision signature ID");
    auto selected = types.find(unsigned(id->getZExtValue()));
    if (selected == types.end()) continue;
    IRBuilder<> at(call);
    bool floating = selected->second == CompType::getF16();
    Type *wide = floating ? at.getFloatTy() : at.getInt32Ty();
    SmallVector<Value *, 8> args;
    for (auto &arg : call->arg_operands()) args.push_back(arg);
    const bool store = op == OpCode::StoreOutput || op == OpCode::StorePatchConstant;
    if (store) {
      auto *value = args[4];
      args[4] = floating ? at.CreateFPExt(value, wide) : at.CreateSExt(value, wide);
    }
    auto *replacement = at.CreateCall(dm.GetOP()->GetOpFunc(op, wide), args);
    replacement->setDebugLoc(call->getDebugLoc());
    if (!store) {
      auto *value = floating ? at.CreateFPTrunc(replacement, call->getType()) :
          at.CreateTrunc(replacement, call->getType());
      call->replaceAllUsesWith(value);
    }
    call->eraseFromParent();
  }
  return true;
}

bool RoundMinimumPrecisionConversions(DxilModule &dm) {
  if (!dm.m_ShaderFlags.GetLowPrecisionPresent() ||
      dm.m_ShaderFlags.GetUseNativeLowPrecision()) return false;
  std::vector<FPTruncInst *> casts;
  for (auto &f : *dm.GetModule()) for (auto &block : f) for (auto &inst : block)
    if (auto *cast = dyn_cast<FPTruncInst>(&inst))
      if (cast->getSrcTy()->isFloatTy() && cast->getDestTy()->isHalfTy()) casts.push_back(cast);
  // D3D11.3 3.2.2 and 7.20.3.4 require round-to-zero when narrowing.
  // LLVM fptrunc uses nearest-even; the legacy DXIL conversion carries the
  // required native rounding mode without changing half arithmetic or native16.
  for (auto *cast : casts) {
    IRBuilder<> at(cast);
    auto *bits = at.CreateCall(dm.GetOP()->GetOpFunc(OpCode::LegacyF32ToF16, at.getFloatTy()),
        {at.getInt32(unsigned(OpCode::LegacyF32ToF16)), cast->getOperand(0)});
    auto *value = at.CreateBitCast(at.CreateTrunc(bits, at.getInt16Ty()), at.getHalfTy());
    cast->replaceAllUsesWith(value);
    cast->eraseFromParent();
  }
  return !casts.empty();
}

bool RetypeFlatFloatInputs(DxilModule &dm) {
  // Native flat float varyings may clamp infinities during rasterization.
  // Integer varyings carry the same register bits without interpolation.
  if (!dm.GetShaderModel()->IsPS()) return false;
  std::set<unsigned> ids;
  for (auto &element : dm.GetInputSignature().GetElements()) {
    if (element->GetKind() == SemanticKind::Arbitrary &&
        element->GetCompType() == CompType::getF32() &&
        element->GetInterpolationMode()->IsConstant()) {
      ids.insert(element->GetID());
      element->SetCompType(CompType::getU32());
    }
  }
  if (ids.empty()) return false;
  std::vector<CallInst *> loads;
  for (auto &f : *dm.GetModule()) for (auto &block : f) for (auto &inst : block)
    if (auto *call = dyn_cast<CallInst>(&inst))
      if (Opcode(call) == OpCode::LoadInput) loads.push_back(call);
  for (auto *call : loads) {
    auto *id = dyn_cast<ConstantInt>(call->getArgOperand(1));
    if (!id) throw std::runtime_error("dynamic flat input signature ID");
    if (!ids.count(unsigned(id->getZExtValue()))) continue;
    IRBuilder<> at(call);
    SmallVector<Value *, 8> args;
    for (auto &arg : call->arg_operands()) args.push_back(arg);
    auto *load = at.CreateCall(dm.GetOP()->GetOpFunc(OpCode::LoadInput, at.getInt32Ty()), args);
    load->setDebugLoc(call->getDebugLoc());
    call->replaceAllUsesWith(at.CreateBitCast(load, call->getType()));
    call->eraseFromParent();
  }
  return true;
}

}

bool ScorpiWidenMinimumPrecisionIO(DxilModule &dm) { return WidenMinimumPrecisionIO(dm); }

bool ScorpiLowerTypedBuffers(DxilModule &dm, const ScorpiTypedBufferBinding *bindings,
    size_t count, uint32_t space, uint32_t selfOffset, uint32_t options,
    const ScorpiSamplingRequirement *policy, size_t policyCount) {
  const bool inlinedSubroutines = ScorpiInlineDxbcSubroutines(dm);
  const bool widenedMinimumPrecisionIO = WidenMinimumPrecisionIO(dm);
  const bool roundedMinimumPrecision = RoundMinimumPrecisionConversions(dm);
  const bool flatFloatInputs = (options & SCORPI_SHADER_FLAT_FLOAT_INPUT_BITS) && RetypeFlatFloatInputs(dm);
  std::vector<CallInst *> calls;
  std::vector<std::pair<CallInst *, ResourceKind>> samples;
  std::vector<CallInst *> textureLoads;
  std::vector<CallInst *> bufferLoads;
  std::vector<CallInst *> rawLoads;
  std::vector<CallInst *> feedbackChecks;
  std::vector<CallInst *> nullBufferAccesses;
  const bool loweredIntegerArithmetic = LowerExtendedIntegerArithmetic(dm);
  const bool fixedTrig = FixNonfiniteTrig(dm);
  const bool flushedMath = FlushFp32Math(dm, options);
  for (auto &f : *dm.GetModule()) for (auto &block : f) for (auto &inst : block) {
    auto *call = dyn_cast<CallInst>(&inst);
    const auto op = Opcode(call);
    if ((options & SCORPI_SHADER_BINDING_POLICY) &&
        (op == OpCode::BufferLoad || op == OpCode::RawBufferLoad ||
         op == OpCode::BufferStore || op == OpCode::RawBufferStore ||
         op == OpCode::AtomicBinOp || op == OpCode::AtomicCompareExchange)) {
      std::set<Value *> seen;
      const auto kind = Kind(call->getArgOperand(1), dm, seen);
      seen.clear();
      if ((kind == ResourceKind::RawBuffer || kind == ResourceKind::StructuredBuffer) &&
          (SamplingRequirements(dm, call->getArgOperand(1), policy, policyCount, seen) &
           SCORPI_SAMPLING_NULL_BUFFER)) nullBufferAccesses.push_back(call);
    }
    if ((op == OpCode::BufferLoad || op == OpCode::RawBufferLoad) && RawLoadComponentMask(call)) {
      std::set<Value *> seen;
      if (Kind(call->getArgOperand(1), dm, seen) == ResourceKind::RawBuffer) {
        seen.clear();
        if (RawHandleHasBounds(dm, call->getArgOperand(1), bindings, count, seen))
          rawLoads.push_back(call);
      }
    }
    if(op==OpCode::CheckAccessFullyMapped)feedbackChecks.push_back(call);
    if (op == OpCode::BufferLoad || op == OpCode::RawBufferLoad) bufferLoads.push_back(call);
    if (op == OpCode::TextureLoad) {
      std::set<Value *> seen;
      auto kind = Kind(call->getArgOperand(1),dm,seen);
      if (kind == ResourceKind::Texture2D || kind == ResourceKind::Texture2DArray || kind == ResourceKind::Texture3D)
        textureLoads.push_back(call);
    }
    if (op == OpCode::SampleLevel || op == OpCode::SampleGrad || Implicit(op) || Gather(op) || Comparison(op)) {
      std::set<Value *> seen;
      auto kind = Kind(call->getArgOperand(1), dm, seen);
      if (kind == ResourceKind::Texture2D || kind == ResourceKind::Texture2DArray || kind == ResourceKind::Texture3D || kind == ResourceKind::TextureCube || kind == ResourceKind::TextureCubeArray)
        samples.emplace_back(call,kind);
    }
    if (op != OpCode::BufferLoad && op != OpCode::BufferStore && op != OpCode::GetDimensions &&
        op != OpCode::AtomicBinOp && op != OpCode::AtomicCompareExchange) continue;
    std::set<Value *> seen;
    if (Kind(call->getArgOperand(1), dm, seen) == ResourceKind::TypedBuffer) calls.push_back(call);
  }
  // Do not even generate fallback IR for proven ordinary bindings. A branch
  // around software sampling still bloats Metal compilation and register use.
  // Keep typed-buffer addressing adaptation independent of texture emulation.
  auto instructionClamp = [](const std::pair<CallInst *, ResourceKind> &sample) {
    auto *call = sample.first;
    const auto op = Opcode(call);
    if (op != OpCode::Sample && op != OpCode::SampleBias &&
        op != OpCode::SampleGrad && op != OpCode::SampleCmp) return false;
    const unsigned index = op == OpCode::SampleGrad ? 16 :
        (op == OpCode::SampleBias || op == OpCode::SampleCmp) ? 11 : 10;
    auto *clamp = call->getArgOperand(index);
    auto *constant = dyn_cast<ConstantFP>(clamp);
    return !isa<UndefValue>(clamp) && (!constant || !constant->isZero());
  };
  auto feedbackUsed = [&](CallInst *call) {
    if (feedbackChecks.empty()) return false;
    for (auto *user : call->users()) {
      auto *extract = dyn_cast<ExtractValueInst>(user);
      // Aggregate users may propagate the status through PHIs or calls. Keep
      // those conservative, but scalar color components need no feedback IR.
      if (!extract) return true;
      if (extract->getNumIndices() == 1 && *extract->idx_begin() == 4 &&
          !extract->use_empty()) return true;
    }
    return false;
  };
  auto requirements = [&](Value *handle) {
    std::set<Value *> seen;
    return SamplingRequirements(dm, handle, policy, policyCount, seen);
  };
  bufferLoads.erase(std::remove_if(bufferLoads.begin(), bufferLoads.end(),
      [&](CallInst *call) { return !feedbackUsed(call); }), bufferLoads.end());
  if (options & (SCORPI_SHADER_NATIVE_TYPED_BUFFERS | SCORPI_SHADER_BINDING_POLICY))
    calls.erase(std::remove_if(calls.begin(), calls.end(), [&](CallInst *call) {
      return !(options & SCORPI_SHADER_BINDING_POLICY) ||
          !(requirements(call->getArgOperand(1)) & (SCORPI_SAMPLING_TYPED_PREFIX | SCORPI_SAMPLING_TYPED_RGB32));
    }), calls.end());
  if (options & SCORPI_SHADER_TYPED_DIMENSIONS_ONLY)
    calls.erase(std::remove_if(calls.begin(), calls.end(), [&](CallInst *call) {
      return Opcode(call) != OpCode::GetDimensions &&
          // Native prefix/RGB32 addressing can wrap a large unsigned index
          // back into the texture. Bound it in logical element space first.
          !(requirements(call->getArgOperand(1)) & (SCORPI_SAMPLING_TYPED_PREFIX |
              SCORPI_SAMPLING_TYPED_RGB32 | SCORPI_SAMPLING_TYPED_EXTENDED_PREFIX));
    }), calls.end());
  if (options & (SCORPI_SHADER_NATIVE_TEXTURES | SCORPI_SHADER_BINDING_POLICY)) {
    // A clamp belongs to one instruction, not the entire shader. Ordinary
    // accesses remain untouched even when another sample needs emulation.
    samples.erase(std::remove_if(samples.begin(), samples.end(),
        [&](const std::pair<CallInst *, ResourceKind> &sample) {
          auto *call = sample.first;
          const bool cubeComparison = (sample.second == ResourceKind::TextureCube || sample.second == ResourceKind::TextureCubeArray) &&
              (Opcode(call) == OpCode::SampleCmp || Opcode(call) == OpCode::SampleCmpLevelZero);
          if ((options & SCORPI_SHADER_BINDING_POLICY) &&
              ((requirements(call->getArgOperand(1)) & (SCORPI_SAMPLING_RESOURCE_MIN_LOD | (feedbackUsed(call) || cubeComparison ? SCORPI_SAMPLING_SPARSE : 0))) ||
               (requirements(call->getArgOperand(2)) & SCORPI_SAMPLING_REDUCTION))) return false;
          return !instructionClamp(sample) && !feedbackUsed(call);
        }), samples.end());
    textureLoads.erase(std::remove_if(textureLoads.begin(), textureLoads.end(),
        [&](CallInst *call) {
          if ((options & SCORPI_SHADER_BINDING_POLICY) &&
              (requirements(call->getArgOperand(1)) & SCORPI_SAMPLING_RESOURCE_MIN_LOD)) return false;
          return !feedbackUsed(call);
        }), textureLoads.end());
  }
  if (rawLoads.empty() && calls.empty() && samples.empty() && textureLoads.empty() && bufferLoads.empty() &&
      nullBufferAccesses.empty()) return inlinedSubroutines || widenedMinimumPrecisionIO || roundedMinimumPrecision || flatFloatInputs || flushedMath || fixedTrig || loweredIntegerArithmetic;
  // Only statuses whose producers are replaced use our canonical 0/1 ABI.
  // Leave unrelated instructions' opaque statuses to the native converter.
  for(auto *check:feedbackChecks){
    std::set<Value*> seen;
    std::function<bool(Value*)> supported=[&](Value *v)->bool{
      if(!seen.insert(v).second)return true;
      if(auto *extract=dyn_cast<ExtractValueInst>(v)){
        if(extract->getNumIndices()!=1 || *extract->idx_begin()!=4)return false;
        auto *source=dyn_cast<CallInst>(extract->getAggregateOperand());
        if(std::find(textureLoads.begin(),textureLoads.end(),source)!=textureLoads.end())return true;
        if(std::find(bufferLoads.begin(),bufferLoads.end(),source)!=bufferLoads.end())return true;
        for(auto &sample:samples)if(sample.first==source)return true;
      }
      if(auto *phi=dyn_cast<PHINode>(v)){for(auto &value:phi->incoming_values())if(!supported(value))return false;return true;}
      if(auto *select=dyn_cast<SelectInst>(v))return supported(select->getTrueValue())&&supported(select->getFalseValue());
      if(auto *binary=dyn_cast<BinaryOperator>(v))if(binary->getOpcode()==Instruction::Or)return supported(binary->getOperand(0))&&supported(binary->getOperand(1));
      if(auto *constant=dyn_cast<ConstantInt>(v))return constant->isZero();
      return false;
    };
    if(supported(check->getArgOperand(1))){
      // DXC ORs the opaque statuses of the loads forming a structured result.
      // For our mapped=true ABI this is an AND, including across PHIs/selects.
      std::map<Value*,Value*> converted;
      std::function<Value*(Value*)> mapped=[&](Value *v)->Value*{
        if(converted.count(v))return converted[v];
        if(isa<ConstantInt>(v))return ConstantInt::getTrue(v->getContext());
        if(auto *phi=dyn_cast<PHINode>(v)){
          auto *p=PHINode::Create(Type::getInt1Ty(v->getContext()),phi->getNumIncomingValues(),"scorpi.mapped",phi);converted[v]=p;
          for(unsigned i=0;i<phi->getNumIncomingValues();++i)p->addIncoming(mapped(phi->getIncomingValue(i)),phi->getIncomingBlock(i));return p;
        }
        IRBuilder<> b(cast<Instruction>(v)->getNextNode());Value *result;
        if(auto *select=dyn_cast<SelectInst>(v))result=b.CreateSelect(select->getCondition(),mapped(select->getTrueValue()),mapped(select->getFalseValue()));
        else if(auto *binary=dyn_cast<BinaryOperator>(v))result=b.CreateAnd(mapped(binary->getOperand(0)),mapped(binary->getOperand(1)));
        else result=b.CreateICmpNE(v,b.getInt32(0));
        return converted[v]=result;
      };
      check->replaceAllUsesWith(mapped(check->getArgOperand(1)));check->eraseFromParent();
    }
  }
  // Fully backed buffers (including NULL views and OOB accesses) have no
  // missing tiles. Fold only their status; preserve native data/OOB handling.
  // Keep unknown handles and dynamically indexed sparse ranges conservative.
  if (options & SCORPI_SHADER_BINDING_POLICY)
    bufferLoads.erase(std::remove_if(bufferLoads.begin(), bufferLoads.end(), [&](CallInst *call) {
      if (requirements(call->getArgOperand(1)) & SCORPI_SAMPLING_SPARSE) return false;
      Lowering::ReplaceStatus(call, ConstantInt::getTrue(call->getContext()));
      return true;
    }), bufferLoads.end());
  if (!calls.empty() || !samples.empty() || !textureLoads.empty() || !bufferLoads.empty()) {
    if (!bindings || !count || space == UINT32_MAX)
      throw std::runtime_error("typed-buffer metadata root binding missing");
    Lowering lowering(dm, bindings, count, space, selfOffset, options, policy, policyCount);
    for (auto *call : bufferLoads) lowering.RewriteBufferStatus(call);
    for (auto *call : calls) lowering.Rewrite(call,
        (options & SCORPI_SHADER_TYPED_DIMENSIONS_ONLY) &&
        (!(options & SCORPI_SHADER_BINDING_POLICY) ||
         (requirements(call->getArgOperand(1)) & SCORPI_SAMPLING_TYPED_RGB32)),
        (options & SCORPI_SHADER_BINDING_POLICY) &&
            (requirements(call->getArgOperand(1)) & SCORPI_SAMPLING_TYPED_EXTENDED_PREFIX),
        options & SCORPI_SHADER_TYPED_DIMENSIONS_ONLY);
    for (auto *call : textureLoads) lowering.RewriteTextureLoad(call);
    for (auto &sample : samples) lowering.RewriteSample(sample.first,sample.second);
  }
  for (auto *call : rawLoads) BoundRawLoadComponents(dm, call);
  // The native converter elides constant-offset raw/structured buffer bounds
  // checks even with BoundsCheck enabled. A null UAV can consequently write
  // the zero-address page, corrupting later null reads. Guard only handles
  // whose current binding policy includes null buffer views. GetDimensions
  // reads native descriptor size; no private metadata or extra resource is
  // needed. Non-null variants and unrelated handles remain untouched.
  for (auto *call : nullBufferAccesses) {
    IRBuilder<> b(call);
    auto *dimensionOp = dm.GetOP()->GetOpFunc(OpCode::GetDimensions, b.getInt32Ty());
    auto *dimensions = b.CreateCall(dimensionOp, {b.getInt32(unsigned(OpCode::GetDimensions)),
        call->getArgOperand(1), b.getInt32(0)});
    auto *present = b.CreateICmpNE(b.CreateExtractValue(dimensions, 0), b.getInt32(0));
    auto *before = call->getParent();
    auto *after = before->splitBasicBlock(call->getNextNode(), "scorpi.null.buffer.end");
    before->getTerminator()->eraseFromParent();
    auto *access = BasicBlock::Create(call->getContext(), "scorpi.buffer.access",
        before->getParent(), after);
    call->removeFromParent();
    access->getInstList().push_back(call);
    BranchInst::Create(after, access);
    BranchInst::Create(access, after, present, before);
    if (!call->getType()->isVoidTy()) {
      auto *result = PHINode::Create(call->getType(), 2, "scorpi.null.buffer.result", &*after->begin());
      call->replaceAllUsesWith(result);
      result->addIncoming(call, access);
      // A null descriptor has no missing sparse tiles: native opaque status
      // zero also denotes fully mapped, matching untouched native loads.
      result->addIncoming(Constant::getNullValue(call->getType()), before);
    }
  }
  if (!samples.empty() || !feedbackChecks.empty() || !nullBufferAccesses.empty()) {
    // Canonicalize aggregate sample-result PHIs and dead descriptor arithmetic
    // before IR conversion. Conditional multi-sample shaders otherwise produce
    // AIR rejected by the Metal compiler's bitcode reader.
    legacy::PassManager cleanup;
    cleanup.add(createInstructionCombiningPass());
    cleanup.add(createCFGSimplificationPass());
    cleanup.run(*dm.GetModule());
    dm.CollectShaderFlagsForModule();
  }
  return true;
}
