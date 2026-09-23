#include "ScorpiRasterReplay.h"
#include "dxc/DXIL/DxilCBuffer.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilResource.h"
#include "dxc/DXIL/DxilResourceBinding.h"
#include "dxc/DXIL/DxilResourceProperties.h"
#include "dxc/DXIL/DxilShaderModel.h"
#include "dxc/DXIL/DxilSignature.h"
#include "dxc/DXIL/DxilSignatureElement.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include <algorithm>
#include <functional>
#include <vector>

namespace scorpi {
namespace {
using namespace llvm;
using hlsl::DXIL::OpCode;

bool IsWrite(OpCode op) {
  switch (op) {
  case OpCode::BufferStore:
  case OpCode::RawBufferStore:
  case OpCode::TextureStore:
  case OpCode::TextureStoreSample:
  case OpCode::AtomicBinOp:
  case OpCode::AtomicCompareExchange:
  case OpCode::BufferUpdateCounter:
    return true;
  default:
    return false;
  }
}

// Split only the side effect, not the surrounding shader. Inactive replay
// lanes still execute interpolation, texture sampling and derivatives.
void PredicateWrite(CallInst *call, Value *active) {
  auto *before = call->getParent();
  auto *after = before->splitBasicBlock(call, "replay.continue");
  auto *effect = BasicBlock::Create(call->getContext(), "replay.write",
                                    before->getParent(), after);
  before->getTerminator()->eraseFromParent();
  BranchInst::Create(effect, after, active, before);
  call->removeFromParent();
  effect->getInstList().push_back(call);
  BranchInst::Create(after, effect);
  if (!call->getType()->isVoidTy()) {
    auto *result =
        PHINode::Create(call->getType(), 2, "replay.result", &after->front());
    call->replaceAllUsesWith(result);
    result->addIncoming(call, effect);
    // D3D permits an undefined value for inactive-lane atomics; use zero.
    result->addIncoming(Constant::getNullValue(call->getType()), before);
  }
}
} // namespace

bool PrepareMsaa8TextureLoads(hlsl::DxilModule &dm, std::string &error,
                              bool &changed,
                              const ScorpiSamplingRequirement *policy,
                              size_t policy_count) {
  using namespace llvm;
  using namespace hlsl;
  using Kind = DXIL::ResourceKind;
  auto ms = [](Kind k) {
    return k == Kind::Texture2DMS || k == Kind::Texture2DMSArray;
  };
  changed = false;
  // Unspecified sample counts are specialized from the immutable binding
  // snapshot, not from unrelated eight-sample resources elsewhere in a heap.
  for (auto &r : dm.GetSRVs())
    if (ms(r->GetKind()) && r->GetSampleCount() != 8) {
      uint64_t selected = 0;
      for (size_t i = 0; i < policy_count; ++i) {
        const auto &p = policy[i];
        if (p.resource_class || p.space != r->GetSpaceID() ||
            !(p.requirements & SCORPI_SAMPLING_MSAA8))
          continue;
        const uint64_t lo =
            std::max<uint64_t>(p.base_register, r->GetLowerBound());
        const uint64_t hi = std::min<uint64_t>(
            uint64_t(p.base_register) + p.count,
            uint64_t(r->GetLowerBound()) + r->GetRangeSize());
        if (hi > lo)
          selected += hi - lo;
      }
      if (!selected)
        continue;
      if (r->GetSampleCount() != 0 || selected != r->GetRangeSize()) {
        error = "8x texture binding has a conflicting sample count or mixed "
                "descriptor-array layout";
        return false;
      }
      r->SetSampleCount(8);
    }
  for (const auto &r : dm.GetSRVs())
    changed |= ms(r->GetKind()) && r->GetSampleCount() == 8;
  if (!changed)
    return true;
  struct Info {
    Kind kind = Kind::Invalid;
    unsigned samples = 0;
  };
  std::function<Info(Value *, unsigned)> info = [&](Value *v,
                                                    unsigned depth) -> Info {
    if (depth > 16)
      return {};
    if (auto *c = dyn_cast<CallInst>(v)) {
      if (!c->getCalledFunction() ||
          !c->getCalledFunction()->getName().startswith("dx.op."))
        return {};
      auto *constant = dyn_cast<ConstantInt>(c->getArgOperand(0));
      if (!constant)
        return {};
      auto op = OpCode(constant->getZExtValue());
      if (op == OpCode::AnnotateHandle) {
        auto p = resource_helper::loadPropsFromConstant(
            *cast<Constant>(c->getArgOperand(2)));
        if (ms(p.getResourceKind()) && !p.Typed.SampleCount) {
          auto binding = info(c->getArgOperand(1), depth + 1);
          if (binding.samples == 8)
            return binding;
        }
        return {p.getResourceKind(), p.Typed.SampleCount};
      }
      if (op == OpCode::CreateHandleFromBinding) {
        auto binding = resource_helper::loadBindingFromConstant(
            *cast<Constant>(c->getArgOperand(1)));
        if (binding.resourceClass == unsigned(DXIL::ResourceClass::SRV))
          for (const auto &r : dm.GetSRVs())
            if (r->GetSpaceID() == binding.spaceID &&
                r->GetLowerBound() == binding.rangeLowerBound)
              return {r->GetKind(), r->GetSampleCount()};
      }
      if (op == OpCode::CreateHandle &&
          cast<ConstantInt>(c->getArgOperand(1))->getZExtValue() ==
              unsigned(DXIL::ResourceClass::SRV)) {
        auto id = cast<ConstantInt>(c->getArgOperand(2))->getZExtValue();
        if (id < dm.GetSRVs().size())
          return {dm.GetSRV(id).GetKind(), dm.GetSRV(id).GetSampleCount()};
      }
    }
    if (auto *s = dyn_cast<SelectInst>(v)) {
      auto a = info(s->getTrueValue(), depth + 1),
           b = info(s->getFalseValue(), depth + 1);
      if (a.kind == b.kind && a.samples == b.samples)
        return a;
      if (a.samples == 8 || b.samples == 8 || a.samples == UINT32_MAX ||
          b.samples == UINT32_MAX)
        return {Kind::Invalid, UINT32_MAX};
    }
    if (auto *p = dyn_cast<PHINode>(v)) {
      Info result;
      bool first = true;
      for (auto &incoming : p->incoming_values()) {
        auto next = info(incoming, depth + 1);
        if (!first &&
            (result.kind != next.kind || result.samples != next.samples))
          return {Kind::Invalid, UINT32_MAX};
        result = next;
        first = false;
      }
      return result;
    }
    return {};
  };
  std::vector<CallInst *> calls, annotations;
  for (auto &f : *dm.GetModule())
    for (auto &block : f)
      for (auto &inst : block)
        if (auto *c = dyn_cast<CallInst>(&inst))
          if (c->getCalledFunction() &&
              c->getCalledFunction()->getName().startswith("dx.op."))
            calls.push_back(c);
  for (auto *call : calls) {
    auto *opcode = dyn_cast<ConstantInt>(call->getArgOperand(0));
    if (!opcode)
      continue;
    auto op = OpCode(opcode->getZExtValue());
    if (op == OpCode::AnnotateHandle) {
      annotations.push_back(call);
      continue;
    }
    if (call->getNumArgOperands() < 2)
      continue;
    auto resource = info(call->getArgOperand(1), 0);
    if (resource.samples == UINT32_MAX) {
      error = "mixed native/split multisample handle selection";
      return false;
    }
    if (!ms(resource.kind) || resource.samples != 8)
      continue;
    IRBuilder<> b(call);
    if (op == OpCode::TextureLoad) {
      auto sample = call->getArgOperand(2);
      Value *layer = resource.kind == Kind::Texture2DMSArray
                         ? call->getArgOperand(5)
                         : b.getInt32(0);
      call->setArgOperand(5, b.CreateAdd(b.CreateMul(layer, b.getInt32(2)),
                                         b.CreateLShr(sample, b.getInt32(2))));
      call->setArgOperand(2, b.CreateAnd(sample, b.getInt32(3)));
      auto *valid = b.CreateAnd(b.CreateICmpULT(sample, b.getInt32(8)),
                                b.CreateICmpULT(layer, b.getInt32(1024)));
      call->setArgOperand(
          3, b.CreateSelect(valid, call->getArgOperand(3), b.getInt32(-1)));
    } else if (op == OpCode::GetDimensions) {
      auto *native = cast<CallInst>(call->clone());
      native->insertBefore(call);
      auto *dims = b.CreateInsertValue(UndefValue::get(call->getType()),
                                       b.getInt32(8), 3);
      Value *result = dims;
      for (unsigned i = 0; i < 3; ++i) {
        Value *v = b.CreateExtractValue(native, i);
        if (i == 2 && resource.kind == Kind::Texture2DMSArray)
          v = b.CreateLShr(v, b.getInt32(1));
        result = b.CreateInsertValue(result, v, i);
      }
      call->replaceAllUsesWith(result);
      call->eraseFromParent();
    } else if (op == OpCode::Texture2DMSGetSamplePosition) {
      static const int8_t xy[8][2] = {{1, -3}, {-1, 3},  {5, 1}, {-3, -5},
                                      {-5, 5}, {-7, -1}, {3, 7}, {7, -7}};
      Value *result = UndefValue::get(call->getType());
      for (unsigned axis = 0; axis < 2; ++axis) {
        Value *v = ConstantFP::get(b.getFloatTy(), 0);
        for (unsigned s = 0; s < 8; ++s)
          v = b.CreateSelect(
              b.CreateICmpEQ(call->getArgOperand(2), b.getInt32(s)),
              ConstantFP::get(b.getFloatTy(), xy[s][axis] / 16.f), v);
        result = b.CreateInsertValue(result, v, axis);
      }
      call->replaceAllUsesWith(result);
      call->eraseFromParent();
    } else {
      error = "unsupported operation on split 8x texture";
      return false;
    }
  }
  for (auto *call : annotations) {
    auto p = resource_helper::loadPropsFromConstant(
        *cast<Constant>(call->getArgOperand(2)));
    if (ms(p.getResourceKind()) && info(call, 0).samples == 8) {
      p.setResourceKind(Kind::Texture2DMSArray);
      p.Typed.SampleCount = 4;
      call->setArgOperand(2, resource_helper::getAsConstant(
                                 p, dm.GetOP()->GetResourcePropertiesType(),
                                 *dm.GetShaderModel()));
    }
  }
  for (auto &r : dm.GetSRVs())
    if (ms(r->GetKind()) && r->GetSampleCount() == 8) {
      r->SetKind(Kind::Texture2DMSArray);
      r->SetSampleCount(4);
    }
  return true;
}

bool PrepareRasterReplayPixelShader(hlsl::DxilModule &dm,
                                    uint32_t registerSpace,
                                    uint32_t sampleCount, uint32_t sampleMask,
                                    std::string &error, bool generated, bool attachments, bool alphaToCoverage) {
  using namespace llvm;
  using namespace hlsl;
  auto fail = [&](const char *why) {
    error = why;
    return false;
  };
  if (!dm.GetShaderModel()->IsPS() || !dm.GetEntryFunction())
    return fail("raster replay requires a pixel shader");
  if (sampleCount != 1 && sampleCount != 4 && sampleCount != 8 && sampleCount != 16)
    return fail("raster replay requires 1, 4, 8 or 16 logical samples");
  sampleMask &= (1u << sampleCount) - 1u;
  unsigned alphaID = UINT32_MAX, alphaRow = 0;
  for (const auto &e : dm.GetOutputSignature().GetElements()) {
    if (e->GetKind() == DXIL::SemanticKind::Coverage) alphaToCoverage = false;
    if (e->GetKind() == DXIL::SemanticKind::Target && e->GetCols() == 4)
      for (unsigned row = 0; row < e->GetSemanticIndexVec().size(); ++row)
        if (e->GetSemanticIndexVec()[row] == 0) { alphaID = e->GetID(); alphaRow = row; }
  }
  alphaToCoverage &= attachments && alphaID != UINT32_MAX;
  if (alphaToCoverage && dm.m_ShaderFlags.GetForceEarlyDepthStencil())
    return fail("8x early-depth/stencil with alpha-to-coverage requires pre-test masking");
  if (attachments && dm.m_ShaderFlags.GetForceEarlyDepthStencil() && sampleMask != 255)
    return fail("8x early-depth/stencil with partial OM sample mask requires pre-test masking");
  for (const auto &r : dm.GetCBuffers())
    if (r->GetSpaceID() == registerSpace && r->GetLowerBound() == 0)
      return fail("raster replay cbuffer binding collides with application");
  for (const auto &r : dm.GetSRVs())
    if (r->GetSpaceID() == registerSpace && r->GetLowerBound() == 0)
      return fail("raster replay coverage binding collides with application");
  for (const auto &r : dm.GetUAVs())
    if (r->GetSpaceID() == registerSpace && r->GetLowerBound() <= 1 &&
        uint64_t(r->GetLowerBound()) + r->GetRangeSize() > 1)
      return fail("raster replay visibility binding collides with application");

  if (generated) for (const auto &r : dm.GetUAVs())
    if (r->GetSpaceID() == registerSpace && r->GetLowerBound() == 0)
      return fail("generated replay storage binding collides with application");

  auto *entry = dm.GetEntryFunction();
  std::vector<CallInst *> originalCalls;
  bool sampleFrequency = false;
  for (const auto &e : dm.GetInputSignature().GetElements())
    sampleFrequency |= e->GetInterpolationMode()->IsAnySample();
  for (auto &block : *entry)
    for (auto &inst : block) {
      auto *call = dyn_cast<CallInst>(&inst);
      if (!call)
        continue;
      auto *function = call->getCalledFunction();
      if (!function || (!function->isDeclaration() && function != entry))
        return fail("raster replay requires inlined pixel-shader functions");
      if (!function->getName().startswith("dx.op.")) {
        if (call->mayWriteToMemory())
          return fail(
              "raster replay encountered an unknown memory-writing call");
        continue;
      }
      auto *opcode = dyn_cast<ConstantInt>(call->getArgOperand(0));
      if (!opcode)
        return fail("raster replay encountered a nonconstant DXIL opcode");
      auto op = static_cast<OpCode>(opcode->getZExtValue());
      sampleFrequency |= op == OpCode::SampleIndex;
      originalCalls.push_back(call);
    }

  auto &ctx = entry->getContext();
  // Pixel-frequency UAV values cannot be independently recomputed for each
  // physical sample group. Until result broadcast is available, reject those
  // attachment shaders rather than duplicate effects or shade with dummy values.
  if (attachments && !sampleFrequency)
    for (auto *call : originalCalls)
      if (IsWrite(static_cast<OpCode>(cast<ConstantInt>(call->getArgOperand(0))->getZExtValue())))
        return fail("8x attachment replay requires pixel-frequency UAV result broadcast");
  auto *module = dm.GetModule();
  auto *ops = dm.GetOP();
  auto *i32 = Type::getInt32Ty(ctx);
  auto *f32 = Type::getFloatTy(ctx);
  IRBuilder<> b(&*entry->getEntryBlock().getFirstInsertionPt());
  auto *alpha = alphaToCoverage ? b.CreateAlloca(f32, nullptr, "scorpi.alpha") : nullptr;
  if (alpha) b.CreateStore(ConstantFP::get(f32, 1.0), alpha);
  auto opCall = [&](OpCode op, Type *type, std::vector<Value *> args) {
    args.insert(args.begin(), b.getInt32(static_cast<unsigned>(op)));
    return b.CreateCall(ops->GetOpFunc(op, type), args);
  };
  auto cb = std::make_unique<DxilCBuffer>();
  cb->SetID(static_cast<unsigned>(dm.GetCBuffers().size()));
  cb->SetSpaceID(registerSpace);
  cb->SetLowerBound(0);
  cb->SetRangeSize(1);
  cb->SetSize(48);
  cb->SetGlobalName("ScorpiRasterReplay");
  auto *cbType = StructType::create(
      ctx, {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32}, "ScorpiRasterReplay");
  cb->SetGlobalSymbol(UndefValue::get(cbType->getPointerTo()));
  unsigned cbID = dm.AddCBuffer(std::move(cb));
  auto srv = std::make_unique<DxilResource>();
  srv->SetID(static_cast<unsigned>(dm.GetSRVs().size()));
  srv->SetRW(false);
  srv->SetKind(DXIL::ResourceKind::RawBuffer);
  srv->SetSpaceID(registerSpace);
  srv->SetLowerBound(0);
  srv->SetRangeSize(1);
  srv->SetGlobalName("ScorpiRasterCoverage");
  auto *srvType = StructType::create(ctx, {i32}, "ScorpiRasterCoverage");
  srv->SetGlobalSymbol(UndefValue::get(srvType->getPointerTo()));
  unsigned srvID = dm.AddSRV(std::move(srv));
  auto visibility = std::make_unique<DxilResource>();
  visibility->SetID(unsigned(dm.GetUAVs().size())); visibility->SetRW(true);
  visibility->SetKind(DXIL::ResourceKind::RawBuffer); visibility->SetSpaceID(registerSpace);
  visibility->SetLowerBound(1); visibility->SetRangeSize(1);
  visibility->SetGlobalName("ScorpiRasterVisibility");
  visibility->SetGlobalSymbol(UndefValue::get(
      StructType::create(ctx, {i32}, "ScorpiRasterVisibility")->getPointerTo()));
  unsigned visibilityID = dm.AddUAV(std::move(visibility));
  unsigned captureID = 0;
  if (generated) {
    auto storage = std::make_unique<DxilResource>();
    storage->SetID(unsigned(dm.GetUAVs().size())); storage->SetRW(true);
    storage->SetKind(DXIL::ResourceKind::RawBuffer); storage->SetSpaceID(registerSpace);
    storage->SetLowerBound(0); storage->SetRangeSize(1); storage->SetGlobalName("ScorpiCapturedGeometry");
    storage->SetGlobalSymbol(UndefValue::get(StructType::create(ctx, {i32}, "ScorpiCapturedGeometry")->getPointerTo()));
    captureID = dm.AddUAV(std::move(storage));
  }
  auto handle = [&](DXIL::ResourceClass cls, unsigned id) {
    DxilResourceBase *resource =
          cls == DXIL::ResourceClass::CBuffer
              ? static_cast<DxilResourceBase *>(&dm.GetCBuffer(id))
              : cls == DXIL::ResourceClass::UAV ? &dm.GetUAV(id) : &dm.GetSRV(id);
    if (dm.GetShaderModel()->GetMinor() >= 6) {
      auto binding = resource_helper::loadBindingFromResourceBase(resource);
      auto *bindingValue = resource_helper::getAsConstant(
          binding, ops->GetResourceBindingType(), *dm.GetShaderModel());
      auto *unannotated = opCall(OpCode::CreateHandleFromBinding, b.getVoidTy(),
                                 {bindingValue, b.getInt32(resource->GetLowerBound()), b.getFalse()});
      auto properties = resource_helper::loadPropsFromResourceBase(resource);
      auto *propertiesValue = resource_helper::getAsConstant(
          properties, ops->GetResourcePropertiesType(), *dm.GetShaderModel());
      return opCall(OpCode::AnnotateHandle, b.getVoidTy(),
                    {unannotated, propertiesValue});
    }
    return opCall(OpCode::CreateHandle, b.getVoidTy(),
                  {b.getInt8(static_cast<unsigned>(cls)), b.getInt32(id),
                   b.getInt32(resource->GetLowerBound()), b.getFalse()});
  };
  auto *cbHandle = handle(DXIL::ResourceClass::CBuffer, cbID);
  auto *coverageHandle = handle(DXIL::ResourceClass::SRV, srvID);
  auto *visibilityHandle = handle(DXIL::ResourceClass::UAV, visibilityID);
  auto *queryParams = opCall(OpCode::CBufferLoadLegacy, i32, {cbHandle, b.getInt32(2)});
  auto *queryEnabled = b.CreateICmpNE(b.CreateExtractValue(queryParams, 2), b.getInt32(0));
  auto *params =
      opCall(OpCode::CBufferLoadLegacy, i32, {cbHandle, b.getInt32(0)});
  auto *width = b.CreateExtractValue(params, 0);
  auto *height = b.CreateExtractValue(params, 1);
  auto *baseElement = b.CreateExtractValue(params, 2);
  auto *sampleBase = b.CreateExtractValue(params, 3);
  auto *origin =
      opCall(OpCode::CBufferLoadLegacy, i32, {cbHandle, b.getInt32(1)});
  auto *primitiveID = b.CreateExtractValue(origin, 2);
  unsigned positionID = UINT32_MAX, rasterPrimitiveID = UINT32_MAX, nextRow = 0;
  for (const auto &e : dm.GetInputSignature().GetElements()) {
    if (e->GetKind() == DXIL::SemanticKind::Position)
      positionID = e->GetID();
    if (e->GetKind() == DXIL::SemanticKind::PrimitiveID)
      rasterPrimitiveID = e->GetID();
    if (e->IsAllocated())
      nextRow = std::max(nextRow, e->GetStartRow() + e->GetRows());
  }
  if (positionID == UINT32_MAX) {
    auto position =
        std::make_unique<DxilSignatureElement>(DXIL::SigPointKind::PSIn);
    position->Initialize("SV_Position", CompType::getF32(),
                         DXIL::InterpolationMode::Linear, 1, 4, nextRow, 0);
    position->AppendSemanticIndex(0);
    position->SetKind(DXIL::SemanticKind::Position);
    positionID = dm.GetInputSignature().AppendElement(std::move(position));
    ++nextRow;
  }
  if (generated) {
    if (rasterPrimitiveID == UINT32_MAX) {
      auto element = std::make_unique<DxilSignatureElement>(DXIL::SigPointKind::PSIn);
      element->Initialize("SV_PrimitiveID", CompType::getU32(), DXIL::InterpolationMode::Constant,
                          1, 1, nextRow, 0);
      element->AppendSemanticIndex(0); element->SetKind(DXIL::SemanticKind::PrimitiveID);
      rasterPrimitiveID = dm.GetInputSignature().AppendElement(std::move(element));
    }
    auto *rasterID = opCall(OpCode::LoadInput, i32,
        {b.getInt32(rasterPrimitiveID), b.getInt32(0), b.getInt8(0), UndefValue::get(i32)});
    baseElement = b.CreateAdd(baseElement, b.CreateMul(rasterID, b.CreateMul(width, height)));
    auto *record = b.CreateAdd(b.CreateExtractValue(origin, 3), b.CreateMul(rasterID, b.getInt32(16)));
    auto *loadedID = opCall(OpCode::BufferLoad, i32,
        {handle(DXIL::ResourceClass::UAV, captureID), b.CreateAdd(record, b.getInt32(12)), UndefValue::get(i32)});
    primitiveID = b.CreateExtractValue(loadedID, 0);
  }
  auto coordinate = [&](unsigned component) {
    return b.CreateFPToSI(opCall(OpCode::LoadInput, f32,
                                 {b.getInt32(positionID), b.getInt32(0),
                                  b.getInt8(component), UndefValue::get(i32)}),
                          i32);
  };
  auto *x = b.CreateSub(coordinate(0), b.CreateExtractValue(origin, 0));
  auto *y = b.CreateSub(coordinate(1), b.CreateExtractValue(origin, 1));
  auto *inBounds =
      b.CreateAnd(b.CreateICmpULT(x, width), b.CreateICmpULT(y, height));
  auto *element =
      b.CreateAdd(baseElement, b.CreateAdd(b.CreateMul(y, width), x));
  // Helper lanes outside the viewport read a safe element and remain inactive.
  auto *address = b.CreateMul(b.CreateSelect(inBounds, element, baseElement),
                              b.getInt32(4));
  auto *loaded = opCall(OpCode::BufferLoad, i32,
                        {coverageHandle, address, UndefValue::get(i32)});
  auto *coverage =
      b.CreateSelect(inBounds, b.CreateExtractValue(loaded, 0), b.getInt32(0));
  auto *maskedCoverage = b.CreateAnd(coverage, b.getInt32(sampleMask));
  // Attachment queries use native visibility slots. Pure pixel-frequency
  // shaders must stay pixel-frequency (in particular SV_Position stays at
  // the pixel center); only sample-frequency shaders require SampleIndex.
  Value *nativeSample = b.getInt32(0);
  if (!attachments || sampleFrequency) nativeSample = opCall(OpCode::SampleIndex, i32, {});
  auto *sample = b.CreateAdd(nativeSample, sampleBase);
  auto *bit = b.CreateShl(b.getInt32(1), sample);
  auto *active =
      b.CreateICmpNE(b.CreateAnd(maskedCoverage, bit), b.getInt32(0));
  if (!sampleFrequency) {
    auto *firstBit = b.CreateAnd(maskedCoverage, b.CreateNeg(maskedCoverage));
    active = b.CreateAnd(active, b.CreateICmpEQ(bit, firstBit));
  }
  auto *alive = b.CreateAlloca(b.getInt1Ty(), nullptr, "replay.alive");
  b.CreateStore(active, alive);
  auto *outputCoverage = b.CreateAlloca(i32, nullptr, "replay.outputCoverage");
  auto *queryStart = b.CreateStore(b.getInt32(UINT32_MAX), outputCoverage);
  const bool earlyDepth = dm.m_ShaderFlags.GetForceEarlyDepthStencil();
  auto recordVisibility = [&](IRBuilder<> &at, Value *survives, Value *outputMask) {
    Value *bits = sampleFrequency ? bit : maskedCoverage;
    bits = at.CreateAnd(bits, outputMask);
    auto *enabled = at.CreateAnd(queryEnabled, survives);
    auto *write = at.CreateCall(ops->GetOpFunc(OpCode::AtomicBinOp, i32),
        {at.getInt32(unsigned(OpCode::AtomicBinOp)), visibilityHandle,
         at.getInt32(unsigned(DXIL::AtomicBinOpCode::Or)), address,
         UndefValue::get(i32), UndefValue::get(i32), bits});
    PredicateWrite(write, enabled);
  };
  // DXIL EvalSnapped offsets use sixteenths of a pixel about its center.
  static const int8_t positions[3][16][2] = {
      {{-2, -6}, {6, -2}, {-6, 2}, {2, 6}},
      {{1, -3}, {-1, 3}, {5, 1}, {-3, -5}, {-5, 5}, {-7, -1}, {3, 7}, {7, -7}},
      {{1, 1},
       {-1, -3},
       {-3, 2},
       {4, -1},
       {-5, -2},
       {2, 5},
       {5, 3},
       {3, -5},
       {-2, 6},
       {0, -7},
       {-4, -6},
       {-6, 4},
       {-8, 0},
       {7, -4},
       {6, 7},
       {-7, -8}}};
  const auto &pattern = positions[sampleCount == 4   ? 0
                                  : sampleCount == 8 ? 1
                                                     : 2];
  auto offsetAtSample = [&](IRBuilder<> &at, Value *index, unsigned axis) {
    Value *offset = at.getInt32(0);
    for (unsigned i = 0; i < sampleCount; ++i)
      offset = at.CreateSelect(
          at.CreateICmpEQ(index, at.getInt32(i)),
          at.getInt32(static_cast<uint32_t>(sampleCount == 1 ? 0 : pattern[i][axis])), offset);
    return offset;
  };
  Value *centroidSample = b.getInt32(0);
  auto *centroidMask =
      b.CreateSelect(b.CreateICmpNE(maskedCoverage, b.getInt32(0)),
                     maskedCoverage, b.getInt32(sampleMask));
  for (unsigned i = sampleCount; i > 0; --i)
    centroidSample = b.CreateSelect(
        b.CreateICmpNE(b.CreateAnd(centroidMask, b.getInt32(1u << (i - 1))),
                       b.getInt32(0)),
        b.getInt32(i - 1), centroidSample);
  auto *fullCoverage =
      b.CreateICmpEQ(coverage, b.getInt32((1u << sampleCount) - 1));
  auto *helperAtCenter =
      b.CreateAnd(b.CreateICmpEQ(maskedCoverage, b.getInt32(0)),
                  b.getInt1(sampleMask == ((1u << sampleCount) - 1)));
  auto *centroidAtCenter = b.CreateOr(fullCoverage, helperAtCenter);
  Value *centroidOffset[2];
  for (unsigned axis = 0; axis < 2; ++axis)
    centroidOffset[axis] =
        b.CreateSelect(centroidAtCenter, b.getInt32(0),
                       offsetAtSample(b, centroidSample, axis));
  for (auto *call : originalCalls) {
    auto op = static_cast<OpCode>(
        cast<ConstantInt>(call->getArgOperand(0))->getZExtValue());
    bool centroidLoad = false, primitiveLoad = false;
    if (op == OpCode::LoadInput) {
      auto *id = dyn_cast<ConstantInt>(call->getArgOperand(1));
      if (id)
        for (const auto &e : dm.GetInputSignature().GetElements())
          if (e->GetID() == id->getZExtValue()) {
            centroidLoad = e->GetInterpolationMode()->IsAnyCentroid();
            primitiveLoad = e->GetKind() == DXIL::SemanticKind::PrimitiveID;
          }
    }
    if (attachments && op == OpCode::RenderTargetGetSampleCount) {
      call->replaceAllUsesWith(b.getInt32(sampleCount));
      call->eraseFromParent();
    } else if (attachments && op == OpCode::RenderTargetGetSamplePosition) {
      IRBuilder<> at(call);
      Value *position = UndefValue::get(call->getType());
      for (unsigned axis=0; axis<2; ++axis)
        position = at.CreateInsertValue(position,
            at.CreateFMul(at.CreateSIToFP(offsetAtSample(at,call->getArgOperand(1),axis),f32),
                ConstantFP::get(f32,1.0/16)),axis);
      call->replaceAllUsesWith(position);
      call->eraseFromParent();
    } else if (op == OpCode::EvalSampleIndex || op == OpCode::EvalCentroid ||
        centroidLoad) {
      IRBuilder<> at(call);
      auto *ox = op == OpCode::EvalSampleIndex
                     ? offsetAtSample(at, call->getArgOperand(4), 0)
                     : centroidOffset[0];
      auto *oy = op == OpCode::EvalSampleIndex
                     ? offsetAtSample(at, call->getArgOperand(4), 1)
                     : centroidOffset[1];
      auto *value = at.CreateCall(
          ops->GetOpFunc(OpCode::EvalSnapped, call->getType()),
          {at.getInt32(static_cast<unsigned>(OpCode::EvalSnapped)),
           call->getArgOperand(1), call->getArgOperand(2),
           call->getArgOperand(3), ox, oy});
      call->replaceAllUsesWith(value);
      call->eraseFromParent();
    } else if (op == OpCode::PrimitiveID || primitiveLoad) {
      call->replaceAllUsesWith(primitiveID);
      call->eraseFromParent();
    } else if (op == OpCode::SampleIndex || op == OpCode::Coverage) {
      call->replaceAllUsesWith(op == OpCode::SampleIndex ? sample
                                                         : maskedCoverage);
      call->eraseFromParent();
    } else if (op == OpCode::Discard) {
      IRBuilder<> at(call);
      at.CreateStore(at.CreateAnd(at.CreateLoad(alive),
                                  at.CreateNot(call->getArgOperand(1))),
                     alive);
      // Keep the discard itself as an ordering boundary.
    } else if (op == OpCode::StoreOutput) {
      if (alpha && call->getArgOperand(1) == b.getInt32(alphaID) &&
          call->getArgOperand(2) == b.getInt32(alphaRow) && call->getArgOperand(3) == b.getInt8(3)) {
        IRBuilder<> at(call);
        auto *value = call->getArgOperand(4);
        if (value->getType()->isHalfTy()) value = at.CreateFPExt(value, f32);
        if (value->getType() != f32) return fail("8x alpha-to-coverage requires floating-point target alpha");
        at.CreateStore(value, alpha);
      }
      // SV_Coverage affects ordinary visibility even with no attachments.
      if (auto *id = dyn_cast<ConstantInt>(call->getArgOperand(1)))
        for (const auto &e : dm.GetOutputSignature().GetElements())
          if (e->GetID() == id->getZExtValue() && e->GetKind() == DXIL::SemanticKind::Coverage) {
            IRBuilder<> at(call);
            at.CreateStore(call->getArgOperand(4), outputCoverage);
          }
      // This variant is used only with no color or depth/stencil attachments.
      // Outputs are discarded, but computations leading to them (including
      // UAV side effects) remain subject to the normal replay predicates.
      if (!attachments) call->eraseFromParent();
    } else if (IsWrite(op)) {
      IRBuilder<> at(call);
      PredicateWrite(call, at.CreateLoad(alive));
    }
  }
  if (!attachments && earlyDepth) {
    // Insert before original shader work, so neither discard nor output
    // coverage can reduce a FORCE_EARLY_DEPTH_STENCIL query.
    IRBuilder<> at(queryStart);
    recordVisibility(at, active, at.getInt32(UINT32_MAX));
  } else if (!attachments) {
    std::vector<ReturnInst *> returns;
    for (auto &block : *entry)
      if (auto *ret = dyn_cast<ReturnInst>(block.getTerminator())) returns.push_back(ret);
    for (auto *ret : returns) {
      IRBuilder<> at(ret);
      recordVisibility(at, at.CreateLoad(alive), at.CreateLoad(outputCoverage));
    }
  }
  if (attachments) {
    // Hardware coverage is local to one four-sample group. Keep target/depth
    // outputs, and translate the logical eight-bit mask only at the OM boundary.
    unsigned coverageID = UINT32_MAX;
    for (const auto &e : dm.GetOutputSignature().GetElements())
      if (e->GetKind() == DXIL::SemanticKind::Coverage) coverageID = e->GetID();
    if (coverageID == UINT32_MAX) {
      auto e = std::make_unique<DxilSignatureElement>(DXIL::SigPointKind::PSOut);
      e->Initialize("SV_Coverage", CompType::getU32(), DXIL::InterpolationMode::Undefined, 1, 1);
      e->AppendSemanticIndex(0); e->SetKind(DXIL::SemanticKind::Coverage);
      coverageID = dm.GetOutputSignature().AppendElement(std::move(e));
    }
    std::vector<CallInst *> oldCoverage;
    for (auto &block : *entry) for (auto &inst : block)
      if (auto *call = dyn_cast<CallInst>(&inst))
        if (call->getCalledFunction() && call->getCalledFunction()->getName().startswith("dx.op.storeOutput"))
          if (auto *id = dyn_cast<ConstantInt>(call->getArgOperand(1)))
            if (id->getZExtValue() == coverageID) oldCoverage.push_back(call);
    for (auto *call : oldCoverage) call->eraseFromParent();
    for (auto &block : *entry) if (auto *ret = dyn_cast<ReturnInst>(block.getTerminator())) {
      IRBuilder<> at(ret);
      Value *logicalMask = at.CreateAnd(at.CreateLoad(outputCoverage), maskedCoverage);
      if (alpha) {
        // Nested, spatially distributed masks over all eight logical samples.
        // Ordered comparisons also map NaN to zero coverage. Do not quantize
        // alpha independently in each physical four-sample group.
        const unsigned order[8] = {0, 4, 2, 6, 1, 5, 3, 7};
        Value *alphaMask = at.getInt32(0);
        auto *value = at.CreateLoad(alpha);
        for (unsigned rank = 0; rank < 8; ++rank)
          alphaMask = at.CreateOr(alphaMask, at.CreateSelect(
              at.CreateFCmpOGE(value, ConstantFP::get(f32, (rank + 0.5) / 8.0)),
              at.getInt32(1u << order[rank]), at.getInt32(0)));
        logicalMask = at.CreateAnd(logicalMask, alphaMask);
      }
      auto *mask = at.CreateLShr(logicalMask, sampleBase);
      at.CreateCall(ops->GetOpFunc(OpCode::StoreOutput, i32),
          {at.getInt32(unsigned(OpCode::StoreOutput)),at.getInt32(coverageID),at.getInt32(0),at.getInt8(0),mask});
    }
  } else dm.GetOutputSignature().ClearElements();
  DominatorTree dominators;
  dominators.recalculate(*entry);
  PromoteMemToReg({alive, outputCoverage}, dominators);
  if (alpha) PromoteMemToReg({alpha}, dominators);
  dm.CollectShaderFlagsForModule();
  // DXIL rejects declarations left behind after replacing an intrinsic.
  for (auto it = module->begin(); it != module->end();) {
    auto *function = &*it++;
    if (function->isDeclaration() && function->use_empty()) {
      dm.RemoveFunction(function);
      function->eraseFromParent();
    }
  }
  if (verifyModule(*module))
    return fail("raster replay produced invalid LLVM IR");
  return true;
}
} // namespace scorpi
