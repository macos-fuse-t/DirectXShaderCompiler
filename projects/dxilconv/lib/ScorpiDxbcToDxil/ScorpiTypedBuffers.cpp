#include "ScorpiTypedBuffers.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilResource.h"
#include "dxc/DXIL/DxilResourceBinding.h"
#include "dxc/DXIL/DxilResourceProperties.h"
#include "dxc/DXIL/DxilShaderModel.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/Scalar.h"
#include <map>
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

class Lowering {
  DxilModule &dm;
  const ScorpiTypedBufferBinding *bindings;
  size_t count;
  uint32_t space, selfOffset, resourceID;
  std::map<Function *, Value *> metadataHandles;
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
public:
  Lowering(DxilModule &module, const ScorpiTypedBufferBinding *ranges,
      size_t n, uint32_t s, uint32_t self) : dm(module), bindings(ranges), count(n), space(s), selfOffset(self) {
    auto &ctx = dm.GetModule()->getContext();
    auto resource = std::make_unique<DxilResource>();
    resource->SetID(dm.GetSRVs().size()); resource->SetRW(false);
    resource->SetKind(ResourceKind::RawBuffer); resource->SetSpaceID(space);
    resource->SetLowerBound(0); resource->SetRangeSize(1);
    resource->SetGlobalName("ScorpiTypedBufferMetadata");
    resource->SetGlobalSymbol(UndefValue::get(StructType::create(ctx,
        {Type::getInt32Ty(ctx)}, "ScorpiTypedBufferMetadata")->getPointerTo()));
    resourceID = dm.AddSRV(std::move(resource));
  }

  Value *Word(Value *handle, unsigned word) {
    auto key = std::make_pair(handle, word);
    auto found = words.find(key);
    if (found != words.end()) return found->second;
    if (auto *phi = dyn_cast<PHINode>(handle)) {
      auto *value = PHINode::Create(Type::getInt32Ty(phi->getContext()), phi->getNumIncomingValues(),
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
    const ScorpiTypedBufferBinding *range = nullptr;
    for (size_t i = 0; i < count; ++i) {
      const auto &r = bindings[i];
      if (cls == r.resource_class && resourceSpace == r.space && lower >= r.base_register &&
          uint64_t(lower) - r.base_register < r.count) { range = &r; break; }
    }
    if (!range) throw std::runtime_error("typed buffer not covered by a visible descriptor-table range");
    auto *metadata = MetadataHandle(call->getParent()->getParent());
    IRBuilder<> b(call->getNextNode());
    // All table entries live in the same <4GiB allocation. Low-word subtraction
    // is deliberately modular, including when a GPU address crosses 4GiB.
    auto *relative = b.CreateSub(Load(b, metadata, b.getInt32(range->root_offset)),
        Load(b, metadata, b.getInt32(selfOffset)));
    auto *slot = b.CreateAdd(b.CreateSub(index, b.getInt32(range->base_register)), b.getInt32(range->table_offset));
    auto *offset = b.CreateAdd(relative, b.CreateAdd(b.CreateMul(slot, b.getInt32(24)), b.getInt32(word * 4)));
    return words[key] = Load(b, metadata, offset);
  }

  void Rewrite(CallInst *call) {
    auto *handle = call->getArgOperand(1);
    auto *word = Word(handle, 5);
    IRBuilder<> b(call);
    auto *prefix = b.CreateAnd(word, b.getInt32(255), "scorpi.view.prefix");
    if (Opcode(call) == OpCode::GetDimensions) {
      b.SetInsertPoint(call->getNextNode());
      auto *width = cast<Instruction>(b.CreateExtractValue(call, 0));
      auto *fixed = cast<Instruction>(b.CreateInsertValue(call, b.CreateSub(width, prefix), 0));
      call->replaceAllUsesWith(fixed);
      width->setOperand(0, call); fixed->setOperand(0, call);
      return;
    }
    unsigned indexOperand = Opcode(call) == OpCode::AtomicBinOp ? 3 : 2;
    auto *index = call->getArgOperand(indexOperand);
    auto *dimensions = Call(b, OpCode::GetDimensions, b.getInt32Ty(), {handle, b.getInt32(0)});
    auto *length = b.CreateSub(b.CreateExtractValue(dimensions, 0), prefix);
    auto *valid = b.CreateICmpULT(index, length);
    call->setArgOperand(indexOperand, b.CreateSelect(valid, b.CreateAdd(index, prefix), b.getInt32(UINT32_MAX)));
  }

  // Software reduction uses integer texture reads, so every addressing mode,
  // zero-weight exclusion and mip clamp is independent of native sampler
  // reduction support. Ordinary sampling keeps the original instruction.
  void RewriteSample(CallInst *call, ResourceKind kind) {
    const auto opcode = Opcode(call);
    const bool implicit = opcode == OpCode::Sample || opcode == OpCode::SampleBias;
    auto *texture = call->getArgOperand(1), *sampler = call->getArgOperand(2);
    auto *state = Word(sampler, 2), *minimum = Word(sampler, 5), *maximum = Word(sampler, 3);
    auto *bias = Word(sampler, 4);
    auto *resourceMinimum = Word(texture, 4);
    const unsigned axes = kind == ResourceKind::Texture3D ? 3 : 2;
    Value *sourceLod = call->getArgOperand(10);
    if (opcode == OpCode::SampleGrad || implicit) {
      IRBuilder<> b(call);
      auto *size = Call(b, OpCode::GetDimensions, b.getInt32Ty(), {texture,b.getInt32(0)});
      Value *dx2 = ConstantFP::get(b.getFloatTy(),0), *dy2 = dx2;
      for (unsigned axis=0;axis<axes;++axis) {
        auto *n = b.CreateUIToFP(b.CreateExtractValue(size,axis),b.getFloatTy());
        // Derivatives must execute before the sampler-mode branch: descriptor
        // indexing can make that branch non-uniform within a pixel quad.
        auto *gx = implicit ? Call(b,OpCode::DerivCoarseX,b.getFloatTy(),{call->getArgOperand(3+axis)}) : call->getArgOperand(10+axis);
        auto *gy = implicit ? Call(b,OpCode::DerivCoarseY,b.getFloatTy(),{call->getArgOperand(3+axis)}) : call->getArgOperand(13+axis);
        auto *dx = b.CreateFMul(gx,n);
        auto *dy = b.CreateFMul(gy,n);
        dx2 = b.CreateFAdd(dx2,b.CreateFMul(dx,dx));
        dy2 = b.CreateFAdd(dy2,b.CreateFMul(dy,dy));
      }
      auto *rho2 = Call(b,OpCode::FMax,b.getFloatTy(),{dx2,dy2});
      sourceLod = b.CreateFMul(Call(b,OpCode::Log,b.getFloatTy(),{rho2}),ConstantFP::get(b.getFloatTy(),.5));
      if (opcode == OpCode::SampleBias) sourceLod = b.CreateFAdd(sourceLod,call->getArgOperand(10));
    }
    auto *before = call->getParent();
    auto *fn = before->getParent();
    auto *after = before->splitBasicBlock(call->getNextNode(), "scorpi.sample.end");
    auto *normal = BasicBlock::Create(call->getContext(), "scorpi.sample.native", fn, after);
    auto *reduce = BasicBlock::Create(call->getContext(), "scorpi.sample.reduce", fn, after);
    before->getTerminator()->eraseFromParent();
    // Keep native implicit derivatives in the original control-flow context.
    if (!implicit) { call->removeFromParent(); normal->getInstList().push_back(call); }
    IRBuilder<> nb(normal); nb.CreateBr(after);
    IRBuilder<> pre(before);
    pre.CreateCondBr(pre.CreateICmpNE(pre.CreateAnd(state, pre.getInt32(0x100)), pre.getInt32(0)), reduce, normal);
    IRBuilder<> b(reduce);
    auto f = [&](double n) { return ConstantFP::get(b.getFloatTy(), n); };
    auto unary = [&](OpCode op, Value *v) -> Value * { return Call(b, op, b.getFloatTy(), {v}); };
    auto min = [&](Value *a, Value *z) -> Value * { return Call(b, OpCode::FMin, b.getFloatTy(), {a,z}); };
    auto max = [&](Value *a, Value *z) -> Value * { return Call(b, OpCode::FMax, b.getFloatTy(), {a,z}); };
    auto *dims = Call(b, OpCode::GetDimensions, b.getInt32Ty(), {texture, b.getInt32(0)});
    Value *lod = b.CreateFAdd(sourceLod,b.CreateBitCast(bias,b.getFloatTy()));
    // D3D gives MinLOD precedence even when it exceeds MaxLOD.
    lod = max(min(lod, b.CreateBitCast(maximum, b.getFloatTy())), b.CreateBitCast(minimum, b.getFloatTy()));
    lod = max(lod, b.CreateBitCast(resourceMinimum, b.getFloatTy()));
    if (opcode != OpCode::SampleLevel) {
      const unsigned clamp = opcode == OpCode::SampleGrad ? 16 : opcode == OpCode::SampleBias ? 11 : 10;
      if (!isa<UndefValue>(call->getArgOperand(clamp))) lod = max(lod,call->getArgOperand(clamp));
    }
    lod = min(max(lod, f(0)), b.CreateUIToFP(b.CreateSub(b.CreateExtractValue(dims, 3), b.getInt32(1)), b.getFloatTy()));
    auto *mipLinear = b.CreateICmpNE(b.CreateAnd(state, b.getInt32(1)), b.getInt32(0));
    auto *linear = b.CreateICmpNE(b.CreateAnd(state,
        b.CreateSelect(b.CreateFCmpOGT(lod, f(0)), b.getInt32(0x10), b.getInt32(4))), b.getInt32(0));
    auto *isMin = b.CreateICmpEQ(b.CreateAnd(state, b.getInt32(0x180)), b.getInt32(0x100));
    auto *base = unary(OpCode::Round_ni, lod);
    auto *mip0 = b.CreateFPToUI(b.CreateSelect(mipLinear, base, unary(OpCode::Round_ni, b.CreateFAdd(lod,f(.5)))), b.getInt32Ty());
    auto *useMip1 = b.CreateAnd(mipLinear, b.CreateFCmpONE(lod, base));
    const bool array = kind == ResourceKind::Texture2DArray;
    auto *elementType = call->getType()->getStructElementType(0);
    Value *result = UndefValue::get(call->getType());
    auto combine = [&](Value *a, Value *z) -> Value * {
      return b.CreateSelect(isMin, Call(b, OpCode::FMin, elementType, {a,z}), Call(b, OpCode::FMax, elementType, {a,z}));
    };
    for (unsigned mip = 0; mip < 2; ++mip) {
      auto *level = mip == 0 ? mip0 : b.CreateAdd(mip0, b.CreateZExt(useMip1, b.getInt32Ty()));
      auto *size = Call(b, OpCode::GetDimensions, b.getInt32Ty(), {texture, level});
      Value *length[3], *start[3], *upper[3];
      for (unsigned axis = 0; axis < axes; ++axis) {
        length[axis] = b.CreateExtractValue(size, axis);
        auto *coord = b.CreateFMul(call->getArgOperand(3+axis), b.CreateUIToFP(length[axis], b.getFloatTy()));
        coord = b.CreateFSub(coord, b.CreateSelect(linear, f(.5), f(0)));
        auto *floor = unary(OpCode::Round_ni, coord);
        start[axis] = b.CreateFPToSI(floor, b.getInt32Ty());
        if (!isa<UndefValue>(call->getArgOperand(7+axis)))
          start[axis] = b.CreateAdd(start[axis], call->getArgOperand(7+axis));
        upper[axis] = b.CreateAnd(linear, b.CreateFCmpONE(coord, floor));
      }
      for (unsigned tap = 0; tap < (1u << axes); ++tap) {
        Value *active = mip == 0 ? b.getTrue() : useMip1;
        Value *outside = b.getFalse();
        Value *coords[3] = {UndefValue::get(b.getInt32Ty()),UndefValue::get(b.getInt32Ty()),UndefValue::get(b.getInt32Ty())};
        for (unsigned axis = 0; axis < axes; ++axis) {
          const bool high = (tap & (1u << axis)) != 0;
          if (high) active = b.CreateAnd(active, upper[axis]);
          auto *index = high ? b.CreateAdd(start[axis], b.getInt32(1)) : start[axis];
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
        auto *loaded = Call(b,OpCode::TextureLoad,elementType,
            {texture,level,coords[0],coords[1],coords[2],b.getInt32(0),b.getInt32(0),b.getInt32(0)});
        auto *border = b.CreateAnd(b.CreateLShr(state,b.getInt32(18)),b.getInt32(3));
        for (unsigned c = 0; c < 4; ++c) {
          auto *borderValue = b.CreateSelect(c == 3 ? b.CreateICmpNE(border,b.getInt32(0)) : b.CreateICmpEQ(border,b.getInt32(2)),
              ConstantFP::get(elementType,1),ConstantFP::get(elementType,0));
          auto *value = b.CreateSelect(outside,borderValue,b.CreateExtractValue(loaded,c));
          if (mip || tap) {
            auto *old = b.CreateExtractValue(result,c);
            value = b.CreateSelect(active,combine(old,value),old);
          }
          result = b.CreateInsertValue(result,value,c);
        }
        // Non-sparse resources are fully mapped. Sparse residency reduction is
        // added with the reserved-resource backend, before its tier is exposed.
        result = b.CreateInsertValue(result,b.CreateExtractValue(loaded,4),4);
      }
    }
    b.CreateBr(after);
    auto *phi = PHINode::Create(call->getType(),2,"scorpi.sample.result",&*after->begin());
    call->replaceAllUsesWith(phi);
    phi->addIncoming(call,normal); phi->addIncoming(result,reduce);
  }
};
}

bool ScorpiLowerTypedBuffers(DxilModule &dm, const ScorpiTypedBufferBinding *bindings,
    size_t count, uint32_t space, uint32_t selfOffset) {
  std::vector<CallInst *> calls;
  std::vector<std::pair<CallInst *, ResourceKind>> samples;
  for (auto &f : *dm.GetModule()) for (auto &block : f) for (auto &inst : block) {
    auto *call = dyn_cast<CallInst>(&inst);
    const auto op = Opcode(call);
    if (op == OpCode::SampleLevel || op == OpCode::SampleGrad || op == OpCode::Sample || op == OpCode::SampleBias) {
      std::set<Value *> seen;
      auto kind = Kind(call->getArgOperand(1), dm, seen);
      if (kind == ResourceKind::Texture2D || kind == ResourceKind::Texture2DArray || kind == ResourceKind::Texture3D)
        samples.emplace_back(call,kind);
    }
    if (op != OpCode::BufferLoad && op != OpCode::BufferStore && op != OpCode::GetDimensions &&
        op != OpCode::AtomicBinOp && op != OpCode::AtomicCompareExchange) continue;
    std::set<Value *> seen;
    if (Kind(call->getArgOperand(1), dm, seen) == ResourceKind::TypedBuffer) calls.push_back(call);
  }
  if (calls.empty() && samples.empty()) return false;
  if (!bindings || !count || space == UINT32_MAX)
    throw std::runtime_error("typed-buffer metadata root binding missing");
  Lowering lowering(dm, bindings, count, space, selfOffset);
  for (auto *call : calls) lowering.Rewrite(call);
  for (auto &sample : samples) lowering.RewriteSample(sample.first,sample.second);
  if (!samples.empty()) {
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
