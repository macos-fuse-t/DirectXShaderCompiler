#include "ScorpiTypedBuffers.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilResource.h"
#include "dxc/DXIL/DxilResourceBinding.h"
#include "dxc/DXIL/DxilResourceProperties.h"
#include "dxc/DXIL/DxilShaderModel.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
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
  std::map<Value *, Value *> prefixes;

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

  Value *Prefix(Value *handle) {
    auto found = prefixes.find(handle);
    if (found != prefixes.end()) return found->second;
    if (auto *phi = dyn_cast<PHINode>(handle)) {
      auto *value = PHINode::Create(Type::getInt32Ty(phi->getContext()), phi->getNumIncomingValues(),
          "scorpi.view.prefix", phi);
      prefixes[handle] = value;
      for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i)
        value->addIncoming(Prefix(phi->getIncomingValue(i)), phi->getIncomingBlock(i));
      return value;
    }
    if (auto *select = dyn_cast<SelectInst>(handle)) {
      auto *a = Prefix(select->getTrueValue()), *b = Prefix(select->getFalseValue());
      IRBuilder<> builder(select->getNextNode());
      return prefixes[handle] = builder.CreateSelect(select->getCondition(), a, b);
    }
    auto *call = dyn_cast<CallInst>(handle);
    if (!call) throw std::runtime_error("unsupported typed-buffer handle");
    if (Opcode(call) == OpCode::AnnotateHandle) return prefixes[handle] = Prefix(call->getArgOperand(1));
    uint32_t cls, resourceSpace, lower;
    Value *index;
    if (Opcode(call) == OpCode::CreateHandle) {
      cls = cast<ConstantInt>(call->getArgOperand(1))->getZExtValue();
      unsigned id = cast<ConstantInt>(call->getArgOperand(2))->getZExtValue();
      auto &resource = cls == unsigned(ResourceClass::SRV) ? dm.GetSRV(id) : dm.GetUAV(id);
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
    auto *offset = b.CreateAdd(relative, b.CreateAdd(b.CreateMul(slot, b.getInt32(24)), b.getInt32(20)));
    return prefixes[handle] = b.CreateAnd(Load(b, metadata, offset), b.getInt32(255), "scorpi.view.prefix");
  }

  void Rewrite(CallInst *call) {
    auto *handle = call->getArgOperand(1);
    auto *prefix = Prefix(handle);
    IRBuilder<> b(call);
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
};
}

bool ScorpiLowerTypedBuffers(DxilModule &dm, const ScorpiTypedBufferBinding *bindings,
    size_t count, uint32_t space, uint32_t selfOffset) {
  std::vector<CallInst *> calls;
  for (auto &f : *dm.GetModule()) for (auto &block : f) for (auto &inst : block) {
    auto *call = dyn_cast<CallInst>(&inst);
    const auto op = Opcode(call);
    if (op != OpCode::BufferLoad && op != OpCode::BufferStore && op != OpCode::GetDimensions &&
        op != OpCode::AtomicBinOp && op != OpCode::AtomicCompareExchange) continue;
    std::set<Value *> seen;
    if (Kind(call->getArgOperand(1), dm, seen) == ResourceKind::TypedBuffer) calls.push_back(call);
  }
  if (calls.empty()) return false;
  if (!bindings || !count || space == UINT32_MAX)
    throw std::runtime_error("typed-buffer metadata root binding missing");
  Lowering lowering(dm, bindings, count, space, selfOffset);
  for (auto *call : calls) lowering.Rewrite(call);
  return true;
}
