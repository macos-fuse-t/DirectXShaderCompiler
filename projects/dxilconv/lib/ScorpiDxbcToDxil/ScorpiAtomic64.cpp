#include "ScorpiAtomic64.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilResource.h"
#include "dxc/DXIL/DxilResourceBinding.h"
#include "dxc/DXIL/DxilResourceProperties.h"
#include "dxc/DXIL/DxilShaderModel.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include <stdexcept>
#include <vector>

using namespace llvm;
using namespace hlsl;
using namespace hlsl::DXIL;
namespace {
OpCode Opcode(CallInst *call) {
  if (call->getCalledFunction() &&
      call->getCalledFunction()->getName().startswith("dx.op.") &&
      call->getNumArgOperands())
    if (auto *op = dyn_cast<ConstantInt>(call->getArgOperand(0)))
      return static_cast<OpCode>(op->getZExtValue());
  return OpCode::NumOpCodes;
}
ResourceKind Kind(Value *handle, DxilModule &dm) {
  auto *call = dyn_cast<CallInst>(handle);
  if (call && Opcode(call) == OpCode::AnnotateHandle)
    return resource_helper::loadPropsFromConstant(
               *cast<Constant>(call->getArgOperand(2)))
        .getResourceKind();
  if (call && Opcode(call) == OpCode::CreateHandle) {
    auto *cls = dyn_cast<ConstantInt>(call->getArgOperand(1));
    auto *id = dyn_cast<ConstantInt>(call->getArgOperand(2));
    if (cls && id && cls->getZExtValue() == unsigned(ResourceClass::UAV) &&
        id->getZExtValue() < dm.GetUAVs().size())
      return dm.GetUAV(id->getZExtValue()).GetKind();
  }
  return ResourceKind::Invalid;
}
bool HasTableBounds(Value *handle, DxilModule &dm,
                    const ScorpiTypedBufferBinding *bindings, size_t count) {
  auto *call = dyn_cast<CallInst>(handle);
  if (!call)
    throw std::runtime_error(
        "64-bit atomic resource handle flow is unsupported");
  if (Opcode(call) == OpCode::AnnotateHandle)
    return HasTableBounds(call->getArgOperand(1), dm, bindings, count);
  uint32_t space, reg;
  if (Opcode(call) == OpCode::CreateHandleFromBinding) {
    auto binding = resource_helper::loadBindingFromConstant(
        *cast<Constant>(call->getArgOperand(1)));
    space = binding.spaceID;
    reg = binding.rangeLowerBound;
  } else if (Opcode(call) == OpCode::CreateHandle) {
    auto &resource =
        dm.GetUAV(cast<ConstantInt>(call->getArgOperand(2))->getZExtValue());
    space = resource.GetSpaceID();
    reg = resource.GetLowerBound();
  } else if (Opcode(call) == OpCode::CreateHandleFromHeap) {
    return true;
  } else {
    throw std::runtime_error(
        "64-bit atomic resource handle flow is unsupported");
  }
  for (size_t i = 0; i < count; ++i) {
    const auto &r = bindings[i];
    if (r.resource_class == unsigned(ResourceClass::UAV) && r.space == space &&
        reg >= r.base_register && uint64_t(reg) - r.base_register < r.count)
      return r.table_offset != UINT32_MAX;
  }
  // An unresolved binding must not silently bypass table bounds checking.
  throw std::runtime_error(
      "64-bit atomic UAV is not covered by a visible root binding");
}
} // namespace

bool ScorpiHasAtomic64(DxilModule &dm) {
  for (auto &fn : *dm.GetModule())
    for (auto &block : fn)
      for (auto &inst : block)
        if (auto *call = dyn_cast<CallInst>(&inst))
          if (call->getType()->isIntegerTy(64) &&
              (Opcode(call) == OpCode::AtomicBinOp ||
               Opcode(call) == OpCode::AtomicCompareExchange))
            return true;
  return false;
}

bool ScorpiLowerAtomic64(DxilModule &dm, uint32_t lock_space,
                         const ScorpiTypedBufferBinding *bindings,
                         size_t binding_count) {
  std::vector<CallInst *> calls;
  for (auto &fn : *dm.GetModule())
    for (auto &block : fn)
      for (auto &inst : block) {
        auto *call = dyn_cast<CallInst>(&inst);
        if (call && call->getType()->isIntegerTy(64) &&
            (Opcode(call) == OpCode::AtomicBinOp ||
             Opcode(call) == OpCode::AtomicCompareExchange)) {
          const auto kind = Kind(call->getArgOperand(1), dm);
          if (kind != ResourceKind::RawBuffer &&
              kind != ResourceKind::StructuredBuffer)
            throw std::runtime_error(
                "64-bit atomic emulation requires a raw or structured UAV");
          calls.push_back(call);
        }
      }
  if (calls.empty())
    return false;
  if (lock_space == UINT32_MAX)
    throw std::runtime_error("64-bit atomic lock binding unavailable");
  auto &ctx = dm.GetModule()->getContext();
  auto lock = std::make_unique<DxilResource>();
  lock->SetID(dm.GetUAVs().size());
  lock->SetRW(true);
  lock->SetKind(ResourceKind::RawBuffer);
  lock->SetSpaceID(lock_space);
  lock->SetLowerBound(0);
  lock->SetRangeSize(1);
  lock->SetGloballyCoherent(true);
  lock->SetGlobalName("ScorpiAtomic64Lock");
  lock->SetGlobalSymbol(UndefValue::get(
      StructType::create(ctx, {Type::getInt32Ty(ctx)}, "ScorpiAtomic64Lock")
          ->getPointerTo()));
  const auto lockID = dm.AddUAV(std::move(lock));
  auto callOp = [&](IRBuilder<> &b, OpCode op, Type *type,
                    std::vector<Value *> args) {
    args.insert(args.begin(), b.getInt32(unsigned(op)));
    return b.CreateCall(dm.GetOP()->GetOpFunc(op, type), args);
  };
  for (auto *original : calls) {
    auto *fn = original->getParent()->getParent();
    IRBuilder<> entry(&*fn->getEntryBlock().getFirstInsertionPt());
    Value *lockHandle;
    auto &resource = dm.GetUAV(lockID);
    if (dm.GetShaderModel()->GetMinor() >= 6) {
      auto binding = resource_helper::loadBindingFromResourceBase(&resource);
      auto *raw =
          callOp(entry, OpCode::CreateHandleFromBinding, entry.getVoidTy(),
                 {resource_helper::getAsConstant(
                      binding, dm.GetOP()->GetResourceBindingType(),
                      *dm.GetShaderModel()),
                  entry.getInt32(0), entry.getFalse()});
      auto props = resource_helper::loadPropsFromResourceBase(&resource);
      lockHandle =
          callOp(entry, OpCode::AnnotateHandle, entry.getVoidTy(),
                 {raw, resource_helper::getAsConstant(
                           props, dm.GetOP()->GetResourcePropertiesType(),
                           *dm.GetShaderModel())});
    } else {
      lockHandle =
          callOp(entry, OpCode::CreateHandle, entry.getVoidTy(),
                 {entry.getInt8(unsigned(ResourceClass::UAV)),
                  entry.getInt32(lockID), entry.getInt32(0), entry.getFalse()});
    }
    const bool compare = Opcode(original) == OpCode::AtomicCompareExchange;
    auto *handle = original->getArgOperand(1);
    const unsigned coord = compare ? 2 : 3;
    auto *x = original->getArgOperand(coord);
    auto *y = original->getArgOperand(coord + 1);
    auto *z = original->getArgOperand(coord + 2);
    auto *operand = original->getArgOperand(6);
    auto *before = original->getParent();
    auto *after = before->splitBasicBlock(original, "atomic64.end");
    before->getTerminator()->eraseFromParent();
    auto *retry = BasicBlock::Create(ctx, "atomic64.retry", fn, after);
    auto *attempt = BasicBlock::Create(ctx, "atomic64.attempt", fn, after);
    auto *acquired = BasicBlock::Create(ctx, "atomic64.acquired", fn, after);
    auto *latch = BasicBlock::Create(ctx, "atomic64.latch", fn, after);
    IRBuilder<> b(before);
    const bool raw = Kind(handle, dm) == ResourceKind::RawBuffer;
    auto *offset = raw ? x : y;
    Value *valid = b.CreateICmpULE(offset, b.getInt32(UINT32_MAX - 4));
    // Root UAVs carry an address, not a view length. GetDimensions is only
    // defined for table/heap descriptors; check the full eight-byte footprint
    // there so an OOB high word cannot leave a partial low-word update.
    if (HasTableBounds(handle, dm, bindings, binding_count)) {
      auto *dimensions = callOp(b, OpCode::GetDimensions, b.getInt32Ty(),
                                {handle, b.getInt32(0)});
      auto *count = b.CreateExtractValue(dimensions, 0);
      auto *bytes = raw ? count : b.CreateExtractValue(dimensions, 1);
      valid = b.CreateAnd(
          b.CreateICmpUGE(bytes, b.getInt32(8)),
          b.CreateICmpULE(offset, b.CreateSub(bytes, b.getInt32(8))));
      if (!raw)
        valid = b.CreateAnd(valid, b.CreateICmpULT(x, count));
    }
    b.CreateCondBr(valid, retry, after);
    b.SetInsertPoint(retry);
    auto *finished = b.CreatePHI(b.getInt1Ty(), 2, "atomic64.finished");
    auto *saved = b.CreatePHI(b.getInt64Ty(), 2, "atomic64.saved");
    finished->addIncoming(b.getFalse(), before);
    saved->addIncoming(UndefValue::get(b.getInt64Ty()), before);
    auto *undef = UndefValue::get(b.getInt32Ty());
    b.CreateCondBr(finished, latch, attempt);
    b.SetInsertPoint(attempt);
    auto *held = callOp(b, OpCode::AtomicCompareExchange, b.getInt32Ty(),
                        {lockHandle, b.getInt32(0), undef, undef, b.getInt32(0),
                         b.getInt32(1)});
    b.CreateCondBr(b.CreateICmpEQ(held, b.getInt32(0)), acquired, latch);
    b.SetInsertPoint(acquired);
    // Acquire/release are both inside the successful lane's branch. No lane
    // spins with the lock held while another lane waits at reconvergence.
    auto fence = [&]() {
      callOp(b, OpCode::Barrier, b.getVoidTy(),
             {b.getInt32(unsigned(BarrierMode::UAVFenceGlobal))});
    };
    fence();
    Value *highX = raw ? b.CreateAdd(x, b.getInt32(4)) : x;
    Value *highY = raw ? y : b.CreateAdd(y, b.getInt32(4));
    auto word = [&](AtomicBinOpCode op, Value *wx, Value *wy, Value *value) {
      return callOp(b, OpCode::AtomicBinOp, b.getInt32Ty(),
                    {handle, b.getInt32(unsigned(op)), wx, wy, z, value});
    };
    auto *low = word(AtomicBinOpCode::Or, x, y, b.getInt32(0));
    auto *high = word(AtomicBinOpCode::Or, highX, highY, b.getInt32(0));
    auto *old = b.CreateOr(b.CreateZExt(low, b.getInt64Ty()),
                           b.CreateShl(b.CreateZExt(high, b.getInt64Ty()), 32));
    Value *value = nullptr;
    if (compare) {
      value = b.CreateSelect(b.CreateICmpEQ(old, original->getArgOperand(5)),
                             operand, old);
    } else {
      auto operation = static_cast<AtomicBinOpCode>(
          cast<ConstantInt>(original->getArgOperand(2))->getZExtValue());
      switch (operation) {
      case AtomicBinOpCode::Add:
        value = b.CreateAdd(old, operand);
        break;
      case AtomicBinOpCode::And:
        value = b.CreateAnd(old, operand);
        break;
      case AtomicBinOpCode::Or:
        value = b.CreateOr(old, operand);
        break;
      case AtomicBinOpCode::Xor:
        value = b.CreateXor(old, operand);
        break;
      case AtomicBinOpCode::IMin:
        value = b.CreateSelect(b.CreateICmpSLT(old, operand), old, operand);
        break;
      case AtomicBinOpCode::IMax:
        value = b.CreateSelect(b.CreateICmpSGT(old, operand), old, operand);
        break;
      case AtomicBinOpCode::UMin:
        value = b.CreateSelect(b.CreateICmpULT(old, operand), old, operand);
        break;
      case AtomicBinOpCode::UMax:
        value = b.CreateSelect(b.CreateICmpUGT(old, operand), old, operand);
        break;
      case AtomicBinOpCode::Exchange:
        value = operand;
        break;
      default:
        throw std::runtime_error("unknown 64-bit atomic operation");
      }
    }
    word(AtomicBinOpCode::Exchange, x, y, b.CreateTrunc(value, b.getInt32Ty()));
    word(AtomicBinOpCode::Exchange, highX, highY,
         b.CreateTrunc(b.CreateLShr(value, 32), b.getInt32Ty()));
    fence();
    callOp(b, OpCode::AtomicBinOp, b.getInt32Ty(),
           {lockHandle, b.getInt32(unsigned(AtomicBinOpCode::Exchange)),
            b.getInt32(0), undef, undef, b.getInt32(0)});
    b.CreateBr(latch);
    b.SetInsertPoint(latch);
    auto *done = b.CreatePHI(b.getInt1Ty(), 3);
    done->addIncoming(finished, retry);
    done->addIncoming(b.getFalse(), attempt);
    done->addIncoming(b.getTrue(), acquired);
    auto *result = b.CreatePHI(b.getInt64Ty(), 3);
    result->addIncoming(saved, retry);
    result->addIncoming(saved, attempt);
    result->addIncoming(old, acquired);
    finished->addIncoming(done, latch);
    saved->addIncoming(result, latch);
    // Keep completed lanes in the wave until all peers finish. Otherwise MSC
    // can legally simplify the per-lane loop into acquire-all then release-all,
    // which deadlocks a SIMD wave. This convergent vote pins release in-loop.
    auto *allDone = callOp(b, OpCode::WaveAllTrue, b.getVoidTy(), {done});
    b.CreateCondBr(allDone, after, retry);
    auto *bounded =
        PHINode::Create(b.getInt64Ty(), 2, "atomic64.bounded", original);
    bounded->addIncoming(b.getInt64(0), before);
    bounded->addIncoming(result, latch);
    original->replaceAllUsesWith(bounded);
    original->eraseFromParent();
  }
  return true;
}
