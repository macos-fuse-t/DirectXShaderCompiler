#include "ScorpiAtomic64.h"
#include "ScorpiTypedBuffers.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilInstructions.h"
#include <climits>
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
constexpr unsigned kWordBytes = sizeof(uint32_t);
constexpr unsigned kWordBits = kWordBytes * CHAR_BIT;
constexpr unsigned kAtomicBytes = sizeof(uint64_t);
constexpr unsigned kAtomicBits = kAtomicBytes * CHAR_BIT;
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

namespace {
bool NativeUnsignedMinMax(CallInst *call, ResourceKind kind) {
  if (!call || !call->use_empty() || Opcode(call)!=OpCode::AtomicBinOp) return false;
  switch (kind) {
  case ResourceKind::RawBuffer: case ResourceKind::StructuredBuffer: case ResourceKind::TypedBuffer:
  case ResourceKind::Texture1D: case ResourceKind::Texture1DArray:
  case ResourceKind::Texture2D: case ResourceKind::Texture2DArray:
  case ResourceKind::Texture3D: break;
  default: return false;
  }
  const auto op=cast<ConstantInt>(DxilInst_AtomicBinOp(call).get_atomicOp())->getZExtValue();
  return op==unsigned(AtomicBinOpCode::UMin) || op==unsigned(AtomicBinOpCode::UMax);
}
}
uint32_t ScorpiClassifyAtomic64(Module &module) {
  uint32_t required=0;
  for (auto &fn:module) for(auto &block:fn) for(auto &inst:block) {
    if (auto *call=dyn_cast<CallInst>(&inst)) {
      if (!call->getType()->isIntegerTy(kAtomicBits) || (Opcode(call)!=OpCode::AtomicBinOp && Opcode(call)!=OpCode::AtomicCompareExchange)) continue;
      auto *annotation=dyn_cast<CallInst>(call->getArgOperand(1));
      if (!annotation || Opcode(annotation)!=OpCode::AnnotateHandle) { required|=SCORPI_ATOMIC64_BUFFER;continue; }
      auto kind=resource_helper::loadPropsFromConstant(*cast<Constant>(DxilInst_AnnotateHandle(annotation).get_props())).getResourceKind();
      required |= kind==ResourceKind::RawBuffer || kind==ResourceKind::StructuredBuffer ? SCORPI_ATOMIC64_BUFFER :
          NativeUnsignedMinMax(call,kind) ? SCORPI_ATOMIC64_NATIVE_TEXTURE_MINMAX : SCORPI_ATOMIC64_TYPED_EMULATION;
    } else if (auto *rmw=dyn_cast<AtomicRMWInst>(&inst)) {
      if (rmw->getType()->isIntegerTy(kAtomicBits)) required|=SCORPI_ATOMIC64_GROUPSHARED;
    } else if (auto *cmp=dyn_cast<AtomicCmpXchgInst>(&inst)) {
      if (cmp->getCompareOperand()->getType()->isIntegerTy(kAtomicBits)) required|=SCORPI_ATOMIC64_GROUPSHARED;
    }
  }
  return required;
}
bool ScorpiHasAtomic64(Module &module) { return ScorpiClassifyAtomic64(module)!=0; }

bool ScorpiLowerAtomic64(DxilModule &dm, uint32_t lock_space,
                         const ScorpiTypedBufferBinding *bindings,
                         size_t binding_count, bool native_minmax) {
  std::vector<Instruction *> calls;
  for (auto &fn : *dm.GetModule()) for (auto &block : fn) for (auto &inst : block) {
    if (auto *call=dyn_cast<CallInst>(&inst)) {
      if (call->getType()->isIntegerTy(kAtomicBits) && (Opcode(call)==OpCode::AtomicBinOp || Opcode(call)==OpCode::AtomicCompareExchange)) {
        auto kind=Kind(call->getArgOperand(1),dm);
        // Even native min/max participates in the same location lock as
        // emulated operations in other shaders and command queues.
        if (kind!=ResourceKind::RawBuffer && kind!=ResourceKind::StructuredBuffer && kind!=ResourceKind::TypedBuffer &&
            kind!=ResourceKind::Texture1D && kind!=ResourceKind::Texture1DArray && kind!=ResourceKind::Texture2D &&
            kind!=ResourceKind::Texture2DArray && kind!=ResourceKind::Texture3D)
          throw std::runtime_error("64-bit atomic resource dimension unsupported");
        calls.push_back(call);
      }
    } else if (auto *rmw=dyn_cast<AtomicRMWInst>(&inst)) {
      if (rmw->getType()->isIntegerTy(kAtomicBits)) {
        if (rmw->getPointerOperand()->getType()->getPointerAddressSpace()!=kTGSMAddrSpace) throw std::runtime_error("64-bit pointer atomic is not groupshared");
        calls.push_back(rmw);
      }
    } else if (auto *cmp=dyn_cast<AtomicCmpXchgInst>(&inst)) {
      if (cmp->getCompareOperand()->getType()->isIntegerTy(kAtomicBits)) {
        if (cmp->getPointerOperand()->getType()->getPointerAddressSpace()!=kTGSMAddrSpace) throw std::runtime_error("64-bit pointer compare exchange is not groupshared");
        calls.push_back(cmp);
      }
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
    auto *call=dyn_cast<CallInst>(original);
    auto *sharedRmw=dyn_cast<AtomicRMWInst>(original);
    auto *sharedCmp=dyn_cast<AtomicCmpXchgInst>(original);
    const bool shared=!call, compare=sharedCmp || (call && Opcode(call)==OpCode::AtomicCompareExchange);
    Value *handle=call ? call->getArgOperand(1) : nullptr;
    const unsigned coord = compare ? DxilInst_AtomicCompareExchange::arg_offset0 : DxilInst_AtomicBinOp::arg_offset0;
    Value *x=call ? call->getArgOperand(coord) : nullptr;
    Value *y=call ? call->getArgOperand(coord+1) : nullptr;
    Value *z=call ? call->getArgOperand(coord+2) : nullptr;
    Value *operand=call ? call->getArgOperand(DxilInst_AtomicBinOp::arg_newValue) : sharedRmw ? sharedRmw->getValOperand() : sharedCmp->getNewValOperand();
    Value *comparison=compare ? (call ? call->getArgOperand(DxilInst_AtomicCompareExchange::arg_compareValue) : sharedCmp->getCompareOperand()) : nullptr;
    const auto kind=shared ? ResourceKind::Invalid : Kind(handle,dm);
    Value *origin = !shared ? ScorpiAtomic64Origin(dm,handle,bindings,binding_count,lock_space) : nullptr;
    auto *before = original->getParent();
    auto *after = before->splitBasicBlock(original, "atomic64.end");
    before->getTerminator()->eraseFromParent();
    auto *setup = BasicBlock::Create(ctx, "atomic64.setup", fn, after);
    auto *ready = BasicBlock::Create(ctx, "atomic64.ready", fn, after);
    auto *retry = BasicBlock::Create(ctx, "atomic64.retry", fn, after);
    auto *attempt = BasicBlock::Create(ctx, "atomic64.attempt", fn, after);
    auto *tryLock = BasicBlock::Create(ctx, "atomic64.try.lock", fn, after);
    auto *acquired = BasicBlock::Create(ctx, "atomic64.acquired", fn, after);
    auto *latch = BasicBlock::Create(ctx, "atomic64.latch", fn, after);
    IRBuilder<> b(before);
    const bool raw=kind==ResourceKind::RawBuffer;
    const bool structured=kind==ResourceKind::StructuredBuffer;
    const bool typed=!shared && !raw && !structured;
    const bool texture=typed && kind!=ResourceKind::TypedBuffer;
    Value *valid=b.getTrue();
    if (raw || structured) {
      auto *offset=raw ? x : y;
      valid=b.CreateICmpULE(offset,b.getInt32(UINT32_MAX-kWordBytes));
      if (HasTableBounds(handle,dm,bindings,binding_count)) {
        auto *dimensions=callOp(b,OpCode::GetDimensions,b.getInt32Ty(),{handle,b.getInt32(0)});
        auto *count=b.CreateExtractValue(dimensions,0);
        auto *bytes=raw ? count : b.CreateExtractValue(dimensions,1);
        valid=b.CreateAnd(b.CreateICmpUGE(bytes,b.getInt32(kAtomicBytes)),b.CreateICmpULE(offset,b.CreateSub(bytes,b.getInt32(kAtomicBytes))));
        if (!raw) valid=b.CreateAnd(valid,b.CreateICmpULT(x,count));
      }
    }
    Value *lockOffset = b.getInt32(0); // Group-shared fallback domain.
    if (raw || structured || kind == ResourceKind::TypedBuffer) {
      Value *position = x;
      if (structured) {
        unsigned stride = 0;
        auto *annotation = dyn_cast<CallInst>(handle);
        if (annotation && Opcode(annotation) == OpCode::AnnotateHandle)
          stride = resource_helper::loadPropsFromConstant(*cast<Constant>(annotation->getArgOperand(2))).getElementStride();
        else if (annotation && Opcode(annotation) == OpCode::CreateHandle)
          stride = dm.GetUAV(cast<ConstantInt>(annotation->getArgOperand(2))->getZExtValue()).GetElementStride();
        if (!stride) throw std::runtime_error("atomic64 structured stride unavailable");
        position = b.CreateAdd(b.CreateMul(x, b.getInt32(stride)), y);
      } else if (kind == ResourceKind::TypedBuffer) position = b.CreateMul(x, b.getInt32(kAtomicBytes));
      auto *word = b.CreateAnd(b.CreateLShr(b.CreateAdd(origin, position), b.getInt32(3)),
          b.getInt32(SCORPI_ATOMIC64_STRIPES - 1));
      lockOffset = b.CreateMul(b.CreateAdd(word, b.getInt32(1)), b.getInt32(kWordBytes * SCORPI_ATOMIC64_WORDS_PER_STRIPE));
    } else if (texture) {
      auto *dimensions = callOp(b, OpCode::GetDimensions, b.getInt32Ty(), {handle,b.getInt32(0)});
      auto *width = b.CreateExtractValue(dimensions,0);
      Value *height = kind == ResourceKind::Texture1D || kind == ResourceKind::Texture1DArray ?
          b.getInt32(1) : b.CreateExtractValue(dimensions,1);
      Value *row = kind == ResourceKind::Texture1D || kind == ResourceKind::Texture1DArray ? b.getInt32(0) : y;
      Value *layer = kind == ResourceKind::Texture1DArray ? y :
          kind == ResourceKind::Texture2DArray || kind == ResourceKind::Texture3D ? z : b.getInt32(0);
      auto *texel = b.CreateAdd(x,b.CreateMul(width,b.CreateAdd(row,b.CreateMul(height,b.CreateAdd(layer,origin)))));
      auto *stripe = b.CreateAnd(texel,b.getInt32(SCORPI_ATOMIC64_STRIPES-1));
      lockOffset = b.CreateMul(b.CreateAdd(stripe,b.getInt32(1)),b.getInt32(kWordBytes*SCORPI_ATOMIC64_WORDS_PER_STRIPE));
    }
    const bool native = native_minmax && call && NativeUnsignedMinMax(call, kind);
    Value *initialDone = b.getFalse();
    Value *nativeOperand = operand;
    // A result-unused min/max to a compile-time uniform location can be
    // linearized as one wave operation. No lane observes an intermediate
    // result. Never combine addresses merely because their lock hashes match.
    Value *grouped = b.getFalse();
    Value *exclusive = nullptr;
    Value *groupOperand = operand;
    AtomicBinOpCode groupOperation = AtomicBinOpCode::Invalid;
    auto combine = [&](Value *left, Value *right, AtomicBinOpCode op) -> Value * {
      switch (op) {
      case AtomicBinOpCode::Add: return b.CreateAdd(left, right);
      case AtomicBinOpCode::And: return b.CreateAnd(left, right);
      case AtomicBinOpCode::Or: return b.CreateOr(left, right);
      case AtomicBinOpCode::Xor: return b.CreateXor(left, right);
      case AtomicBinOpCode::UMin: return b.CreateSelect(b.CreateICmpULT(left, right), left, right);
      case AtomicBinOpCode::UMax: return b.CreateSelect(b.CreateICmpUGT(left, right), left, right);
      case AtomicBinOpCode::IMin: return b.CreateSelect(b.CreateICmpSLT(left, right), left, right);
      case AtomicBinOpCode::IMax: return b.CreateSelect(b.CreateICmpSGT(left, right), left, right);
      case AtomicBinOpCode::Exchange: return right;
      default: throw std::runtime_error("atomic64 wave combination unavailable");
      }
    };
    auto shuffle64 = [&](Value *value, Value *lane) {
      auto read = [&](Value *word) { return callOp(b, OpCode::WaveReadLaneAt, b.getInt32Ty(), {word, lane}); };
      return b.CreateOr(b.CreateZExt(read(b.CreateTrunc(value,b.getInt32Ty())),b.getInt64Ty()),
          b.CreateShl(b.CreateZExt(read(b.CreateTrunc(b.CreateLShr(value,kWordBits),b.getInt32Ty())),b.getInt64Ty()),kWordBits));
    };
    Value *unwrapped = handle;
    if (auto *annotation = dyn_cast_or_null<CallInst>(unwrapped))
      if (Opcode(annotation) == OpCode::AnnotateHandle) unwrapped = annotation->getArgOperand(1);
    auto *creation = dyn_cast_or_null<CallInst>(unwrapped);
    bool uniform = creation && (Opcode(creation) == OpCode::CreateHandleFromBinding || Opcode(creation) == OpCode::CreateHandle);
    if (uniform) uniform = isa<ConstantInt>(creation->getArgOperand(Opcode(creation) == OpCode::CreateHandle ? 3 : 2));
    if (!shared) for (auto *coordinate : {x, y, z})
      uniform &= isa<ConstantInt>(coordinate) || isa<UndefValue>(coordinate);
    if (!native && call && !compare && uniform) {
      groupOperation = static_cast<AtomicBinOpCode>(cast<ConstantInt>(DxilInst_AtomicBinOp(call).get_atomicOp())->getZExtValue());
      auto *active = callOp(b, OpCode::WaveActiveOp, b.getInt32Ty(), {b.getInt32(1),
          b.getInt8(unsigned(WaveOpKind::Sum)), b.getInt8(unsigned(SignedOpKind::Unsigned))});
      // A full 32-lane wave permits a regular prefix scan. Partial waves use
      // the scalar fallback, and shuffle only their own active lane.
      grouped = b.CreateICmpEQ(active,b.getInt32(SCORPI_ATOMIC64_WAVE_LANES));
      auto *lane = callOp(b,OpCode::WaveGetLaneIndex,b.getVoidTy(),{});
      Value *inclusive = operand;
      for (unsigned step=1;step<SCORPI_ATOMIC64_WAVE_LANES;step*=2) {
        auto *has = b.CreateICmpUGE(lane,b.getInt32(step));
        auto *source = b.CreateSelect(b.CreateAnd(grouped,has),b.CreateSub(lane,b.getInt32(step)),lane);
        auto *preceding = shuffle64(inclusive,source);
        inclusive = b.CreateSelect(has,combine(preceding,inclusive,groupOperation),inclusive);
      }
      exclusive = shuffle64(inclusive,b.CreateSelect(b.CreateAnd(grouped,b.CreateICmpNE(lane,b.getInt32(0))),
          b.CreateSub(lane,b.getInt32(1)),lane));
      auto *total = shuffle64(inclusive,b.CreateSelect(grouped,b.getInt32(SCORPI_ATOMIC64_WAVE_LANES-1),lane));
      groupOperand = b.CreateSelect(grouped,total,operand);
      initialDone = b.CreateAnd(grouped,b.CreateNot(callOp(b,OpCode::WaveIsFirstLane,b.getVoidTy(),{})));
    }
    if (native) {
      if (uniform) {
        const auto operation = cast<ConstantInt>(DxilInst_AtomicBinOp(call).get_atomicOp())->getZExtValue();
        const bool maximum = operation == unsigned(AtomicBinOpCode::UMax);
        auto reduce = [&](Value *value) {
          return callOp(b, OpCode::WaveActiveOp, b.getInt32Ty(), {value,
              b.getInt8(unsigned(maximum ? WaveOpKind::Max : WaveOpKind::Min)),
              b.getInt8(unsigned(SignedOpKind::Unsigned))});
        };
        auto *high = b.CreateTrunc(b.CreateLShr(operand, kWordBits), b.getInt32Ty());
        auto *winningHigh = reduce(high);
        auto *low = b.CreateTrunc(operand, b.getInt32Ty());
        low = b.CreateSelect(b.CreateICmpEQ(high, winningHigh), low, b.getInt32(maximum ? 0 : UINT32_MAX));
        nativeOperand = b.CreateOr(b.CreateShl(b.CreateZExt(winningHigh, b.getInt64Ty()), kWordBits),
            b.CreateZExt(reduce(low), b.getInt64Ty()));
        initialDone = b.CreateNot(callOp(b, OpCode::WaveIsFirstLane, b.getVoidTy(), {}));
      }
    }
    auto *undef = UndefValue::get(b.getInt32Ty());
    auto *gateOffset = lockOffset;
    lockOffset = b.CreateAdd(gateOffset, b.getInt32(kWordBytes));
    auto gate = [&](IRBuilder<> &builder, AtomicBinOpCode op, Value *value) {
      return callOp(builder, OpCode::AtomicBinOp, builder.getInt32Ty(),
          {lockHandle, builder.getInt32(unsigned(op)), gateOffset, undef, undef, value});
    };
    auto fence = [&](IRBuilder<> &builder) {
      callOp(builder, OpCode::Barrier, builder.getVoidTy(),
          {builder.getInt32(unsigned(BarrierMode::UAVFenceGlobal) |
              (shared ? unsigned(BarrierMode::TGSMFence) : 0))});
    };
    b.CreateCondBr(valid, setup, after);
    b.SetInsertPoint(setup);
    PHINode *initialFinished = nullptr;
    if (native) {
      auto *registerReader = BasicBlock::Create(ctx, "atomic64.native.register", fn, ready);
      auto *performNative = BasicBlock::Create(ctx, "atomic64.native.perform", fn, ready);
      auto *declineNative = BasicBlock::Create(ctx, "atomic64.native.decline", fn, ready);
      auto *promote = BasicBlock::Create(ctx, "atomic64.promote", fn, ready);
      b.CreateCondBr(initialDone, ready, registerReader);
      b.SetInsertPoint(registerReader);
      auto *previous = gate(b, AtomicBinOpCode::Add, b.getInt32(1));
      auto *nativeAllowed = b.CreateICmpEQ(b.CreateAnd(previous, b.getInt32(SCORPI_ATOMIC64_EMULATED_BIT)), b.getInt32(0));
      b.CreateCondBr(nativeAllowed, performNative, declineNative);
      b.SetInsertPoint(performNative);
      auto *operation = cast<CallInst>(call->clone());
      operation->setArgOperand(DxilInst_AtomicBinOp::arg_newValue, nativeOperand);
      b.Insert(operation);
      // Publish the native update before withdrawing its reader reservation.
      // No mutex or retry loop is held on this path.
      fence(b);
      gate(b, AtomicBinOpCode::Add, b.getInt32(UINT32_MAX));
      b.CreateBr(ready);
      b.SetInsertPoint(declineNative);
      gate(b, AtomicBinOpCode::Add, b.getInt32(UINT32_MAX));
      b.CreateBr(promote);
      b.SetInsertPoint(promote);
      gate(b, AtomicBinOpCode::Or, b.getInt32(SCORPI_ATOMIC64_EMULATED_BIT));
      b.CreateBr(ready);
      b.SetInsertPoint(ready);
      initialFinished = b.CreatePHI(b.getInt1Ty(), 3, "atomic64.initial.finished");
      initialFinished->addIncoming(b.getTrue(), setup);
      initialFinished->addIncoming(b.getTrue(), performNative);
      initialFinished->addIncoming(b.getFalse(), promote);
    } else {
      gate(b, AtomicBinOpCode::Or, b.getInt32(SCORPI_ATOMIC64_EMULATED_BIT));
      b.CreateBr(ready);
      b.SetInsertPoint(ready);
      initialFinished = b.CreatePHI(b.getInt1Ty(), 1, "atomic64.initial.finished");
      initialFinished->addIncoming(initialDone, setup);
    }
    b.CreateBr(retry);
    b.SetInsertPoint(retry);
    auto *finished = b.CreatePHI(b.getInt1Ty(), 2, "atomic64.finished");
    auto *saved = b.CreatePHI(b.getInt64Ty(), 2, "atomic64.saved");
    finished->addIncoming(initialFinished, ready);
    saved->addIncoming(b.getInt64(0), ready);
    b.CreateCondBr(finished, latch, attempt);
    b.SetInsertPoint(attempt);
    // Promotion is permanent for this stripe. A late native reader can only
    // decrement its reservation and join the guarded path; it cannot write.
    auto *readers = gate(b, AtomicBinOpCode::Or, b.getInt32(0));
    b.CreateCondBr(b.CreateICmpEQ(b.CreateAnd(readers, b.getInt32(SCORPI_ATOMIC64_READER_MASK)), b.getInt32(0)), tryLock, latch);
    b.SetInsertPoint(tryLock);
    auto *held = callOp(b, OpCode::AtomicCompareExchange, b.getInt32Ty(),
        {lockHandle, lockOffset, undef, undef, b.getInt32(0), b.getInt32(1)});
    b.CreateCondBr(b.CreateICmpEQ(held, b.getInt32(0)), acquired, latch);
    b.SetInsertPoint(acquired);
    // Acquire/release are both inside the successful lane's branch. No lane
    // spins with the lock held while another lane waits at reconvergence.
    fence(b);
    Value *highX=raw ? b.CreateAdd(x,b.getInt32(kWordBytes)) : x;
    Value *highY=structured ? b.CreateAdd(y,b.getInt32(kWordBytes)) : y;
    Value *sharedLow=nullptr,*sharedHigh=nullptr;
    if (shared) {
      auto *pointer=sharedRmw ? sharedRmw->getPointerOperand() : sharedCmp->getPointerOperand();
      sharedLow=b.CreateBitCast(pointer,PointerType::get(b.getInt32Ty(),kTGSMAddrSpace));
      sharedHigh=b.CreateGEP(sharedLow,b.getInt32(1));
    }
    auto word=[&](AtomicBinOpCode op,Value *wx,Value *wy,Value *value) -> Value* {
      if (shared) return b.CreateAtomicRMW(op==AtomicBinOpCode::Or ? AtomicRMWInst::Or : AtomicRMWInst::Xchg,
          wx,value,AtomicOrdering::SequentiallyConsistent);
      return callOp(b,OpCode::AtomicBinOp,b.getInt32Ty(),{handle,b.getInt32(unsigned(op)),wx,wy,z,value});
    };
    Value *low=nullptr,*high=nullptr;
    if (native) {
      auto *operation = cast<CallInst>(call->clone());
      operation->setArgOperand(DxilInst_AtomicBinOp::arg_newValue, nativeOperand);
      b.Insert(operation);
      low = high = b.getInt32(0);
    } else
    if (typed) {
      auto *loaded=texture ? callOp(b,OpCode::TextureLoad,b.getInt32Ty(),{handle,b.getInt32(0),x,y,z,undef,undef,undef}) :
          callOp(b,OpCode::BufferLoad,b.getInt32Ty(),{handle,x,undef});
      low=b.CreateExtractValue(loaded,0);high=b.CreateExtractValue(loaded,1);
    } else {
      low=word(AtomicBinOpCode::Or,shared ? sharedLow : x,y,b.getInt32(0));
      high=word(AtomicBinOpCode::Or,shared ? sharedHigh : highX,highY,b.getInt32(0));
    }
    auto *old = b.CreateOr(b.CreateZExt(low, b.getInt64Ty()),
                           b.CreateShl(b.CreateZExt(high, b.getInt64Ty()), kWordBits));
    Value *value = nullptr;
    if (compare) {
      value = b.CreateSelect(b.CreateICmpEQ(old, comparison),
                             groupOperand, old);
    } else {
      AtomicBinOpCode operation=AtomicBinOpCode::Invalid;
      if (call) operation=static_cast<AtomicBinOpCode>(cast<ConstantInt>(DxilInst_AtomicBinOp(call).get_atomicOp())->getZExtValue());
      else switch (sharedRmw->getOperation()) {
      case AtomicRMWInst::Add: operation=AtomicBinOpCode::Add;break;
      case AtomicRMWInst::And: operation=AtomicBinOpCode::And;break;
      case AtomicRMWInst::Or: operation=AtomicBinOpCode::Or;break;
      case AtomicRMWInst::Xor: operation=AtomicBinOpCode::Xor;break;
      case AtomicRMWInst::Min: operation=AtomicBinOpCode::IMin;break;
      case AtomicRMWInst::Max: operation=AtomicBinOpCode::IMax;break;
      case AtomicRMWInst::UMin: operation=AtomicBinOpCode::UMin;break;
      case AtomicRMWInst::UMax: operation=AtomicBinOpCode::UMax;break;
      case AtomicRMWInst::Xchg: operation=AtomicBinOpCode::Exchange;break;
      default: throw std::runtime_error("unsupported groupshared atomic64 operation");
      }
      switch (operation) {
      case AtomicBinOpCode::Add:
        value = b.CreateAdd(old, groupOperand);
        break;
      case AtomicBinOpCode::And:
        value = b.CreateAnd(old, groupOperand);
        break;
      case AtomicBinOpCode::Or:
        value = b.CreateOr(old, groupOperand);
        break;
      case AtomicBinOpCode::Xor:
        value = b.CreateXor(old, groupOperand);
        break;
      case AtomicBinOpCode::IMin:
        value = b.CreateSelect(b.CreateICmpSLT(old, groupOperand), old, groupOperand);
        break;
      case AtomicBinOpCode::IMax:
        value = b.CreateSelect(b.CreateICmpSGT(old, groupOperand), old, groupOperand);
        break;
      case AtomicBinOpCode::UMin:
        value = b.CreateSelect(b.CreateICmpULT(old, groupOperand), old, groupOperand);
        break;
      case AtomicBinOpCode::UMax:
        value = b.CreateSelect(b.CreateICmpUGT(old, groupOperand), old, groupOperand);
        break;
      case AtomicBinOpCode::Exchange:
        value = groupOperand;
        break;
      default:
        throw std::runtime_error("unknown 64-bit atomic operation");
      }
    }
    auto *newLow=b.CreateTrunc(value,b.getInt32Ty());
    auto *newHigh=b.CreateTrunc(b.CreateLShr(value,kWordBits),b.getInt32Ty());
    if (!native && typed) {
      if (texture) callOp(b,OpCode::TextureStore,b.getInt32Ty(),{handle,x,y,z,newLow,newHigh,undef,undef,b.getInt8(kCompMask_X | kCompMask_Y)});
      else callOp(b,OpCode::BufferStore,b.getInt32Ty(),{handle,x,undef,newLow,newHigh,undef,undef,b.getInt8(kCompMask_X | kCompMask_Y)});
    } else if (!native) {
      word(AtomicBinOpCode::Exchange,shared ? sharedLow : x,y,newLow);
      word(AtomicBinOpCode::Exchange,shared ? sharedHigh : highX,highY,newHigh);
    }
    fence(b);
    callOp(b, OpCode::AtomicBinOp, b.getInt32Ty(),
           {lockHandle, b.getInt32(unsigned(AtomicBinOpCode::Exchange)),
            lockOffset, undef, undef, b.getInt32(0)});
    b.CreateBr(latch);
    b.SetInsertPoint(latch);
    auto *done = b.CreatePHI(b.getInt1Ty(), 4);
    done->addIncoming(finished, retry);
    done->addIncoming(b.getFalse(), attempt);
    done->addIncoming(b.getFalse(), tryLock);
    done->addIncoming(b.getTrue(), acquired);
    auto *result = b.CreatePHI(b.getInt64Ty(), 4);
    result->addIncoming(saved, retry);
    result->addIncoming(saved, attempt);
    result->addIncoming(saved, tryLock);
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
    if (exclusive) {
      b.SetInsertPoint(original);
      auto firstWord = [&](Value *word) { return callOp(b,OpCode::WaveReadLaneFirst,b.getInt32Ty(),{word}); };
      auto *base = b.CreateOr(b.CreateZExt(firstWord(b.CreateTrunc(bounded,b.getInt32Ty())),b.getInt64Ty()),
          b.CreateShl(b.CreateZExt(firstWord(b.CreateTrunc(b.CreateLShr(bounded,kWordBits),b.getInt32Ty())),b.getInt64Ty()),kWordBits));
      auto *lane = callOp(b,OpCode::WaveGetLaneIndex,b.getVoidTy(),{});
      auto *old = b.CreateSelect(b.CreateICmpEQ(lane,b.getInt32(0)),base,combine(base,exclusive,groupOperation));
      auto *result = b.CreateSelect(b.CreateAnd(grouped,valid),old,bounded);
      original->replaceAllUsesWith(result);
      original->eraseFromParent();
      continue;
    }
    Value *replacement=bounded;
    if (sharedCmp) {
      IRBuilder<> resultBuilder(original);
      replacement=resultBuilder.CreateInsertValue(UndefValue::get(original->getType()),bounded,0);
      replacement=resultBuilder.CreateInsertValue(replacement,resultBuilder.CreateICmpEQ(bounded,comparison),1);
    }
    original->replaceAllUsesWith(replacement);
    original->eraseFromParent();
  }
  return true;
}
