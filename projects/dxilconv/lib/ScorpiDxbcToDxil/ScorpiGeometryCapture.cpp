/*
    Copyright (c) Microsoft Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
    associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial
    portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
    NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
// Isoline coordinate arithmetic below is adapted from the D3D11 reference tessellator.
#include "ScorpiGeometryCapture.h"
#include "dxc/DXIL/DxilCBuffer.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilResource.h"
#include "dxc/DXIL/DxilResourceBinding.h"
#include "dxc/DXIL/DxilResourceProperties.h"
#include "dxc/DXIL/DxilShaderModel.h"
#include "dxc/DXIL/DxilSignature.h"
#include "dxc/DXIL/DxilSignatureElement.h"
#include "dxc/DXIL/DxilTypeSystem.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <unordered_map>
namespace scorpi {
namespace {
using namespace llvm;
using namespace hlsl;
using DXIL::OpCode;
struct CaptureBindings { Value *storage; Value *parameters; };
CaptureBindings AddCaptureBindings(DxilModule &dm, IRBuilder<> &b,
                                   uint32_t space) {
  auto *i32 = b.getInt32Ty();
  auto resource = std::make_unique<DxilResource>();
  resource->SetID(unsigned(dm.GetUAVs().size()));
  resource->SetRW(true); resource->SetKind(DXIL::ResourceKind::RawBuffer);
  resource->SetSpaceID(space); resource->SetLowerBound(0); resource->SetRangeSize(1);
  resource->SetGlobalName("ScorpiCapturedGeometry");
  resource->SetGlobalSymbol(UndefValue::get(StructType::create(dm.GetCtx(), {i32}, "ScorpiCapturedGeometry")->getPointerTo()));
  unsigned uavID=UINT32_MAX;
  for(const auto &r:dm.GetUAVs()) if(r->GetSpaceID()==space && r->GetGlobalName()=="ScorpiCapturedGeometry")uavID=r->GetID();
  if(uavID==UINT32_MAX)uavID=dm.AddUAV(std::move(resource));
  auto cb = std::make_unique<DxilCBuffer>();
  cb->SetID(unsigned(dm.GetCBuffers().size())); cb->SetSize(32);
  cb->SetSpaceID(space); cb->SetLowerBound(0); cb->SetRangeSize(1);
  cb->SetGlobalName("ScorpiRasterReplay");
  cb->SetGlobalSymbol(UndefValue::get(StructType::create(dm.GetCtx(), {i32, i32, i32, i32, i32, i32, i32, i32}, "ScorpiRasterReplay")->getPointerTo()));
  unsigned cbID=UINT32_MAX;
  for(const auto &r:dm.GetCBuffers()) if(r->GetSpaceID()==space && r->GetGlobalName()=="ScorpiRasterReplay")cbID=r->GetID();
  if(cbID==UINT32_MAX)cbID=dm.AddCBuffer(std::move(cb));
  auto call = [&](OpCode op, Type *type, std::vector<Value *> args) {
    args.insert(args.begin(), b.getInt32(unsigned(op)));
    return b.CreateCall(dm.GetOP()->GetOpFunc(op, type), args);
  };
  auto handle = [&](DxilResourceBase &r) {
    if (dm.GetShaderModel()->GetMinor() >= 6) {
      auto binding = resource_helper::loadBindingFromResourceBase(&r);
      auto props = resource_helper::loadPropsFromResourceBase(&r);
      auto *bound = call(OpCode::CreateHandleFromBinding, b.getVoidTy(),
          {resource_helper::getAsConstant(binding, dm.GetOP()->GetResourceBindingType(), *dm.GetShaderModel()), b.getInt32(0), b.getFalse()});
      return call(OpCode::AnnotateHandle, b.getVoidTy(),
          {bound, resource_helper::getAsConstant(props, dm.GetOP()->GetResourcePropertiesType(), *dm.GetShaderModel())});
    }
    return call(OpCode::CreateHandle, b.getVoidTy(),
        {b.getInt8(unsigned(r.GetClass())), b.getInt32(r.GetID()), b.getInt32(0), b.getFalse()});
  };
  auto *uav = handle(dm.GetUAV(uavID));
  auto *parameters = call(OpCode::CBufferLoadLegacy, i32, {handle(dm.GetCBuffer(cbID)), b.getInt32(1)});
  return {uav, parameters};
}
}

// Private DS/GS linkage: a uint4 ordering key followed by uint4 payload rows.
// Keeping the payload as uint bits avoids float conversions and ABI padding
// differences for scalar/float3 application output elements.
static void AddTessellationCaptureSignature(hlsl::DxilSignature &signature, uint32_t words) {
  using namespace hlsl;
  signature.ClearElements();
  auto add = [&](const char *name, uint32_t row, uint32_t rows) {
    auto element=signature.CreateElement();
    std::vector<unsigned> indices;
    for(unsigned i=0;i<rows;++i) indices.push_back(i);
    element->Initialize(name,CompType::getU32(),DXIL::InterpolationMode::Constant,
        rows,4,row,0,DxilSignatureElement::kUndefinedID,indices);
    element->SetKind(DXIL::SemanticKind::Arbitrary);element->SetUsageMask(15);
    signature.AppendElement(std::move(element));
  };
  add("SCORPI_TESS_KEY",0,1);
  add("SCORPI_TESS_DATA",1,(words+3)/4);
}

// The private isoline path uses integer topology and restores D3D fractional
// coordinates before application DS execution. Placement follows Microsoft's
// D3D11 fixed-function tessellator reference (ComputeTessFactorContext and
// PlacePointIn1D), using the same 16-bit fixed-point interpolation.
// Per-patch control: half point count, split, fraction, reciprocal floor/ceil,
// original detail/density, and parity. It lives after the capture allocation.
static void PrepareIsolineTopology(hlsl::DxilModule &dm, uint32_t space) {
  using namespace llvm;using namespace hlsl;
  if(dm.GetTessellatorDomain()!=DXIL::TessellatorDomain::IsoLine)return;
  const auto partition=dm.GetTessellatorPartitioning();
  const bool oddMode=partition==DXIL::TessellatorPartitioning::FractionalOdd;
  const bool fractional=oddMode||partition==DXIL::TessellatorPartitioning::FractionalEven;
  for(auto &function:*dm.GetModule()) {
    if(function.isDeclaration())continue;
    SmallVector<CallInst *,4> stores;
    for(auto &block:function)for(auto &instruction:block)if(auto *c=dyn_cast<CallInst>(&instruction)) {
      auto *fn=c->getCalledFunction();
      if(!fn||!fn->getName().startswith("dx.op.storePatchConstant."))continue;
      auto *id=dyn_cast<ConstantInt>(c->getArgOperand(1));
      auto *row=dyn_cast<ConstantInt>(c->getArgOperand(2));
      auto *col=dyn_cast<ConstantInt>(c->getArgOperand(3));
      if(id&&col&&(!row||row->getZExtValue()<2)&&col->isZero()&&
          dm.GetPatchConstOrPrimSignature().GetElement(unsigned(id->getZExtValue())).GetKind()==DXIL::SemanticKind::TessFactor)stores.push_back(c);
    }
    if(stores.empty())continue;
    // FXC fork phases can write both factors through one loop-indexed store.
    // Specialize only that store, preserving the original phase's invocation
    // count and side effects. Each branch writes its own private control data.
    SmallVector<CallInst *,4> constantStores;
    for(auto *store:stores) {
      auto *row=store->getArgOperand(2);
      if(isa<ConstantInt>(row)){constantStores.push_back(store);continue;}
      IRBuilder<> branch(store);
      TerminatorInst *densityEnd=nullptr,*detailEnd=nullptr;
      SplitBlockAndInsertIfThenElse(branch.CreateICmpEQ(row,branch.getInt32(0)),
                                   store,&densityEnd,&detailEnd);
      for(unsigned i=0;i<2;++i) {
        auto *copy=cast<CallInst>(store->clone());
        copy->setArgOperand(2,branch.getInt32(i));
        copy->insertBefore(i?detailEnd:densityEnd);
        constantStores.push_back(copy);
      }
      store->eraseFromParent();
    }
    IRBuilder<> b(&*function.getEntryBlock().getFirstInsertionPt());auto binding=AddCaptureBindings(dm,b,space);
    auto *address=b.CreateExtractValue(binding.parameters,3);
    auto fp=[&](double x){return ConstantFP::get(b.getFloatTy(),x);};
    auto op=[&](OpCode code,std::vector<Value *> args) {
      args.insert(args.begin(),b.getInt32(unsigned(code)));
      return b.CreateCall(dm.GetOP()->GetOpFunc(code,b.getFloatTy()),args);
    };
    auto write=[&](unsigned offset,std::vector<Value *> words) {
      const uint8_t mask=uint8_t((1u<<words.size())-1);
      while(words.size()<4)words.push_back(UndefValue::get(b.getInt32Ty()));
      b.CreateCall(dm.GetOP()->GetOpFunc(OpCode::BufferStore,b.getInt32Ty()),
        {b.getInt32(unsigned(OpCode::BufferStore)),binding.storage,b.CreateAdd(address,b.getInt32(offset)),UndefValue::get(b.getInt32Ty()),words[0],words[1],words[2],words[3],b.getInt8(mask)});
    };
    for(auto *store:constantStores) {
      b.SetInsertPoint(store);auto *original=store->getArgOperand(4);
      const unsigned row=unsigned(cast<ConstantInt>(store->getArgOperand(2))->getZExtValue());
      // The converter's isoline topology consumes detail before density.
      store->setArgOperand(2,b.getInt32(1-row));
      if(!row) {write(24,{b.CreateBitCast(original,b.getInt32Ty())});continue;}
      Value *factor=op(OpCode::FMin,{op(OpCode::FMax,{original,fp(fractional&&!oddMode?2:1)}),fp(oddMode?63:64)});
      if(!fractional)factor=op(OpCode::Round_pi,{factor});
      auto *fixed=b.CreateFPToUI(op(OpCode::Round_ne,{b.CreateFMul(factor,fp(65536))}),b.getInt32Ty());
      Value *odd=fractional?static_cast<Value *>(b.getInt32(oddMode)):b.CreateAnd(b.CreateLShr(fixed,16),b.getInt32(1));
      auto *half=b.CreateLShr(b.CreateAdd(fixed,b.getInt32(1)),1);
      half=b.CreateAdd(half,b.CreateSelect(b.CreateOr(b.CreateICmpNE(odd,b.getInt32(0)),b.CreateICmpEQ(half,b.getInt32(32768))),b.getInt32(32768),b.getInt32(0)));
      auto *floor=b.CreateLShr(half,16);auto *ceil=b.CreateLShr(b.CreateAdd(half,b.getInt32(65535)),16);
      auto *fraction=b.CreateAnd(half,b.getInt32(65535));
      auto *base=b.CreateSub(floor,odd);Value *msb=b.getInt32(0);
      for(unsigned bit=1;bit<=32;bit*=2)msb=b.CreateSelect(b.CreateICmpUGE(base,b.getInt32(bit)),b.getInt32(bit),msb);
      auto *split=b.CreateAdd(b.CreateShl(b.CreateXor(base,msb),1),b.getInt32(1));
      split=b.CreateSelect(b.CreateAnd(b.CreateICmpNE(odd,b.getInt32(0)),b.CreateICmpEQ(floor,b.getInt32(1))),b.getInt32(0),split);
      split=b.CreateSelect(b.CreateICmpEQ(floor,ceil),b.CreateAdd(ceil,b.getInt32(1)),split);
      auto reciprocal=[&](Value *n){auto *safe=b.CreateSelect(b.CreateICmpEQ(n,b.getInt32(0)),b.getInt32(1),n);
        return b.CreateSelect(b.CreateICmpEQ(n,b.getInt32(0)),b.getInt32(UINT32_MAX),b.CreateUDiv(b.CreateAdd(b.getInt32(65536),b.CreateLShr(safe,1)),safe));};
      auto *segments=b.CreateSub(b.CreateShl(ceil,1),odd);
      write(0,{ceil,split,fraction,reciprocal(b.CreateSub(b.CreateShl(floor,1),odd))});
      write(16,{reciprocal(segments),b.CreateBitCast(original,b.getInt32Ty())});write(28,{odd});
      store->setArgOperand(4,b.CreateSelect(b.CreateFCmpOGT(original,fp(0)),b.CreateUIToFP(segments,b.getFloatTy()),original));
    }
  }
  dm.SetTessellatorPartitioning(DXIL::TessellatorPartitioning::Integer);
}

static void RestoreIsolineCoordinates(hlsl::DxilModule &dm,uint32_t space) {
  using namespace llvm;using namespace hlsl;
  if(dm.GetTessellatorDomain()!=DXIL::TessellatorDomain::IsoLine)return;
  auto *entry=dm.GetEntryFunction();SmallVector<CallInst *,4> coordinates,patchLoads;
  for(auto &block:*entry)for(auto &instruction:block)if(auto *c=dyn_cast<CallInst>(&instruction)) {
    auto *fn=c->getCalledFunction();auto *component=c->getNumArgOperands()>1?dyn_cast<ConstantInt>(c->getArgOperand(1)):nullptr;
    if(fn&&fn->getName().startswith("dx.op.domainLocation.")&&component&&component->getZExtValue()<2)coordinates.push_back(c);
    if(fn&&fn->getName().startswith("dx.op.loadPatchConstant."))patchLoads.push_back(c);
  }
  IRBuilder<> b(&*entry->getEntryBlock().getFirstInsertionPt());auto binding=AddCaptureBindings(dm,b,space);
  auto *address=b.CreateExtractValue(binding.parameters,3);
  auto read=[&](unsigned offset){return b.CreateCall(dm.GetOP()->GetOpFunc(OpCode::BufferLoad,b.getInt32Ty()),
      {b.getInt32(unsigned(OpCode::BufferLoad)),binding.storage,b.CreateAdd(address,b.getInt32(offset)),UndefValue::get(b.getInt32Ty())});};
  auto *control=read(0),*extra=read(16);
  auto *half=b.CreateExtractValue(control,0),*split=b.CreateExtractValue(control,1),*fraction=b.CreateExtractValue(control,2);
  auto *floorInv=b.CreateExtractValue(control,3),*ceilInv=b.CreateExtractValue(extra,0),*odd=b.CreateExtractValue(extra,3);
  auto *segments=b.CreateSub(b.CreateShl(half,1),odd);
  for(auto *original:coordinates) {
    SmallVector<Use *,8> uses;for(auto &use:original->uses())uses.push_back(&use);
    b.SetInsertPoint(original->getNextNode());
    if(cast<ConstantInt>(original->getArgOperand(1))->getZExtValue()==1) {
      auto fp=[&](double x){return ConstantFP::get(b.getFloatTy(),x);};
      auto fop=[&](OpCode code,std::vector<Value *> values){values.insert(values.begin(),b.getInt32(unsigned(code)));return b.CreateCall(dm.GetOP()->GetOpFunc(code,b.getFloatTy()),values);};
      auto *density=b.CreateFPToUI(fop(OpCode::Round_pi,{fop(OpCode::FMin,{fop(OpCode::FMax,{b.CreateBitCast(b.CreateExtractValue(extra,2),b.getFloatTy()),fp(1)}),fp(64)})}),b.getInt32Ty());
      auto *line=b.CreateFPToUI(fop(OpCode::Round_ne,{b.CreateFMul(original,b.CreateUIToFP(density,b.getFloatTy()))}),b.getInt32Ty());
      auto *flipped=b.CreateICmpUGT(b.CreateShl(line,1),density);
      auto *nearIndex=b.CreateSelect(flipped,b.CreateSub(density,line),line);
      auto *reciprocal=b.CreateUDiv(b.CreateAdd(b.getInt32(65536),b.CreateLShr(density,1)),density);
      auto *location=b.CreateMul(nearIndex,reciprocal);
      location=b.CreateSelect(flipped,b.CreateSub(b.getInt32(65536),location),location);
      location=b.CreateSelect(b.CreateICmpEQ(b.CreateShl(line,1),density),b.getInt32(32768),location);
      auto *mapped=b.CreateFMul(b.CreateUIToFP(location,b.getFloatTy()),fp(1.0/65536));
      for(auto *use:uses)use->set(mapped);
      continue;
    }
    auto *scaled=b.CreateFMul(original,b.CreateUIToFP(segments,b.getFloatTy()));
    auto *index=b.CreateFPToUI(b.CreateCall(dm.GetOP()->GetOpFunc(OpCode::Round_ne,b.getFloatTy()),{b.getInt32(unsigned(OpCode::Round_ne)),scaled}),b.getInt32Ty());
    auto *flip=b.CreateICmpUGE(index,half);auto *point=b.CreateSelect(flip,b.CreateSub(segments,index),index);
    auto *floorIndex=b.CreateSub(point,b.CreateZExt(b.CreateICmpUGT(point,split),b.getInt32Ty()));
    auto *low=b.CreateMul(floorIndex,floorInv),*high=b.CreateMul(point,ceilInv);
    auto *fixed=b.CreateLShr(b.CreateAdd(b.CreateAdd(b.CreateMul(low,b.CreateSub(b.getInt32(65536),fraction)),b.CreateMul(high,fraction)),b.getInt32(32768)),16);
    fixed=b.CreateSelect(flip,b.CreateSub(b.getInt32(65536),fixed),fixed);
    fixed=b.CreateSelect(b.CreateICmpEQ(point,half),b.getInt32(32768),fixed);
    auto *mapped=b.CreateFMul(b.CreateUIToFP(fixed,b.getFloatTy()),ConstantFP::get(b.getFloatTy(),1.0/65536));
    for(auto *use:uses)use->set(mapped);
  }
  for(auto *c:patchLoads) {
    auto *id=dyn_cast<ConstantInt>(c->getArgOperand(1));auto *row=dyn_cast<ConstantInt>(c->getArgOperand(2));
    if(id&&row&&row->getZExtValue()<2&&dm.GetPatchConstOrPrimSignature().GetElement(unsigned(id->getZExtValue())).GetKind()==DXIL::SemanticKind::TessFactor) {
      b.SetInsertPoint(c);auto *value=b.CreateBitCast(b.CreateExtractValue(extra,row->isZero()?2:1),b.getFloatTy());
      c->replaceAllUsesWith(value);c->eraseFromParent();
    }
  }
}

bool PrepareTessellationPixelPatchId(hlsl::DxilModule &dm,uint32_t space,std::string &error) {
  using namespace llvm;using namespace hlsl;
  if(!dm.GetShaderModel()->IsPS()){error="patch ID linking requires PS";return false;}
  for(auto &function:*dm.GetModule()) {
    if(function.isDeclaration())continue;
    SmallVector<CallInst *,8> calls;
    for(auto &block:function)for(auto &instruction:block)if(auto *call=dyn_cast<CallInst>(&instruction)) {
      auto *fn=call->getCalledFunction();
      if(!fn || !fn->getName().startswith("dx.op.loadInput."))continue;
      auto *id=dyn_cast<ConstantInt>(call->getArgOperand(1));
      if(id && id->getZExtValue()<dm.GetInputSignature().GetElements().size() &&
          dm.GetInputSignature().GetElement(id->getZExtValue()).GetKind()==DXIL::SemanticKind::PrimitiveID)calls.push_back(call);
    }
    if(calls.empty())continue;
    IRBuilder<> b(&*function.getEntryBlock().getFirstInsertionPt());
    auto bindings=AddCaptureBindings(dm,b,space);
    auto *patch=b.CreateExtractValue(bindings.parameters,2);
    for(auto *call:calls){call->replaceAllUsesWith(patch);call->eraseFromParent();}
  }
  dm.CollectShaderFlagsForModule();return true;
}

bool PrepareTessellationPatchBias(hlsl::DxilModule &dm,uint32_t space,std::string &error) {
  using namespace llvm;using namespace hlsl;
  if(!dm.GetShaderModel()->IsHS() && !dm.GetShaderModel()->IsDS()) {error="patch bias requires HS or DS";return false;}
  for(auto &function:*dm.GetModule()) {
    if(function.isDeclaration())continue;
    SmallVector<CallInst *,8> calls;
    for(auto &block:function)for(auto &instruction:block)if(auto *call=dyn_cast<CallInst>(&instruction)) {
      if(call->getCalledFunction() && call->getCalledFunction()->getName().startswith("dx.op.primitiveID"))calls.push_back(call);
    }
    if(calls.empty())continue;
    IRBuilder<> b(&*function.getEntryBlock().getFirstInsertionPt());
    auto bindings=AddCaptureBindings(dm,b,space);
    auto *bias=b.CreateExtractValue(bindings.parameters,2);
    for(auto *call:calls) {
      SmallVector<Use *,8> uses;for(auto &use:call->uses())uses.push_back(&use);
      b.SetInsertPoint(call->getNextNode());auto *adjusted=b.CreateAdd(call,bias);
      for(auto *use:uses)use->set(adjusted);
    }
  }
  if(dm.GetShaderModel()->IsHS())PrepareIsolineTopology(dm,space);
  dm.CollectShaderFlagsForModule();return true;
}

bool PrepareTessellationCaptureDomain(hlsl::DxilModule &dm, uint32_t &vertexStride, std::string &error, uint32_t space) {
  using namespace llvm;
  using namespace hlsl;
  if(!dm.GetShaderModel()->IsDS()) { error="tessellation capture requires a domain shader"; return false; }
  std::unordered_map<unsigned,std::pair<unsigned,unsigned>> elements;
  unsigned words=0;
  for(const auto &e:dm.GetOutputSignature().GetElements()) {
    if(e->GetCompType().Is64Bit() || e->GetCompType().Is16Bit()) {
      error="tessellation capture requires 32-bit output components";return false;
    }
    elements[e->GetID()]={words,e->GetCols()};words+=e->GetRows()*e->GetCols();
  }
  if(!words || words>128) { error="invalid domain output size";return false; }
  vertexStride=words*4;
  std::vector<CallInst *> outputs;
  // Output intrinsics can survive in helper functions after DXIL linking.
  // Rewrite the whole module before replacing the signature so no helper
  // retains an application signature ID in the private capture interface.
  for(auto &function:*dm.GetModule()) for(auto &block:function) for(auto &instruction:block) {
    auto *call=dyn_cast<CallInst>(&instruction);
    if(call && call->getCalledFunction() && call->getCalledFunction()->getName().startswith("dx.op.storeOutput.")) outputs.push_back(call);
  }
  AddTessellationCaptureSignature(dm.GetOutputSignature(),words);
  dm.GetSerializedViewIdState().clear();
  for(auto *original:outputs) {
    auto *id=dyn_cast<ConstantInt>(original->getArgOperand(1));
    if(!id || !elements.count(unsigned(id->getZExtValue()))) {error="invalid domain output ID";return false;}
    auto e=elements[unsigned(id->getZExtValue())];IRBuilder<> b(original);
    auto *word=b.CreateAdd(b.getInt32(e.first),b.CreateAdd(b.CreateMul(original->getArgOperand(2),b.getInt32(e.second)),b.CreateZExt(original->getArgOperand(3),b.getInt32Ty())));
    auto *value=original->getArgOperand(4);
    if(value->getType()!=b.getInt32Ty()) value=b.CreateBitCast(value,b.getInt32Ty());
    b.CreateCall(dm.GetOP()->GetOpFunc(DXIL::OpCode::StoreOutput,b.getInt32Ty()),
        {b.getInt32(unsigned(DXIL::OpCode::StoreOutput)),b.getInt32(1),b.CreateUDiv(word,b.getInt32(4)),
         b.CreateTrunc(b.CreateURem(word,b.getInt32(4)),b.getInt8Ty()),value});
    original->eraseFromParent();
  }
  for(auto &block:*dm.GetEntryFunction()) if(auto *ret=dyn_cast<ReturnInst>(block.getTerminator())) {
    IRBuilder<> b(ret);
    for(unsigned col=0;col<4;++col) b.CreateCall(dm.GetOP()->GetOpFunc(DXIL::OpCode::StoreOutput,b.getInt32Ty()),
        {b.getInt32(unsigned(DXIL::OpCode::StoreOutput)),b.getInt32(0),b.getInt32(0),b.getInt8(col),b.getInt32(0)});
  }
  RestoreIsolineCoordinates(dm,space);
  dm.CollectShaderFlagsForModule();
  std::string verification;llvm::raw_string_ostream diagnostics(verification);
  if(verifyModule(*dm.GetModule(), &diagnostics)) {diagnostics.flush();error="invalid tessellation capture domain IR: "+verification;return false;}
  return true;
}

bool LowerGeometryPrimitiveId(hlsl::DxilModule &dm, std::string &error) {
  using namespace hlsl;
  if (!dm.GetShaderModel()->IsGS() && !dm.GetShaderModel()->IsPS()) {
    error = "geometry PrimitiveID linking requires GS or PS"; return false;
  }
  auto &signature = dm.GetShaderModel()->IsGS() ? dm.GetOutputSignature() : dm.GetInputSignature();
  struct Element {
    std::string name;
    hlsl::CompType type;
    hlsl::InterpolationMode interpolation;
    unsigned rows, cols, id, usage, stream, dynamic_mask;
    int row, col;
    std::vector<unsigned> indices;
    hlsl::DXIL::SemanticKind kind;
  };
  std::vector<Element> elements;
  bool changed = false;
  for (auto &element : signature.GetElements()) {
    const bool primitive = element->GetKind() == DXIL::SemanticKind::PrimitiveID;
    if (primitive && !element->IsAllocated()) { error = "unallocated geometry PrimitiveID signature"; return false; }
    changed |= primitive;
    elements.push_back({primitive ? "SCORPI_GS_PRIMITIVE_ID" : element->GetName(),
        element->GetCompType(), primitive ? DXIL::InterpolationMode::Constant :
        *element->GetInterpolationMode(), element->GetRows(), element->GetCols(),
        element->GetID(), element->GetUsageMask(), element->GetOutputStream(),
        element->GetDynIdxCompMask(), element->GetStartRow(), element->GetStartCol(),
        element->GetSemanticIndexVec(), primitive ? DXIL::SemanticKind::Arbitrary : element->GetKind()});
  }
  if (!changed) return true;
  signature.ClearElements();
  dm.GetSerializedViewIdState().clear();
  for (const auto &source : elements) {
    auto element = signature.CreateElement();
    element->Initialize(source.name, source.type, source.interpolation,
        source.rows, source.cols, source.row, source.col, source.id, source.indices);
    element->SetKind(source.kind);
    element->SetUsageMask(source.usage);
    element->SetOutputStream(source.stream);
    element->SetDynIdxCompMask(source.dynamic_mask);
    signature.AppendElement(std::move(element));
  }
  dm.CollectShaderFlagsForModule();
  return true;
}

bool PrepareRasterReplayVertexShader(hlsl::DxilModule &dm, uint32_t space,
    std::string &error) {
  using namespace llvm;
  using namespace hlsl;
  if (!dm.GetShaderModel()->IsVS()) { error = "draw ID adjustment requires a vertex shader"; return false; }
  std::vector<std::pair<CallInst *, unsigned>> loads;
  for (auto &block : *dm.GetEntryFunction()) for (auto &instruction : block) {
    auto *call = dyn_cast<CallInst>(&instruction);
    if (!call || !call->getCalledFunction() || !call->getCalledFunction()->getName().startswith("dx.op.loadInput.")) continue;
    auto *id = dyn_cast<ConstantInt>(call->getArgOperand(1));
    if (!id || id->getZExtValue() >= dm.GetInputSignature().GetElements().size()) continue;
    const auto kind = dm.GetInputSignature().GetElements()[id->getZExtValue()]->GetKind();
    if (kind == DXIL::SemanticKind::VertexID || kind == DXIL::SemanticKind::InstanceID)
      loads.emplace_back(call, kind == DXIL::SemanticKind::VertexID ? 0 : 1);
  }
  if (loads.empty()) return true;
  for (const auto &cb : dm.GetCBuffers()) if (cb->GetSpaceID() == space && cb->GetLowerBound() == 0) {
    error = "private replay constant buffer overlaps shader binding"; return false;
  }
  IRBuilder<> b(&*dm.GetEntryFunction()->getEntryBlock().getFirstInsertionPt());
  auto *i32 = b.getInt32Ty();
  auto cb = std::make_unique<DxilCBuffer>();
  cb->SetID(unsigned(dm.GetCBuffers().size())); cb->SetSize(sizeof(ScorpiRasterReplayParameters));
  cb->SetSpaceID(space); cb->SetLowerBound(0); cb->SetRangeSize(1);
  cb->SetGlobalName("ScorpiReplayDraw");
  cb->SetGlobalSymbol(UndefValue::get(StructType::create(dm.GetCtx(),
      {ArrayType::get(i32, sizeof(ScorpiRasterReplayParameters) / 4)}, "ScorpiReplayDraw")->getPointerTo()));
  const auto id = dm.AddCBuffer(std::move(cb));
  auto op = [&](OpCode opcode, Type *type, std::vector<Value *> args) {
    args.insert(args.begin(), b.getInt32(unsigned(opcode)));
    return b.CreateCall(dm.GetOP()->GetOpFunc(opcode, type), args);
  };
  Value *handle;
  auto &resource = dm.GetCBuffer(id);
  if (dm.GetShaderModel()->GetMinor() >= 6) {
    auto binding = resource_helper::loadBindingFromResourceBase(&resource);
    auto props = resource_helper::loadPropsFromResourceBase(&resource);
    auto *bound = op(OpCode::CreateHandleFromBinding, b.getVoidTy(),
        {resource_helper::getAsConstant(binding, dm.GetOP()->GetResourceBindingType(), *dm.GetShaderModel()), b.getInt32(0), b.getFalse()});
    handle = op(OpCode::AnnotateHandle, b.getVoidTy(),
        {bound, resource_helper::getAsConstant(props, dm.GetOP()->GetResourcePropertiesType(), *dm.GetShaderModel())});
  } else handle = op(OpCode::CreateHandle, b.getVoidTy(),
      {b.getInt8(unsigned(resource.GetClass())), b.getInt32(id), b.getInt32(0), b.getFalse()});
  auto *params = op(OpCode::CBufferLoadLegacy, i32, {handle, b.getInt32(2)});
  Value *biases[] = {b.CreateExtractValue(params, 0), b.CreateExtractValue(params, 1)};
  for (const auto &load : loads) {
    // Save the uses before constructing the add, so the new add keeps its
    // reference to the original system value rather than becoming recursive.
    SmallVector<Use *, 8> uses;
    for (auto &use : load.first->uses()) uses.push_back(&use);
    b.SetInsertPoint(load.first->getNextNode());
    auto *adjusted = b.CreateAdd(load.first, biases[load.second]);
    for (auto *use : uses) use->set(adjusted);
  }
  dm.CollectShaderFlagsForModule();
  return true;
}

bool GetGeometryCaptureLayout(const hlsl::DxilModule &source,
    ScorpiGeometryCaptureLayout &layout, std::string &error) {
  layout = {};
  if (!source.GetShaderModel()->IsGS() || !source.GetActiveStreamMask() ||
      (source.GetActiveStreamMask() & ~15u)) {
    error = "generated geometry capture requires one to four GS streams";
    return false;
  }
  layout.stream_mask = source.GetActiveStreamMask();
  switch (source.GetStreamPrimitiveTopology()) {
  case hlsl::DXIL::PrimitiveTopology::PointList: layout.vertices_per_primitive = 1; break;
  case hlsl::DXIL::PrimitiveTopology::LineStrip: layout.vertices_per_primitive = 2; break;
  case hlsl::DXIL::PrimitiveTopology::TriangleStrip: layout.vertices_per_primitive = 3; break;
  default: error = "invalid GS output topology"; return false;
  }
  for (const auto &e : source.GetOutputSignature().GetElements()) {
    if (e->GetCompType().Is64Bit() || e->GetCompType().Is16Bit()) {
      error = "geometry capture requires 32-bit signature storage";
      return false;
    }
    layout.vertex_stride += e->GetRows() * e->GetCols() * 4;
  }
  layout.max_vertices = source.GetMaxVertexCount();
  layout.instance_count = std::max(1u, source.GetGSInstanceCount());
  if (!layout.vertex_stride || !layout.max_vertices || layout.max_vertices > 1024 || layout.instance_count > 32) {
    error = "invalid geometry capture storage bounds";
    return false;
  }
  return true;
}

bool PrepareGeometryShaderCapture(hlsl::DxilModule &dm, uint32_t space,
    const ScorpiGeometryCaptureLayout &layout, std::string &error, bool batched, bool rasterize) {
  using namespace llvm;
  using namespace hlsl;
  auto *entry = dm.GetEntryFunction();
  std::vector<CallInst *> calls;
  for (auto &block : *entry) for (auto &inst : block) {
    auto *call = dyn_cast<CallInst>(&inst);
    if (!call) continue;
    auto *function = call->getCalledFunction();
    if (!function || !function->isDeclaration()) {
      error = "geometry capture requires inlined shader functions";
      return false;
    }
    if (function->getName().startswith("dx.op.")) calls.push_back(call);
  }
  IRBuilder<> b(&*entry->getEntryBlock().getFirstInsertionPt());
  auto binding = AddCaptureBindings(dm, b, space);
  auto *i32 = b.getInt32Ty();
  auto call = [&](OpCode op, Type *type, std::vector<Value *> args) {
    args.insert(args.begin(), b.getInt32(unsigned(op)));
    return b.CreateCall(dm.GetOP()->GetOpFunc(op, type), args);
  };
  auto *primitiveID = b.CreateExtractValue(binding.parameters, 2);
  Value *instance = b.getInt32(0);
  if (layout.instance_count > 1) instance = call(OpCode::GSInstanceID, i32, {});
  const uint32_t recordStride = layout.vertex_stride + 8;
  Value *slot=instance;
  if(batched) slot=b.CreateAdd(b.CreateMul(call(OpCode::PrimitiveID,i32,{}),b.getInt32(layout.instance_count)),instance);
  const uint32_t streamStride = 16 + layout.max_vertices * recordStride;
  const uint32_t streamCount = layout.stream_mask == 1 ? 1 : 4;
  auto *base = b.CreateMul(slot, b.getInt32(streamCount * streamStride));
  auto *counts = b.CreateAlloca(ArrayType::get(i32, 4), nullptr, "capture.counts");
  for (unsigned stream = 0; stream < 4; ++stream)
    b.CreateStore(b.getInt32(0), b.CreateGEP(counts, {b.getInt32(0), b.getInt32(stream)}));
  auto *strips = b.CreateAlloca(ArrayType::get(i32, 4), nullptr, "capture.strips");
  for (unsigned stream = 0; stream < 4; ++stream)
    b.CreateStore(b.getInt32(0), b.CreateGEP(strips, {b.getInt32(0), b.getInt32(stream)}));
  auto *values = b.CreateAlloca(ArrayType::get(i32, layout.vertex_stride / 4), nullptr, "capture.values");
  auto valuePointer = [&](Value *index) { return b.CreateGEP(values, {b.getInt32(0), index}); };
  std::unordered_map<unsigned, unsigned> offsets;
  unsigned offset = 0, primitiveComponent = UINT32_MAX;
  for (const auto &e : dm.GetOutputSignature().GetElements()) {
    offsets[e->GetID()] = offset;
    if (e->GetKind() == DXIL::SemanticKind::PrimitiveID) primitiveComponent = offset;
    offset += e->GetRows() * e->GetCols();
  }
  for (unsigned i = 0; i < offset; ++i) b.CreateStore(b.getInt32(0), valuePointer(b.getInt32(i)));
  auto store = [&](Value *address, Value *bits) {
    call(OpCode::BufferStore, i32, {binding.storage, address, UndefValue::get(i32), bits,
        UndefValue::get(i32), UndefValue::get(i32), UndefValue::get(i32), b.getInt8(1)});
  };
  for (auto *original : calls) {
    auto *op = dyn_cast<ConstantInt>(original->getArgOperand(0));
    if (!op) { error = "nonconstant DXIL opcode"; return false; }
    auto opcode = static_cast<OpCode>(op->getZExtValue());
    b.SetInsertPoint(original);
    if (opcode == OpCode::PrimitiveID) {
      original->replaceAllUsesWith(primitiveID); original->eraseFromParent();
    } else if (opcode == OpCode::StoreOutput) {
      auto *id = dyn_cast<ConstantInt>(original->getArgOperand(1));
      if (!id || !offsets.count(unsigned(id->getZExtValue()))) { error = "invalid GS output ID"; return false; }
      const auto &e = dm.GetOutputSignature().GetElement(unsigned(id->getZExtValue()));
      auto *index = b.CreateAdd(b.getInt32(offsets[unsigned(id->getZExtValue())]),
          b.CreateAdd(b.CreateMul(original->getArgOperand(2), b.getInt32(e.GetCols())), b.CreateZExt(original->getArgOperand(3), i32)));
      auto *value = original->getArgOperand(4);
      b.CreateStore(value->getType() == i32 ? value : b.CreateBitCast(value, i32), valuePointer(index));
      if (!rasterize) original->eraseFromParent();
    } else if (opcode == OpCode::EmitStream || opcode == OpCode::EmitThenCutStream || opcode == OpCode::CutStream) {
      auto *streamID = dyn_cast<ConstantInt>(original->getArgOperand(1));
      if (!streamID || streamID->getZExtValue() >= 4 ||
          !(layout.stream_mask & (1u << streamID->getZExtValue()))) {
        error = "invalid geometry emission stream"; return false;
      }
      const unsigned stream = unsigned(streamID->getZExtValue());
      auto *count = b.CreateGEP(counts, {b.getInt32(0), b.getInt32(stream)});
      auto *streamBase = b.CreateAdd(base, b.getInt32(stream * streamStride));
      auto *strip = b.CreateGEP(strips, {b.getInt32(0), b.getInt32(stream)});
      if (opcode != OpCode::CutStream) {
        if (!rasterize) for(unsigned col=0;col<4;++col)
          call(OpCode::StoreOutput,b.getFloatTy(),{b.getInt32(0),b.getInt32(0),b.getInt8(col),
              ConstantFP::get(b.getFloatTy(),col==3?1.0:0.0)});
        auto *emitted = b.CreateLoad(count);
        auto *safeIndex = b.CreateSelect(b.CreateICmpULT(emitted, b.getInt32(layout.max_vertices)), emitted, b.getInt32(layout.max_vertices - 1));
        auto *record = b.CreateAdd(streamBase, b.CreateAdd(b.getInt32(16), b.CreateMul(safeIndex, b.getInt32(recordStride))));
        Value *id = primitiveComponent == UINT32_MAX ? primitiveID : b.CreateLoad(valuePointer(b.getInt32(primitiveComponent)));
        store(record, id);
        store(b.CreateAdd(record, b.getInt32(4)), b.CreateLoad(strip));
        for (unsigned i = 0; i < offset; ++i)
          store(b.CreateAdd(record, b.getInt32(8 + i * 4)), b.CreateLoad(valuePointer(b.getInt32(i))));
        auto *next = b.CreateAdd(safeIndex, b.getInt32(1));
        b.CreateStore(next, count); b.CreateStore(b.CreateAdd(b.CreateLoad(strip), b.getInt32(1)), strip);
      }
      if (opcode != OpCode::EmitStream) b.CreateStore(b.getInt32(0), strip);
    }
  }
  for (auto &block : *entry) if (auto *ret = dyn_cast<ReturnInst>(block.getTerminator())) {
    b.SetInsertPoint(ret);
    for (unsigned stream = 0; stream < streamCount; ++stream)
      store(b.CreateAdd(base, b.getInt32(stream * streamStride)),
          b.CreateLoad(b.CreateGEP(counts, {b.getInt32(0), b.getInt32(stream)})));
  }
  if (!rasterize) {
    dm.GetOutputSignature().ClearElements();
    dm.GetSerializedViewIdState().clear();
    auto position=dm.GetOutputSignature().CreateElement();
    position->Initialize("SV_Position",CompType::getF32(),DXIL::InterpolationMode::Linear,
        1,4,0,0,DxilSignatureElement::kUndefinedID,{0});
    position->SetKind(DXIL::SemanticKind::Position);position->SetUsageMask(15);
    dm.GetOutputSignature().AppendElement(std::move(position));
  }
  dm.CollectShaderFlagsForModule();
  auto *module = dm.GetModule();
  for (auto it = module->begin(); it != module->end();) {
    auto *function = &*it++;
    if (function->isDeclaration() && function->use_empty()) {
      dm.RemoveFunction(function); function->eraseFromParent();
    }
  }
  if (verifyModule(*dm.GetModule())) { error = "invalid geometry capture IR"; return false; }
  return true;
}

std::unique_ptr<llvm::Module> CreateGeometryCaptureModule(
    const hlsl::DxilModule &source, uint32_t registerSpace,
    uint32_t verticesPerPrimitive, bool capture, uint32_t &vertexStride,
    std::string &error, const ScorpiGeometryCaptureLayout *generated, bool compacted, bool tessellated, bool passthrough, bool discardCapture,
    uint32_t rasterizedStream) {
  using namespace llvm;
  using namespace hlsl;
  using DXIL::OpCode;
  auto fail = [&](const char *why) -> std::unique_ptr<Module> {
    error = why;
    return nullptr;
  };
  if ((!source.GetShaderModel()->IsVS() && !generated && !((tessellated || passthrough) && source.GetShaderModel()->IsDS())) || verticesPerPrimitive < 1 ||
      verticesPerPrimitive > 3)
    return fail("geometry capture requires VS output and point/line/triangle input");
  if (generated && !capture &&
      (rasterizedStream >= 4 || !(generated->stream_mask & (1u << rasterizedStream))))
    return fail("invalid generated geometry rasterized stream");
  vertexStride = 0;
  for (const auto &e : source.GetOutputSignature().GetElements()) {
    if (e->GetCompType().Is64Bit() || e->GetCompType().Is16Bit())
      return fail("geometry capture requires 32-bit signature storage");
    vertexStride += e->GetRows() * e->GetCols() * 4;
  }
  if (!vertexStride)
    return fail("geometry capture requires a nonempty vertex signature");
  auto &ctx = source.GetCtx();
  auto module = std::make_unique<Module>(capture ? "ScorpiCapture" : "ScorpiReplay", ctx);
  module->setDataLayout(source.GetModule()->getDataLayout());
  module->setTargetTriple(source.GetModule()->getTargetTriple());
  auto &dm = module->GetOrCreateDxilModule(true);
  dm.SetShaderModel(ShaderModel::GetByName(capture ? "gs_6_0" : "vs_6_0"), false);
  dm.SetValidatorVersion(1, 8);
  auto *entry = Function::Create(FunctionType::get(Type::getVoidTy(ctx), false),
                                GlobalValue::ExternalLinkage, "main", module.get());
  dm.SetEntryFunction(entry);
  dm.SetEntryFunctionName("main");
  dm.GetTypeSystem().AddFunctionAnnotation(entry);
  if (capture) {
    dm.SetInputPrimitive(verticesPerPrimitive == 3 ? DXIL::InputPrimitive::Triangle
                         : verticesPerPrimitive == 2 ? DXIL::InputPrimitive::Line
                                                    : DXIL::InputPrimitive::Point);
    dm.SetStreamPrimitiveTopology(verticesPerPrimitive == 3 ? DXIL::PrimitiveTopology::TriangleStrip
                                  : verticesPerPrimitive == 2 ? DXIL::PrimitiveTopology::LineStrip
                                                             : DXIL::PrimitiveTopology::PointList);
    // The private capture shader is also a staging boundary for wide domain
    // outputs. A conservative output bound limits the converter's group size;
    // otherwise it places too many wide DS results in threadgroup memory.
    // This changes only scheduling, not the number of vertices emitted.
    dm.SetMaxVertexCount(tessellated && vertexStride > 256 ? 32 : verticesPerPrimitive);
    dm.SetGSInstanceCount(1);
    dm.SetActiveStreamMask(1);
  }
  auto copyElement = [&](DxilSignature &signature, const DxilSignatureElement &e) {
    auto result = signature.CreateElement();
    result->Initialize(e.GetName(), e.GetCompType(), *e.GetInterpolationMode(),
                      e.GetRows(), e.GetCols(), e.GetStartRow(), e.GetStartCol(),
                      DxilSignatureElement::kUndefinedID, e.GetSemanticIndexVec());
    result->SetKind(e.GetKind());
    result->SetUsageMask((1u << e.GetCols()) - 1);
    return signature.AppendElement(std::move(result));
  };
  for (const auto &e : source.GetOutputSignature().GetElements()) {
    if (generated && e->GetKind() == DXIL::SemanticKind::PrimitiveID) continue;
    if (generated && !capture && e->GetOutputStream() != rasterizedStream) continue;
    if (capture) copyElement(dm.GetInputSignature(), *e);
    if (!(tessellated && capture)) copyElement(dm.GetOutputSignature(), *e);
  }
  if (tessellated && capture) AddTessellationCaptureSignature(dm.GetInputSignature(),vertexStride/4);
  if (tessellated && capture) {
    // The capture stage writes application attributes to the UAV, not to a
    // raster stage. Keep only position in its mesh output; forwarding the full
    // application signature needlessly consumes Metal's output attributes.
    auto position = dm.GetOutputSignature().CreateElement();
    position->Initialize("SV_Position",CompType::getF32(),DXIL::InterpolationMode::Linear,
        1,4,0,0,DxilSignatureElement::kUndefinedID,{0});
    position->SetKind(DXIL::SemanticKind::Position);position->SetUsageMask(15);
    dm.GetOutputSignature().AppendElement(std::move(position));
  }
  unsigned vertexID = 0;
  if (!capture) {
    auto element = dm.GetInputSignature().CreateElement();
    element->Initialize("SV_VertexID", CompType::getU32(),
                        DXIL::InterpolationMode::Undefined, 1, 1, 0, 0);
    element->SetKind(DXIL::SemanticKind::VertexID);
    element->SetUsageMask(1);
    element->AppendSemanticIndex(0);
    vertexID = dm.GetInputSignature().AppendElement(std::move(element));
  }
  auto *i32 = Type::getInt32Ty(ctx);
  if (!generated) {
  auto resource = std::make_unique<DxilResource>();
  resource->SetID(0);resource->SetRW(true);resource->SetKind(DXIL::ResourceKind::RawBuffer);
  resource->SetSpaceID(registerSpace);resource->SetLowerBound(0);resource->SetRangeSize(1);
  resource->SetGlobalName("ScorpiCapturedGeometry");
  auto *resourceType = StructType::create(ctx, {i32}, "ScorpiCapturedGeometry");
  resource->SetGlobalSymbol(UndefValue::get(resourceType->getPointerTo()));
  dm.AddUAV(std::move(resource));
  }
  IRBuilder<> b(BasicBlock::Create(ctx, "entry", entry));
  auto call = [&](OpCode op, Type *type, std::vector<Value *> args) {
    args.insert(args.begin(),b.getInt32(static_cast<unsigned>(op)));
    return b.CreateCall(dm.GetOP()->GetOpFunc(op,type),args);
  };
  Value *handle = nullptr, *parameters = nullptr;
  if (generated) {
    auto binding = AddCaptureBindings(dm, b, registerSpace);
    handle = binding.storage; parameters = binding.parameters;
  } else handle=call(OpCode::CreateHandle,b.getVoidTy(),
      {b.getInt8(static_cast<unsigned>(DXIL::ResourceClass::UAV)),b.getInt32(0),b.getInt32(0),b.getFalse()});
  Value *firstVertex;
  if (capture && tessellated) firstVertex=b.getInt32(0);
  else if (capture)
    firstVertex=b.CreateMul(call(OpCode::PrimitiveID,i32,{}),b.getInt32(verticesPerPrimitive));
  else
    firstVertex=call(OpCode::LoadInput,i32,{b.getInt32(vertexID),b.getInt32(0),b.getInt8(0),UndefValue::get(i32)});
  Value *tessellationBase=nullptr,*tessellationPatch=nullptr;
  if(tessellated && capture) {
    Value *last=b.getInt32(0);
    for(unsigned vertex=0;vertex<verticesPerPrimitive;++vertex) {
      auto *key=call(OpCode::LoadInput,i32,{b.getInt32(0),b.getInt32(0),b.getInt8(1),b.getInt32(vertex)});
      last=b.CreateSelect(b.CreateICmpUGT(key,last),key,last);
    }
    tessellationPatch=call(OpCode::LoadInput,i32,{b.getInt32(0),b.getInt32(0),b.getInt8(0),b.getInt32(0)});
    tessellationBase=b.CreateMul(last,b.getInt32(16+verticesPerPrimitive*(vertexStride+8)));
  }
  Value *valid = b.getTrue(), *firstRecord = nullptr;
  if (compacted) {
    auto *table = b.CreateExtractValue(parameters, 3);
    auto *primitive = b.CreateUDiv(firstVertex, b.getInt32(verticesPerPrimitive));
    auto *vertex = b.CreateURem(firstVertex, b.getInt32(verticesPerPrimitive));
    auto *address = b.CreateAdd(table, b.CreateAdd(b.CreateMul(primitive, b.getInt32(16)), b.CreateMul(vertex, b.getInt32(4))));
    firstRecord = b.CreateExtractValue(call(OpCode::BufferLoad, i32, {handle, address, UndefValue::get(i32)}), 0);
  } else if (generated) {
    auto load = [&](Value *address) {
      return b.CreateExtractValue(call(OpCode::BufferLoad, i32, {handle, address, UndefValue::get(i32)}), 0);
    };
    auto *instanceBase = b.CreateExtractValue(parameters, 2);
    firstRecord = b.CreateExtractValue(parameters, 3);
    const unsigned recordStride = generated->vertex_stride + 8;
    auto *lastRecord = b.CreateAdd(firstRecord, b.getInt32((verticesPerPrimitive - 1) * recordStride));
    auto *lastIndex = b.CreateUDiv(b.CreateSub(b.CreateSub(lastRecord, instanceBase), b.getInt32(16)), b.getInt32(recordStride));
    auto *stripIndex = load(b.CreateAdd(lastRecord, b.getInt32(4)));
    valid = b.CreateAnd(b.CreateICmpULT(lastIndex, load(instanceBase)), b.CreateICmpUGE(stripIndex, b.getInt32(verticesPerPrimitive - 1)));
    if (verticesPerPrimitive == 3) {
      auto *odd = b.CreateICmpNE(b.CreateAnd(stripIndex, b.getInt32(1)), b.getInt32(0));
      auto *swap = b.CreateAnd(odd, b.CreateICmpNE(firstVertex, b.getInt32(0)));
      firstVertex = b.CreateSelect(swap, b.CreateSub(b.getInt32(3), firstVertex), firstVertex);
    }
  }
  for (unsigned vertex=0;vertex<(capture?verticesPerPrimitive:1);++vertex) {
    Value *base=b.CreateMul(b.CreateAdd(firstVertex,b.getInt32(vertex)),b.getInt32(vertexStride));
    if (compacted) base = b.CreateAdd(firstRecord, b.getInt32(8));
    else if (generated) base = b.CreateAdd(b.CreateAdd(firstRecord, b.getInt32(8)), b.CreateMul(firstVertex, b.getInt32(generated->vertex_stride + 8)));
    if(tessellated && capture) {
      base=b.CreateAdd(tessellationBase,b.getInt32(16+vertex*(vertexStride+8)+8));
      if(!discardCapture) call(OpCode::BufferStore,i32,{handle,b.CreateSub(base,b.getInt32(8)),UndefValue::get(i32),tessellationPatch,
          UndefValue::get(i32),UndefValue::get(i32),UndefValue::get(i32),b.getInt8(1)});
      if(!discardCapture) call(OpCode::BufferStore,i32,{handle,b.CreateSub(base,b.getInt32(4)),UndefValue::get(i32),b.getInt32(vertex),
          UndefValue::get(i32),UndefValue::get(i32),UndefValue::get(i32),b.getInt8(1)});
    }
    unsigned offset=0,id=0;
    for (const auto &e:source.GetOutputSignature().GetElements()) {
      if (generated && e->GetKind() == DXIL::SemanticKind::PrimitiveID) {
        offset += e->GetRows() * e->GetCols() * 4;
        continue;
      }
      if (generated && !capture && e->GetOutputStream() != rasterizedStream) {
        offset += e->GetRows() * e->GetCols() * 4;
        continue;
      }
      auto *type=e->GetCompType().GetLLVMBaseType(ctx);
      for (unsigned row=0;row<e->GetRows();++row) for(unsigned col=0;col<e->GetCols();++col) {
        auto *address=b.CreateAdd(base,b.getInt32(offset));offset+=4;
        Value *value;
        if(capture) {
          if(tessellated) {
            value=call(OpCode::LoadInput,i32,{b.getInt32(1),b.getInt32((offset-4)/16),b.getInt8(((offset-4)/4)%4),b.getInt32(vertex)});
            if(type!=i32) value=b.CreateBitCast(value,type);
          } else value=call(OpCode::LoadInput,type,{b.getInt32(id),b.getInt32(row),b.getInt8(col),b.getInt32(vertex)});
          auto *bits=type==i32?value:b.CreateBitCast(value,i32);
          if(!passthrough && !discardCapture) call(OpCode::BufferStore,i32,{handle,address,UndefValue::get(i32),bits,UndefValue::get(i32),UndefValue::get(i32),UndefValue::get(i32),b.getInt8(1)});
        } else {
          auto *load=call(OpCode::BufferLoad,i32,{handle,address,UndefValue::get(i32)});
          auto *bits=b.CreateExtractValue(load,0);
          value=type==i32?bits:b.CreateBitCast(bits,type);
        }
        if (generated && e->GetKind() == DXIL::SemanticKind::Position)
          value = b.CreateSelect(valid, value, ConstantFP::get(type, col == 3 ? 1.0 : 2.0));
        if (!(tessellated && capture))
          call(OpCode::StoreOutput,type,{b.getInt32(id),b.getInt32(row),b.getInt8(col),value});
      }
      ++id;
    }
    if(tessellated && capture) for(unsigned col=0;col<4;++col)
      call(OpCode::StoreOutput,b.getFloatTy(),{b.getInt32(0),b.getInt32(0),b.getInt8(col),
          ConstantFP::get(b.getFloatTy(),col==3?1.0:0.0)});
    if(capture)call(OpCode::EmitStream,b.getVoidTy(),{b.getInt8(0)});
  }
  if(tessellated && capture && !discardCapture) call(OpCode::BufferStore,i32,{handle,tessellationBase,UndefValue::get(i32),b.getInt32(verticesPerPrimitive),
      UndefValue::get(i32),UndefValue::get(i32),UndefValue::get(i32),b.getInt8(1)});
  b.CreateRetVoid();
  dm.CollectShaderFlagsForModule();
  if(verifyModule(*module))return fail("invalid geometry capture IR");
  return module;
}
}
