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
  auto uavID = dm.AddUAV(std::move(resource));
  auto cb = std::make_unique<DxilCBuffer>();
  cb->SetID(unsigned(dm.GetCBuffers().size())); cb->SetSize(32);
  cb->SetSpaceID(space); cb->SetLowerBound(0); cb->SetRangeSize(1);
  cb->SetGlobalName("ScorpiRasterReplay");
  cb->SetGlobalSymbol(UndefValue::get(StructType::create(dm.GetCtx(), {i32, i32, i32, i32, i32, i32, i32, i32}, "ScorpiRasterReplay")->getPointerTo()));
  auto cbID = dm.AddCBuffer(std::move(cb));
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

bool LowerGeometryPrimitiveId(hlsl::DxilModule &dm, std::string &error) {
  using namespace hlsl;
  if (!dm.GetShaderModel()->IsGS() && !dm.GetShaderModel()->IsPS()) {
    error = "geometry PrimitiveID linking requires GS or PS"; return false;
  }
  auto &signature = dm.GetShaderModel()->IsGS() ? dm.GetOutputSignature() : dm.GetInputSignature();
  for (auto &element : signature.GetElements()) {
    if (element->GetKind() != DXIL::SemanticKind::PrimitiveID) continue;
    if (!element->IsAllocated()) { error = "unallocated geometry PrimitiveID signature"; return false; }
    auto indices = element->GetSemanticIndexVec();
    auto usage = element->GetUsageMask();
    element->Initialize("SCORPI_GS_PRIMITIVE_ID", element->GetCompType(),
        DXIL::InterpolationMode::Constant, element->GetRows(), element->GetCols(),
        element->GetStartRow(), element->GetStartCol(), element->GetID(), indices);
    element->SetKind(DXIL::SemanticKind::Arbitrary); element->SetUsageMask(usage);
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
  if (!source.GetShaderModel()->IsGS() || source.GetActiveStreamMask() != 1) {
    error = "generated geometry capture requires a single rasterized GS stream";
    return false;
  }
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
    const ScorpiGeometryCaptureLayout &layout, std::string &error) {
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
  auto *base = b.CreateMul(instance, b.getInt32(16 + layout.max_vertices * recordStride));
  auto *count = b.CreateAlloca(i32, nullptr, "capture.count"); b.CreateStore(b.getInt32(0), count);
  auto *strip = b.CreateAlloca(i32, nullptr, "capture.strip"); b.CreateStore(b.getInt32(0), strip);
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
    } else if (opcode == OpCode::EmitStream || opcode == OpCode::EmitThenCutStream || opcode == OpCode::CutStream) {
      if (opcode != OpCode::CutStream) {
        auto *emitted = b.CreateLoad(count);
        auto *safeIndex = b.CreateSelect(b.CreateICmpULT(emitted, b.getInt32(layout.max_vertices)), emitted, b.getInt32(layout.max_vertices - 1));
        auto *record = b.CreateAdd(base, b.CreateAdd(b.getInt32(16), b.CreateMul(safeIndex, b.getInt32(recordStride))));
        Value *id = primitiveComponent == UINT32_MAX ? primitiveID : b.CreateLoad(valuePointer(b.getInt32(primitiveComponent)));
        store(record, id); store(b.CreateAdd(record, b.getInt32(4)), b.CreateLoad(strip));
        for (unsigned i = 0; i < offset; ++i)
          store(b.CreateAdd(record, b.getInt32(8 + i * 4)), b.CreateLoad(valuePointer(b.getInt32(i))));
        auto *next = b.CreateAdd(safeIndex, b.getInt32(1));
        b.CreateStore(next, count); b.CreateStore(b.CreateAdd(b.CreateLoad(strip), b.getInt32(1)), strip);
      }
      if (opcode != OpCode::EmitStream) b.CreateStore(b.getInt32(0), strip);
    }
  }
  for (auto &block : *entry) if (auto *ret = dyn_cast<ReturnInst>(block.getTerminator())) {
    b.SetInsertPoint(ret); store(base, b.CreateLoad(count));
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
    std::string &error, const ScorpiGeometryCaptureLayout *generated, bool compacted) {
  using namespace llvm;
  using namespace hlsl;
  using DXIL::OpCode;
  auto fail = [&](const char *why) -> std::unique_ptr<Module> {
    error = why;
    return nullptr;
  };
  if ((!source.GetShaderModel()->IsVS() && !generated) || verticesPerPrimitive < 1 ||
      verticesPerPrimitive > 3)
    return fail("geometry capture requires VS output and point/line/triangle input");
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
    dm.SetMaxVertexCount(verticesPerPrimitive);
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
    if (capture) copyElement(dm.GetInputSignature(), *e);
    copyElement(dm.GetOutputSignature(), *e);
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
  if (capture)
    firstVertex=b.CreateMul(call(OpCode::PrimitiveID,i32,{}),b.getInt32(verticesPerPrimitive));
  else
    firstVertex=call(OpCode::LoadInput,i32,{b.getInt32(vertexID),b.getInt32(0),b.getInt8(0),UndefValue::get(i32)});
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
    unsigned offset=0,id=0;
    for (const auto &e:source.GetOutputSignature().GetElements()) {
      if (generated && e->GetKind() == DXIL::SemanticKind::PrimitiveID) {
        offset += e->GetRows() * e->GetCols() * 4;
        continue;
      }
      auto *type=e->GetCompType().GetLLVMBaseType(ctx);
      for (unsigned row=0;row<e->GetRows();++row) for(unsigned col=0;col<e->GetCols();++col) {
        auto *address=b.CreateAdd(base,b.getInt32(offset));offset+=4;
        Value *value;
        if(capture) {
          value=call(OpCode::LoadInput,type,{b.getInt32(id),b.getInt32(row),b.getInt8(col),b.getInt32(vertex)});
          auto *bits=type==i32?value:b.CreateBitCast(value,i32);
          call(OpCode::BufferStore,i32,{handle,address,UndefValue::get(i32),bits,UndefValue::get(i32),UndefValue::get(i32),UndefValue::get(i32),b.getInt8(1)});
        } else {
          auto *load=call(OpCode::BufferLoad,i32,{handle,address,UndefValue::get(i32)});
          auto *bits=b.CreateExtractValue(load,0);
          value=type==i32?bits:b.CreateBitCast(bits,type);
        }
        if (generated && e->GetKind() == DXIL::SemanticKind::Position)
          value = b.CreateSelect(valid, value, ConstantFP::get(type, col == 3 ? 1.0 : 2.0));
        call(OpCode::StoreOutput,type,{b.getInt32(id),b.getInt32(row),b.getInt8(col),value});
      }
      ++id;
    }
    if(capture)call(OpCode::EmitStream,b.getVoidTy(),{b.getInt8(0)});
  }
  b.CreateRetVoid();
  dm.CollectShaderFlagsForModule();
  if(verifyModule(*module))return fail("invalid geometry capture IR");
  return module;
}
}
