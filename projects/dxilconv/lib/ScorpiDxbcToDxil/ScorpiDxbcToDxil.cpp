#include "llvm/Support/Debug.h" // Must be included first.

#define SCORPI_DXBC_TO_DXIL_BUILD_DLL
#include "ScorpiDxbcToDxil.h"

#include "DxbcConverterImpl.h"
#include "dxc/DxilContainer/DxilContainer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilShaderModel.h"
#include "dxc/DXIL/DxilSignature.h"
#include "dxc/DXIL/DxilSignatureElement.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void EnsureDxcStaticRuntimeInitialized() {
  static std::once_flag once;
  static std::error_code init_error;
  std::call_once(once, []() {
    init_error = llvm::sys::fs::SetupPerThreadFileSystem();
  });
  if (init_error) {
    throw std::runtime_error("failed to initialize DXC filesystem TLS");
  }
}

void EnsureDxcAllocatorInitialized() {
  static std::once_flag once;
  static HRESULT init_hr = S_OK;
  std::call_once(once, []() { init_hr = DxcInitThreadMalloc(); });
  if (FAILED(init_hr)) {
    throw std::runtime_error("failed to initialize DXC allocator");
  }
}

class DxcMallocScope {
public:
  DxcMallocScope() {
    EnsureDxcAllocatorInitialized();
    if (DxcGetThreadMallocNoRef() == nullptr) {
      DxcSetThreadMallocToDefault();
      installed_ = true;
    }
  }

  ~DxcMallocScope() {
    if (installed_) {
      DxcClearThreadMalloc();
    }
  }

  DxcMallocScope(const DxcMallocScope &) = delete;
  DxcMallocScope &operator=(const DxcMallocScope &) = delete;

private:
  bool installed_ = false;
};

class DxcBlobScope {
public:
  DxcBlobScope() = default;

  ~DxcBlobScope() {
    if (blob_ != nullptr) {
      blob_->Release();
    }
  }

  IDxcBlob **Out() { return &blob_; }
  IDxcBlob *Get() const { return blob_; }

  DxcBlobScope(const DxcBlobScope &) = delete;
  DxcBlobScope &operator=(const DxcBlobScope &) = delete;

private:
  IDxcBlob *blob_ = nullptr;
};

struct SignatureView {
  const D3D12DDIARG_SIGNATURE_ENTRY_0012 *input = nullptr;
  const D3D12DDIARG_SIGNATURE_ENTRY_0012 *output = nullptr;
  const D3D12DDIARG_SIGNATURE_ENTRY_0012 *patch = nullptr;
  uint32_t input_count = 0;
  uint32_t output_count = 0;
  uint32_t patch_count = 0;
};

struct SignatureStorage {
  std::vector<D3D12DDIARG_SIGNATURE_ENTRY_0012> input;
  std::vector<D3D12DDIARG_SIGNATURE_ENTRY_0012> output;
  std::vector<D3D12DDIARG_SIGNATURE_ENTRY_0012> patch;

  SignatureView View() const {
    SignatureView view = {};
    view.input = input.empty() ? nullptr : input.data();
    view.output = output.empty() ? nullptr : output.data();
    view.patch = patch.empty() ? nullptr : patch.data();
    view.input_count = static_cast<uint32_t>(input.size());
    view.output_count = static_cast<uint32_t>(output.size());
    view.patch_count = static_cast<uint32_t>(patch.size());
    return view;
  }
};

void SetError(char *error_message, size_t error_message_size,
              const std::string &message) {
  if (error_message == nullptr || error_message_size == 0) {
    return;
  }

  const size_t copy_size =
      std::min(error_message_size - 1, message.size());
  std::memcpy(error_message, message.data(), copy_size);
  error_message[copy_size] = '\0';
}

const ScorpiShaderSignatureEntry *SignaturePointer(
    const uint8_t *data, size_t data_size, uint32_t offset, uint32_t count,
    uint32_t entry_size) {
  if (count == 0) {
    return nullptr;
  }

  const uint64_t bytes = static_cast<uint64_t>(count) * entry_size;
  if (offset > data_size || bytes > data_size - offset) {
    throw std::runtime_error("signature range exceeds signature dump");
  }
  return reinterpret_cast<const ScorpiShaderSignatureEntry *>(data + offset);
}

std::vector<D3D12DDIARG_SIGNATURE_ENTRY_0012> ConvertWireSignatureEntries(
    const ScorpiShaderSignatureEntry *entries, uint32_t count) {
  std::vector<D3D12DDIARG_SIGNATURE_ENTRY_0012> converted;
  converted.reserve(count);
  for (uint32_t index = 0; index < count; ++index) {
    D3D12DDIARG_SIGNATURE_ENTRY_0012 entry = {};
    entry.SystemValue =
        static_cast<D3D10_SB_NAME>(entries[index].system_value);
    entry.Register = entries[index].register_index;
    entry.Mask = static_cast<BYTE>(entries[index].mask);
    entry.Stream = static_cast<BYTE>(entries[index].stream);
    entry.RegisterComponentType =
        static_cast<D3D10_SB_REGISTER_COMPONENT_TYPE>(
            entries[index].component_type);
    entry.MinPrecision =
        static_cast<D3D11_SB_OPERAND_MIN_PRECISION>(
            entries[index].min_precision);
    converted.emplace_back(entry);
  }
  return converted;
}

SignatureView ParseSignatures(const void *signature_dump,
                              size_t signature_dump_size,
                              SignatureStorage *storage) {
  if (storage == nullptr) {
    throw std::invalid_argument("signature storage is null");
  }
  *storage = {};

  if (signature_dump == nullptr || signature_dump_size == 0) {
    return storage->View();
  }
  if (signature_dump_size < sizeof(ScorpiShaderSignatureMetadata)) {
    throw std::runtime_error("signature dump is missing metadata");
  }

  const auto *data = static_cast<const uint8_t *>(signature_dump);
  const auto *metadata =
      reinterpret_cast<const ScorpiShaderSignatureMetadata *>(data);
  if (metadata->version != SCORPI_SHADER_SIGNATURE_METADATA_VERSION) {
    throw std::runtime_error("unsupported signature metadata version");
  }
  if (metadata->signature_entry_size !=
      sizeof(ScorpiShaderSignatureEntry)) {
    throw std::runtime_error("signature entry size does not match DXC");
  }

  storage->input = ConvertWireSignatureEntries(
      SignaturePointer(data, signature_dump_size, metadata->input_offset,
                       metadata->input_count,
                       metadata->signature_entry_size),
      metadata->input_count);
  storage->output = ConvertWireSignatureEntries(
      SignaturePointer(data, signature_dump_size, metadata->output_offset,
                       metadata->output_count,
                       metadata->signature_entry_size),
      metadata->output_count);
  storage->patch = ConvertWireSignatureEntries(
      SignaturePointer(data, signature_dump_size, metadata->patch_offset,
                       metadata->patch_count,
                       metadata->signature_entry_size),
      metadata->patch_count);
  return storage->View();
}

const uint32_t *ValidateShaderTokenStream(const void *shader_code,
                                          size_t shader_code_size) {
  if (shader_code == nullptr || shader_code_size < sizeof(uint32_t) * 2 ||
      (shader_code_size % sizeof(uint32_t)) != 0) {
    throw std::runtime_error("shader code is not a DWORD token stream");
  }

  const auto *token_stream = static_cast<const uint32_t *>(shader_code);
  const uint32_t token_count = token_stream[1];
  if (token_count < 2 || token_count != shader_code_size / sizeof(uint32_t)) {
    throw std::runtime_error("shader token count does not match file size");
  }
  return token_stream;
}

bool IsDxbcContainer(const void *shader_code, size_t shader_code_size) {
  if (shader_code == nullptr ||
      shader_code_size < sizeof(hlsl::DxilContainerHeader) ||
      shader_code_size > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  const auto *container =
      reinterpret_cast<const hlsl::DxilContainerHeader *>(shader_code);
  return container->HeaderFourCC == hlsl::DFCC_Container &&
         container->ContainerSizeInBytes == shader_code_size;
}

bool IsDxilContainer(const void *shader_code, size_t shader_code_size) {
  const auto *container =
      hlsl::IsDxilContainerLike(shader_code, shader_code_size);
  return container != nullptr &&
         hlsl::IsValidDxilContainer(container, shader_code_size) &&
         hlsl::GetDxilProgramHeader(container, hlsl::DFCC_DXIL) != nullptr;
}

bool IsDxilProgram(const void *shader_code, size_t shader_code_size) {
  if (shader_code == nullptr ||
      shader_code_size < sizeof(hlsl::DxilProgramHeader) ||
      shader_code_size > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  const auto *program =
      static_cast<const hlsl::DxilProgramHeader *>(shader_code);
  return hlsl::IsValidDxilProgramHeader(
      program, static_cast<uint32_t>(shader_code_size));
}

std::vector<uint8_t> WrapDxilProgram(const void *dxil, size_t dxil_size,
                                     uint32_t source_version_token) {
  if (IsDxilProgram(dxil, dxil_size)) {
    const auto *program = static_cast<const hlsl::DxilProgramHeader *>(dxil);
    const size_t program_size =
        static_cast<size_t>(program->SizeInUint32) * sizeof(uint32_t);
    return std::vector<uint8_t>(static_cast<const uint8_t *>(dxil),
                                static_cast<const uint8_t *>(dxil) +
                                    program_size);
  }

  if (dxil_size > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("DXIL bitcode exceeds 32-bit size");
  }

  hlsl::DxilProgramHeader header = {};
  const uint32_t source_type =
      DECODE_D3D10_SB_TOKENIZED_PROGRAM_TYPE(source_version_token);
  const uint32_t source_major =
      DECODE_D3D10_SB_TOKENIZED_PROGRAM_MAJOR_VERSION(source_version_token);
  const uint32_t source_minor =
      DECODE_D3D10_SB_TOKENIZED_PROGRAM_MINOR_VERSION(source_version_token);
  const uint32_t shader_version =
      hlsl::EncodeVersion(static_cast<hlsl::DXIL::ShaderKind>(source_type),
                          source_major, source_minor);
  const uint32_t dxil_version = hlsl::DXIL::MakeDxilVersion(1, 0);
  hlsl::InitProgramHeader(header, shader_version, dxil_version,
                          static_cast<uint32_t>(dxil_size));

  const size_t padded_size =
      sizeof(header) + ((dxil_size + sizeof(uint32_t) - 1) & ~size_t(3));
  std::vector<uint8_t> program(padded_size);
  std::memcpy(program.data(), &header, sizeof(header));
  std::memcpy(program.data() + sizeof(header), dxil, dxil_size);
  return program;
}

std::vector<uint8_t> WrapDxilContainer(const std::vector<uint8_t> &program) {
  if (program.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("DXIL program exceeds 32-bit size");
  }

  const uint32_t part_count = 1;
  const size_t part_offset_table_size = sizeof(uint32_t) * part_count;
  const size_t part_offset =
      sizeof(hlsl::DxilContainerHeader) + part_offset_table_size;
  const size_t container_size =
      part_offset + sizeof(hlsl::DxilPartHeader) + program.size();
  if (container_size > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("DXIL container exceeds 32-bit size");
  }

  std::vector<uint8_t> container(container_size);
  auto *header =
      reinterpret_cast<hlsl::DxilContainerHeader *>(container.data());
  hlsl::InitDxilContainer(header, part_count,
                          static_cast<uint32_t>(container_size));

  auto *offset_table = reinterpret_cast<uint32_t *>(header + 1);
  offset_table[0] = static_cast<uint32_t>(part_offset);

  auto *part =
      reinterpret_cast<hlsl::DxilPartHeader *>(container.data() + part_offset);
  part->PartFourCC = hlsl::DFCC_DXIL;
  part->PartSize = static_cast<uint32_t>(program.size());
  std::memcpy(part + 1, program.data(), program.size());
  return container;
}

void WriteFile(const char *path, const void *data, size_t size) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("failed to open output file");
  }
  stream.write(static_cast<const char *>(data),
               static_cast<std::streamsize>(size));
  if (!stream) {
    throw std::runtime_error("failed to write output file");
  }
}

std::vector<uint8_t> NormalizeExistingDxil(const void *shader_code,
                                           size_t shader_code_size) {
  if (IsDxilContainer(shader_code, shader_code_size)) {
    return std::vector<uint8_t>(
        static_cast<const uint8_t *>(shader_code),
        static_cast<const uint8_t *>(shader_code) + shader_code_size);
  }

  if (IsDxilProgram(shader_code, shader_code_size)) {
    const auto *program =
        static_cast<const hlsl::DxilProgramHeader *>(shader_code);
    const size_t program_size =
        static_cast<size_t>(program->SizeInUint32) * sizeof(uint32_t);
    return WrapDxilContainer(std::vector<uint8_t>(
        static_cast<const uint8_t *>(shader_code),
        static_cast<const uint8_t *>(shader_code) + program_size));
  }

  return {};
}

std::vector<uint8_t>
ConvertTokenStreamToDxilContainer(const void *shader_code,
                                  size_t shader_code_size,
                                  const void *signature_dump,
                                  size_t signature_dump_size) {
  const uint32_t *token_stream =
      ValidateShaderTokenStream(shader_code, shader_code_size);
  SignatureStorage signature_storage;
  const SignatureView signatures =
      ParseSignatures(signature_dump, signature_dump_size, &signature_storage);

  EnsureDxcStaticRuntimeInitialized();
  DxcMallocScope dxc_malloc;

  hlsl::DxbcConverter converter;
  void *dxil_binary = nullptr;
  uint32_t dxil_size = 0;
  LPWSTR diagnostics = nullptr;
  const HRESULT status = converter.ConvertInDriverToContainer(
      token_stream, signatures.input, signatures.input_count, signatures.output,
      signatures.output_count, signatures.patch, signatures.patch_count, nullptr,
      &dxil_binary, &dxil_size, &diagnostics);
  if (FAILED(status)) {
    char message[128] = {};
    std::snprintf(message, sizeof(message),
                  "shader token stream conversion failed hr=0x%08x",
                  static_cast<unsigned>(status));
    throw std::runtime_error(message);
  }
  if (dxil_binary == nullptr || dxil_size == 0) {
    throw std::runtime_error(
        "shader token stream conversion succeeded without DXIL container");
  }

  CComHeapPtr<void> dxil_holder(dxil_binary);
  return std::vector<uint8_t>(static_cast<const uint8_t *>(dxil_binary),
                              static_cast<const uint8_t *>(dxil_binary) +
                                  dxil_size);
}

std::vector<uint8_t> ConvertToDxilBinary(const void *shader_code,
                                         size_t shader_code_size,
                                         const void *signature_dump,
                                         size_t signature_dump_size) {
  if (shader_code == nullptr || shader_code_size == 0) {
    throw std::invalid_argument("shader code is empty");
  }

  std::vector<uint8_t> existing =
      NormalizeExistingDxil(shader_code, shader_code_size);
  if (!existing.empty()) {
    return existing;
  }

  if (!IsDxbcContainer(shader_code, shader_code_size)) {
    return ConvertTokenStreamToDxilContainer(
        shader_code, shader_code_size, signature_dump, signature_dump_size);
  }

  EnsureDxcStaticRuntimeInitialized();
  DxcMallocScope dxc_malloc;

  hlsl::DxbcConverter converter;
  void *dxil_binary = nullptr;
  uint32_t dxil_size = 0;
  LPWSTR diagnostics = nullptr;
  const HRESULT status = converter.Convert(
      shader_code, static_cast<uint32_t>(shader_code_size), nullptr,
      &dxil_binary, &dxil_size, &diagnostics);
  if (FAILED(status)) {
    throw std::runtime_error("shader conversion failed");
  }
  if (dxil_binary == nullptr || dxil_size == 0) {
    throw std::runtime_error("conversion succeeded without DXIL blob");
  }

  CComHeapPtr<void> dxil_holder(dxil_binary);
  std::vector<uint8_t> converted(
      static_cast<const uint8_t *>(dxil_binary),
      static_cast<const uint8_t *>(dxil_binary) + dxil_size);
  return converted;
}

int ConvertToDxilFile(const void *shader_code, size_t shader_code_size,
                      const void *signature_dump,
                      size_t signature_dump_size, const char *output_path) {
  if (output_path == nullptr || output_path[0] == '\0') {
    throw std::invalid_argument("output path is empty");
  }

  const std::vector<uint8_t> dxil_container =
      ConvertToDxilBinary(shader_code, shader_code_size, signature_dump,
                          signature_dump_size);
  WriteFile(output_path, dxil_container.data(), dxil_container.size());
  return SCORPI_DXBC_TO_DXIL_OK;
}

} // namespace

extern "C" int ScorpiCreateDxilBinaryFile(
    const void *shader_code, size_t shader_code_size,
    const void *signature_dump, size_t signature_dump_size,
    const char *output_path, char *error_message, size_t error_message_size) {
  SetError(error_message, error_message_size, "");

  try {
    return ConvertToDxilFile(shader_code, shader_code_size, signature_dump,
                             signature_dump_size, output_path);
  } catch (const std::invalid_argument &error) {
    SetError(error_message, error_message_size, error.what());
    return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  } catch (const std::runtime_error &error) {
    const std::string message = error.what();
    SetError(error_message, error_message_size, message);
    if (message.find("signature") != std::string::npos) {
      return SCORPI_DXBC_TO_DXIL_INVALID_SIGNATURES;
    }
    if (message.find("shader") != std::string::npos ||
        message.find("DXIL") != std::string::npos) {
      return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
    }
    if (message.find("conversion") != std::string::npos) {
      return SCORPI_DXBC_TO_DXIL_CONVERSION_FAILED;
    }
    if (message.find("output") != std::string::npos ||
        message.find("file") != std::string::npos) {
      return SCORPI_DXBC_TO_DXIL_IO_FAILED;
    }
    return SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR;
  } catch (...) {
    SetError(error_message, error_message_size, "unexpected exception");
    return SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR;
  }
}

extern "C" int ScorpiCreateDxilBinary(
    const void *shader_code, size_t shader_code_size,
    const void *signature_dump, size_t signature_dump_size, void **dxil_binary,
    size_t *dxil_binary_size, char *error_message, size_t error_message_size) {
  SetError(error_message, error_message_size, "");
  if (dxil_binary == nullptr || dxil_binary_size == nullptr) {
    SetError(error_message, error_message_size, "output pointer is null");
    return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  }
  *dxil_binary = nullptr;
  *dxil_binary_size = 0;

  try {
    const std::vector<uint8_t> dxil_container =
        ConvertToDxilBinary(shader_code, shader_code_size, signature_dump,
                            signature_dump_size);
    if (dxil_container.empty()) {
      SetError(error_message, error_message_size, "DXIL container is empty");
      return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
    }

    void *buffer = std::malloc(dxil_container.size());
    if (buffer == nullptr) {
      SetError(error_message, error_message_size, "failed to allocate output");
      return SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR;
    }
    std::memcpy(buffer, dxil_container.data(), dxil_container.size());
    *dxil_binary = buffer;
    *dxil_binary_size = dxil_container.size();
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (const std::invalid_argument &error) {
    SetError(error_message, error_message_size, error.what());
    return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  } catch (const std::runtime_error &error) {
    const std::string message = error.what();
    SetError(error_message, error_message_size, message);
    if (message.find("signature") != std::string::npos) {
      return SCORPI_DXBC_TO_DXIL_INVALID_SIGNATURES;
    }
    if (message.find("shader") != std::string::npos ||
        message.find("DXIL") != std::string::npos) {
      return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
    }
    if (message.find("conversion") != std::string::npos) {
      return SCORPI_DXBC_TO_DXIL_CONVERSION_FAILED;
    }
    return SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR;
  } catch (...) {
    SetError(error_message, error_message_size, "unexpected exception");
    return SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR;
  }
}

extern "C" void ScorpiFreeDxilBinary(void *dxil_binary) {
  std::free(dxil_binary);
}

// This deliberately recognizes a narrow, side-effect-free mask producer rather
// than inferring shader behavior from a hash or from the output signature alone.
extern "C" uint32_t ScorpiInspectCoverageShader(
    const void *code, size_t size, const void *metadata, size_t metadata_size) {
  try {
    DxcMallocScope malloc_scope;
    auto binary = ConvertToDxilBinary(code, size, metadata, metadata_size);
    auto *container = reinterpret_cast<const hlsl::DxilContainerHeader *>(binary.data());
    if (!hlsl::IsValidDxilContainer(container, binary.size())) return 0;
    auto *program = hlsl::GetDxilProgramHeader(container, hlsl::DFCC_DXIL);
    if (!program) return 0;
    llvm::LLVMContext context;
    auto buffer = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(hlsl::GetDxilBitcodeData(program),
                        hlsl::GetDxilBitcodeSize(program)), "coverage", false);
    auto parsed = llvm::parseBitcodeFile(buffer->getMemBufferRef(), context);
    if (!parsed) return 0;
    auto &module = *parsed.get();
    auto *dxil = hlsl::DxilModule::TryGetDxilModule(&module);
    if (!dxil || !dxil->GetUAVs().empty() ||
        dxil->m_ShaderFlags.GetResourceDescriptorHeapIndexing()) return 0;
    if (dxil->GetShaderModel()->IsVS()) return SCORPI_COVERAGE_REPLAY_SAFE_VERTEX;
    if (!dxil->GetShaderModel()->IsPS()) return 0;
    const auto &outputs = dxil->GetOutputSignature().GetElements();
    if (outputs.size() != 1 || outputs[0]->GetKind() != hlsl::DXIL::SemanticKind::Target ||
        outputs[0]->GetSemanticStartIndex() != 0 || outputs[0]->GetRows() != 1 ||
        outputs[0]->GetCompType().GetKind() != hlsl::DXIL::ComponentType::U32) return 0;
    auto *entry = dxil->GetEntryFunction();
    if (!entry || entry->size() != 1) return 0;
    llvm::CallInst *coverage = nullptr;
    unsigned components = 0;
    auto constant = [](llvm::Value *v, uint64_t expected) {
      auto *c = llvm::dyn_cast<llvm::ConstantInt>(v);
      return c && c->getZExtValue() == expected;
    };
    for (auto &inst : entry->front()) {
      if (llvm::isa<llvm::ReturnInst>(inst)) continue;
      auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
      if (!call || !call->getCalledFunction()) return 0;
      auto name = call->getCalledFunction()->getName();
      if (name == "dx.op.coverage.i32" && call->getNumArgOperands() == 1 &&
          constant(call->getArgOperand(0), 91) && !coverage) {
        coverage = call;
      } else if (name == "dx.op.storeOutput.i32" &&
                 call->getNumArgOperands() == 5 && coverage &&
                 constant(call->getArgOperand(0), 5) &&
                 constant(call->getArgOperand(1), 0) &&
                 constant(call->getArgOperand(2), 0) &&
                 call->getArgOperand(4) == coverage) {
        auto *column = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(3));
        if (!column || column->getZExtValue() > 3) return 0;
        unsigned bit = 1u << column->getZExtValue();
        if (components & bit) return 0;
        components |= bit;
      } else return 0;
    }
    return coverage && (components & 1) ? SCORPI_COVERAGE_MASK_PIXEL : 0;
  } catch (...) { return 0; }
}
