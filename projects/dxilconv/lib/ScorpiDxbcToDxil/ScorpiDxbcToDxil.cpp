#include "llvm/Support/Debug.h" // Must be included first.

#define SCORPI_DXBC_TO_DXIL_BUILD_DLL
#include "ScorpiDxbcToDxil.h"
#include "ScorpiRasterReplay.h"
#include "ScorpiGeometryCapture.h"
#include "ScorpiTypedBuffers.h"
#include "ScorpiAtomic64.h"

#include "DxbcConverterImpl.h"
#include "dxc/DxilHash/DxilHash.h"
#include "dxc/DxilContainer/DxilContainer.h"
#include "dxc/DxilContainer/DxilContainerAssembler.h"
#include "dxc/Support/dxcapi.impl.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/Support/MemoryBuffer.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilShaderModel.h"
#include "dxc/DXIL/DxilSignature.h"
#include "dxc/DXIL/DxilSignatureElement.h"
#include "dxc/DXIL/DxilResource.h"
#include "dxc/DXIL/DxilResourceProperties.h"
#include "dxc/DXIL/DxilResourceBinding.h"
#include "dxc/DXIL/DxilOperations.h"
#include "llvm/IR/Verifier.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
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
  uint32_t flags = 0;
  const D3D12DDIARG_SIGNATURE_ENTRY_0012 *input = nullptr;
  const D3D12DDIARG_SIGNATURE_ENTRY_0012 *output = nullptr;
  const D3D12DDIARG_SIGNATURE_ENTRY_0012 *patch = nullptr;
  uint32_t input_count = 0;
  uint32_t output_count = 0;
  uint32_t patch_count = 0;
};

struct SignatureStorage {
  uint32_t flags = 0;
  std::vector<D3D12DDIARG_SIGNATURE_ENTRY_0012> input;
  std::vector<D3D12DDIARG_SIGNATURE_ENTRY_0012> output;
  std::vector<D3D12DDIARG_SIGNATURE_ENTRY_0012> patch;

  SignatureView View() const {
    SignatureView view = {};
    view.flags = flags;
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

  const uint32_t allowed = SCORPI_SHADER_SIGNATURE_STREAM_OUTPUT_ONLY |
      SCORPI_SHADER_SIGNATURE_RASTERIZED_STREAM | SCORPI_SHADER_SIGNATURE_RASTERIZED_STREAM_MASK;
  if ((metadata->flags & ~allowed) ||
      ((metadata->flags & SCORPI_SHADER_SIGNATURE_RASTERIZED_STREAM_MASK) &&
       !(metadata->flags & SCORPI_SHADER_SIGNATURE_RASTERIZED_STREAM)) ||
      ((metadata->flags & SCORPI_SHADER_SIGNATURE_STREAM_OUTPUT_ONLY) &&
       (metadata->flags & SCORPI_SHADER_SIGNATURE_RASTERIZED_STREAM)))
    throw std::runtime_error("unsupported signature metadata flags");
  storage->flags = metadata->flags;
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
  const wchar_t *options = nullptr;
  std::wstring rasterizedOption;
  if (signatures.flags & SCORPI_SHADER_SIGNATURE_RASTERIZED_STREAM) {
    if ((token_stream[0] >> 16) != 2)
      throw std::runtime_error("rasterized stream signatures require a geometry shader");
    const uint32_t stream = (signatures.flags & SCORPI_SHADER_SIGNATURE_RASTERIZED_STREAM_MASK) >>
        SCORPI_SHADER_SIGNATURE_RASTERIZED_STREAM_SHIFT;
    rasterizedOption = L"-scorpi-rasterized-stream=" + std::to_wstring(stream);
    options = rasterizedOption.c_str();
  }
  if (signatures.flags & SCORPI_SHADER_SIGNATURE_STREAM_OUTPUT_ONLY) {
    if ((token_stream[0] >> 16) != 2)
      throw std::runtime_error("stream-output-only signatures require a geometry shader");
    options = L"-scorpi-stream-output-only";
  }
  const HRESULT status = converter.ConvertInDriverToContainer(
      token_stream, signatures.input, signatures.input_count, signatures.output,
      signatures.output_count, signatures.patch, signatures.patch_count, options,
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

extern "C" int ScorpiStripShaderDebugInfo(const void *code, size_t size,
    void **binary, size_t *binary_size, char *error, size_t error_size) {
  SetError(error, error_size, "");
  if (!binary || !binary_size) return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  *binary = nullptr; *binary_size = 0;
  try {
    using namespace llvm;
    using namespace hlsl;
    DxcMallocScope malloc_scope;
    auto normalized = ConvertToDxilBinary(code, size, nullptr, 0);
    auto *container = reinterpret_cast<const DxilContainerHeader *>(normalized.data());
    auto *program = GetDxilProgramHeader(container, DFCC_DXIL);
    if (!program) throw std::runtime_error("missing DXIL program");
    LLVMContext context;
    auto source = MemoryBuffer::getMemBuffer(StringRef(GetDxilBitcodeData(program),
        GetDxilBitcodeSize(program)), "strip-debug", false);
    auto parsed = parseBitcodeFile(source->getMemBufferRef(), context);
    if (!parsed) throw std::runtime_error("invalid DXIL bitcode");
    if (!StripDebugInfo(**parsed)) return SCORPI_DXBC_TO_DXIL_OK;
    auto *dm = DxilModule::TryGetDxilModule(parsed.get().get());
    if (!dm) throw std::runtime_error("missing DXIL module");
    dm->StripDebugRelatedCode();
    CComPtr<AbstractMemoryStream> bitcode, output;
    IFT(CreateMemoryStream(DxcGetThreadMallocNoRef(), &bitcode));
    IFT(CreateMemoryStream(DxcGetThreadMallocNoRef(), &output));
    { raw_stream_ostream stream(bitcode.p); WriteBitcodeToFile(parsed.get().get(), stream); }
    SerializeDxilContainerForModule(dm, bitcode, nullptr, output, StringRef(), SerializeDxilFlags::None);
    auto *result = std::malloc(output->GetPtrSize());
    if (!result) throw std::runtime_error("debug-stripped output allocation failed");
    std::memcpy(result, output->GetPtr(), output->GetPtrSize());
    auto *header = static_cast<DxilContainerHeader *>(result);
    constexpr uint32_t hashOffset = uint32_t(offsetof(DxilContainerHeader, Version));
    ComputeHashRetail(static_cast<const uint8_t *>(result) + hashOffset,
        header->ContainerSizeInBytes - hashOffset, header->Hash.Digest);
    *binary = result; *binary_size = output->GetPtrSize();
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (const std::exception &exception) {
    SetError(error, error_size, exception.what());
    return SCORPI_DXBC_TO_DXIL_CONVERSION_FAILED;
  } catch (...) {
    SetError(error, error_size, "unexpected debug stripping failure");
    return SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR;
  }
}

extern "C" int ScorpiCreateSignedRenderTargetShader(const void *code, size_t size,
    uint32_t signed_targets, void **binary, size_t *binary_size,
    char *error, size_t error_size) {
  SetError(error, error_size, "");
  if (!binary || !binary_size || (signed_targets & ~255u))
    return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  *binary = nullptr; *binary_size = 0;
  if (!signed_targets) return SCORPI_DXBC_TO_DXIL_OK;
  try {
    using namespace llvm;
    using namespace hlsl;
    DxcMallocScope malloc_scope;
    auto normalized = ConvertToDxilBinary(code, size, nullptr, 0);
    auto *container = reinterpret_cast<const DxilContainerHeader *>(normalized.data());
    if (!IsValidDxilContainer(container, normalized.size()))
      throw std::runtime_error("invalid signed-target shader container");
    auto *program = GetDxilProgramHeader(container, DFCC_DXIL);
    if (!program) throw std::runtime_error("missing signed-target DXIL program");
    LLVMContext context;
    auto source = MemoryBuffer::getMemBuffer(StringRef(GetDxilBitcodeData(program),
        GetDxilBitcodeSize(program)), "signed-targets", false);
    auto parsed = parseBitcodeFile(source->getMemBufferRef(), context);
    if (!parsed) throw std::runtime_error("invalid signed-target DXIL bitcode");
    auto *dm = DxilModule::TryGetDxilModule(parsed.get().get());
    if (!dm || !dm->GetShaderModel()->IsPS()) throw std::runtime_error("signed targets require a pixel shader");
    auto selected = [&](const DxilSignatureElement &e, unsigned row) {
      return e.GetKind() == DXIL::SemanticKind::Target &&
          row < e.GetSemanticIndexVec().size() && e.GetSemanticIndexVec()[row] < 8 &&
          (signed_targets & (1u << e.GetSemanticIndexVec()[row]));
    };
    bool changed = false;
    for (const auto &e : dm->GetOutputSignature().GetElements())
      for (unsigned row = 0; row < e->GetRows(); ++row)
        changed |= selected(*e, row) && !(e->GetCompType() == CompType::getI32());
    if (!changed) return SCORPI_DXBC_TO_DXIL_OK;

    DxilSignature original(dm->GetOutputSignature());
    auto &signature = dm->GetOutputSignature(); signature.ClearElements();
    struct Group { unsigned first, count, id; bool signed_type; };
    std::vector<std::vector<Group>> groups(original.GetElements().size());
    for (const auto &e : original.GetElements()) {
      if (e->GetID() >= groups.size()) throw std::runtime_error("invalid output signature ID");
      for (unsigned first = 0; first < e->GetRows();) {
        const auto type = selected(*e, first) ? CompType::getI32() : e->GetCompType();
        unsigned end = first + 1;
        while (end < e->GetRows() &&
            (selected(*e, end) ? CompType::getI32() : e->GetCompType()) == type) ++end;
        auto copy = signature.CreateElement();
        const auto &indices = e->GetSemanticIndexVec();
        if (end > indices.size()) throw std::runtime_error("invalid output semantic rows");
        std::vector<unsigned> subset(indices.begin() + first, indices.begin() + end);
        copy->Initialize(e->GetName(), type, *e->GetInterpolationMode(), end-first,
            e->GetCols(), e->GetStartRow()+int(first), e->GetStartCol(),
            DxilSignatureElement::kUndefinedID, subset);
        copy->SetKind(e->GetKind()); copy->SetUsageMask(e->GetUsageMask());
        groups[e->GetID()].push_back({first,end-first,signature.AppendElement(std::move(copy)),type==CompType::getI32()});
        first = end;
      }
    }
    std::vector<CallInst *> stores;
    for (auto &f : *parsed.get()) for (auto &block : f) for (auto &inst : block)
      if (auto *call = dyn_cast<CallInst>(&inst)) {
        auto *fn = call->getCalledFunction();
        auto *op = call->getNumArgOperands() ? dyn_cast<ConstantInt>(call->getArgOperand(0)) : nullptr;
        if (fn && fn->getName().startswith("dx.op.storeOutput.") && op &&
            op->getZExtValue() == unsigned(DXIL::OpCode::StoreOutput)) stores.push_back(call);
      }
    for (auto *call : stores) {
      auto *id = dyn_cast<ConstantInt>(call->getArgOperand(1));
      if (!id || id->getZExtValue() >= groups.size()) throw std::runtime_error("invalid output store ID");
      const auto &ranges = groups[id->getZExtValue()];
      auto emit = [&](IRBuilder<> &at, const Group &g, Value *row) {
        Value *value = call->getArgOperand(4);
        if (g.signed_type && !value->getType()->isIntegerTy(32)) {
          if (!value->getType()->isFloatTy()) throw std::runtime_error("signed RTV exports require 32-bit registers");
          value = at.CreateBitCast(value, at.getInt32Ty());
        }
        auto *store = at.CreateCall(dm->GetOP()->GetOpFunc(DXIL::OpCode::StoreOutput,value->getType()),
            {call->getArgOperand(0),at.getInt32(g.id),at.CreateSub(row,at.getInt32(g.first)),call->getArgOperand(3),value});
        store->setDebugLoc(call->getDebugLoc());
      };
      auto *row = call->getArgOperand(2);
      if (auto *constant = dyn_cast<ConstantInt>(row)) {
        bool found = false;
        for (const auto &g : ranges) if (constant->getZExtValue() >= g.first &&
            constant->getZExtValue()-g.first < g.count) { IRBuilder<> at(call); emit(at,g,row); found=true; break; }
        if (!found) throw std::runtime_error("invalid output store row");
      } else if (ranges.size() == 1) { IRBuilder<> at(call); emit(at,ranges[0],row); }
      else {
        // A dynamically indexed output array may span different RTV types.
        // Preserve the selected row and issue exactly one typed store.
        auto *before=call->getParent(), *after=before->splitBasicBlock(call,"rt.continue");
        before->getTerminator()->eraseFromParent();
        auto *branch=SwitchInst::Create(row,after,original.GetElement(id->getZExtValue()).GetRows(),before);
        for (const auto &g : ranges) {
          auto *block=BasicBlock::Create(context,"rt.store",before->getParent(),after);
          IRBuilder<> at(block); emit(at,g,row); at.CreateBr(after);
          for (unsigned i=0;i<g.count;++i) branch->addCase(at.getInt32(g.first+i),block);
        }
      }
      call->eraseFromParent();
    }
    if (verifyModule(*parsed.get())) throw std::runtime_error("invalid signed-target lowering IR");
    dm->GetSerializedViewIdState().clear();
    DxilModule::ClearDxilMetadata(*parsed.get()); dm->EmitDxilMetadata();
    CComPtr<AbstractMemoryStream> bitcode, output;
    IFT(CreateMemoryStream(DxcGetThreadMallocNoRef(), &bitcode));
    IFT(CreateMemoryStream(DxcGetThreadMallocNoRef(), &output));
    { raw_stream_ostream stream(bitcode.p); WriteBitcodeToFile(parsed.get().get(), stream); }
    SerializeDxilContainerForModule(dm, bitcode, nullptr, output, StringRef(), SerializeDxilFlags::None);
    auto *result=std::malloc(output->GetPtrSize());
    if (!result) throw std::runtime_error("signed-target output allocation failed");
    std::memcpy(result,output->GetPtr(),output->GetPtrSize());
    auto *header=static_cast<DxilContainerHeader *>(result);
    constexpr uint32_t hashOffset=uint32_t(offsetof(DxilContainerHeader,Version));
    ComputeHashRetail(static_cast<const uint8_t *>(result)+hashOffset,
        header->ContainerSizeInBytes-hashOffset,header->Hash.Digest);
    *binary=result; *binary_size=output->GetPtrSize();
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (const std::exception &exception) {
    SetError(error,error_size,exception.what()); return SCORPI_DXBC_TO_DXIL_CONVERSION_FAILED;
  } catch (...) {
    SetError(error,error_size,"unexpected signed-target conversion failure");return SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR;
  }
}

extern "C" int ScorpiInspectOutputCoverage(const void *code, size_t size,
    const void *metadata, size_t metadata_size, int *writes_coverage) {
  if (!code || !size || !writes_coverage)
    return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  *writes_coverage = 0;
  try {
    DxcMallocScope malloc_scope;
    auto binary = ConvertToDxilBinary(code, size, metadata, metadata_size);
    auto *container = reinterpret_cast<const hlsl::DxilContainerHeader *>(binary.data());
    if (!hlsl::IsValidDxilContainer(container, binary.size()))
      return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
    const auto *part = hlsl::GetDxilPartByType(container, hlsl::DFCC_OutputSignature);
    if (!part) return SCORPI_DXBC_TO_DXIL_OK;
    if (part->PartSize < sizeof(hlsl::DxilProgramSignature))
      return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
    hlsl::DxilProgramSignature signature;
    const auto *data = hlsl::GetDxilPartData(part);
    std::memcpy(&signature, data, sizeof(signature));
    if (signature.ParamOffset < sizeof(signature) || signature.ParamOffset > part->PartSize ||
        signature.ParamCount > (part->PartSize - signature.ParamOffset) /
            sizeof(hlsl::DxilProgramSignatureElement))
      return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
    for (uint32_t i = 0; i < signature.ParamCount; ++i) {
      hlsl::DxilProgramSignatureElement element;
      std::memcpy(&element, data + signature.ParamOffset + i * sizeof(element), sizeof(element));
      if (element.SystemValue == hlsl::DxilProgramSigSemantic::Coverage) {
        *writes_coverage = 1;
        break;
      }
    }
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (...) { return SCORPI_DXBC_TO_DXIL_INVALID_SHADER; }
}

namespace {
uint32_t InspectDxbcRasterizer(const void *code, size_t size) {
  // Inspect declarations/opcodes with the converter's own token parser. Do not
  // convert every D3D11 shader into DXIL merely to choose its variant key.
  const auto *tokens = ValidateShaderTokenStream(code, size);
  D3D10ShaderBinary::CShaderCodeParser parser(tokens);
  D3D10ShaderBinary::CInstruction instruction;
  uint32_t requirements = 0;
  while (!parser.EndOfShader()) {
    const auto offset = parser.CurrentTokenOffset();
    const auto opcode = DECODE_D3D10_SB_OPCODE_TYPE(tokens[offset]);
    uint32_t length = DECODE_D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH(tokens[offset]);
    if (opcode == D3D10_SB_OPCODE_CUSTOMDATA ||
        ((opcode == D3D11_SB_OPCODE_DCL_INTERFACE ||
          opcode == D3D11_SB_OPCODE_DCL_FUNCTION_TABLE) &&
         DECODE_IS_D3D10_SB_OPCODE_EXTENDED(tokens[offset]))) {
      if (tokens[1] - offset < 2) throw std::runtime_error("truncated DXBC instruction");
      length = tokens[offset + 1];
    }
    if (!length || length > tokens[1] - offset)
      throw std::runtime_error("invalid DXBC instruction length");
    switch (opcode) {
    case D3D10_1_SB_OPCODE_SAMPLE_INFO:
    case D3D10_SB_OPCODE_DCL_INPUT:
    case D3D10_SB_OPCODE_DCL_INPUT_PS:
    case D3D10_SB_OPCODE_DCL_INPUT_PS_SIV:
    case D3D10_SB_OPCODE_DCL_INPUT_PS_SGV:
      if (length < 2) throw std::runtime_error("truncated DXBC rasterizer operand");
      parser.ParseInstruction(&instruction);
      if (parser.CurrentTokenOffset() != offset + length)
        throw std::runtime_error("invalid DXBC rasterizer instruction");
      if (opcode == D3D10_1_SB_OPCODE_SAMPLE_INFO) {
        if (instruction.m_Operands[1].OperandType() == D3D10_SB_OPERAND_TYPE_RASTERIZER)
          requirements |= SCORPI_RASTERIZER_SAMPLE_COUNT;
      } else {
        if (instruction.m_Operands[0].OperandType() == D3D11_SB_OPERAND_TYPE_INPUT_COVERAGE_MASK)
          requirements |= SCORPI_RASTERIZER_INPUT_COVERAGE;
        if (opcode == D3D10_SB_OPCODE_DCL_INPUT_PS || opcode == D3D10_SB_OPCODE_DCL_INPUT_PS_SIV) {
          const auto mode = DECODE_D3D10_SB_INPUT_INTERPOLATION_MODE(tokens[offset]);
          if (mode == D3D10_SB_INTERPOLATION_LINEAR_CENTROID ||
              mode == D3D10_SB_INTERPOLATION_LINEAR_NOPERSPECTIVE_CENTROID)
            requirements |= SCORPI_RASTERIZER_CENTROID;
        }
      }
      break;
    case D3D11_SB_OPCODE_EVAL_CENTROID:
      requirements |= SCORPI_RASTERIZER_CENTROID;
      break;
    default: break;
    }
    parser.SetCurrentTokenOffset(offset + length);
  }
  return requirements;
}

std::vector<llvm::CallInst *> RasterizerSampleCountCalls(llvm::Module &module) {
  std::vector<llvm::CallInst *> calls;
  for (auto &function : module) {
    if (!function.getName().startswith("dx.op.renderTargetGetSampleCount"))
      continue;
    for (auto *user : function.users()) {
      auto *call = llvm::dyn_cast<llvm::CallInst>(user);
      auto *opcode = call && call->getNumArgOperands() == 1
          ? llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0)) : nullptr;
      if (!call || call->getCalledFunction() != &function || !opcode ||
          opcode->getZExtValue() != unsigned(hlsl::DXIL::OpCode::RenderTargetGetSampleCount) ||
          !call->getType()->isIntegerTy(32))
        throw std::runtime_error("invalid rasterizer sample-count instruction");
      calls.push_back(call);
    }
  }
  return calls;
}

bool LowerPixelRasterizer(hlsl::DxilModule &dm, uint32_t samples,
    uint32_t forced_samples, uint32_t sample_mask, bool &native_centroid) {
  using namespace llvm;
  using namespace hlsl;
  if (!dm.GetShaderModel()->IsPS()) return false;
  const uint32_t all = samples == 32 ? UINT32_MAX : (1u << samples) - 1;
  // Native centroid already handles the full-mask case. Forced rasterization
  // also ignores the output sample mask when choosing its centroid location.
  const bool correct_centroid = !forced_samples && samples > 1 && (sample_mask & all) != all;
  std::vector<unsigned> centroid_ids;
  for (const auto &element : dm.GetInputSignature().GetElements())
    if (element->GetInterpolationMode()->IsAnyCentroid())
      centroid_ids.push_back(element->GetID());
  struct Work { CallInst *call; bool coverage; bool load; };
  std::vector<Work> work;
  for (auto &function : *dm.GetModule()) for (auto &block : function) for (auto &inst : block) {
    auto *call = dyn_cast<CallInst>(&inst);
    if (!call || !call->getCalledFunction() || !call->getNumArgOperands() ||
        !call->getCalledFunction()->getName().startswith("dx.op.")) continue;
    auto *opcode = dyn_cast<ConstantInt>(call->getArgOperand(0));
    if (!opcode) continue;
    const auto op = static_cast<DXIL::OpCode>(opcode->getZExtValue());
    bool centroid = op == DXIL::OpCode::EvalCentroid;
    if (op == DXIL::OpCode::LoadInput && call->getNumArgOperands() >= 4)
      if (auto *id = dyn_cast<ConstantInt>(call->getArgOperand(1)))
        centroid = std::find(centroid_ids.begin(), centroid_ids.end(), id->getZExtValue()) != centroid_ids.end();
    const bool coverage = op == DXIL::OpCode::Coverage && forced_samples == 1;
    if ((centroid && correct_centroid) || coverage)
      work.push_back({call, coverage, op == DXIL::OpCode::LoadInput});
  }
  bool changed = false;
  auto *ops = dm.GetOP();
  for (const auto &item : work) {
    auto *call = item.call;
    IRBuilder<> at(call);
    if (item.coverage) {
      // Coincident native samples broadcast a single forced rasterizer sample.
      // D3D input coverage still contains exactly one logical coverage bit.
      auto *raw = at.CreateCall(call->getCalledFunction(), {call->getArgOperand(0)});
      auto *logical = at.CreateZExt(at.CreateICmpNE(raw, at.getInt32(0)), at.getInt32Ty());
      call->replaceAllUsesWith(logical);
      call->eraseFromParent();
      changed = true;
      continue;
    }
    // MSC masks dx.op.coverage before exposing it to the shader. Preserve
    // native masking/discard behavior and select the centroid later in AIR,
    // where the unmasked sample-mask input remains available. An explicit
    // evaluation makes an ordinary centroid input a native interpolant.
    native_centroid = true;
    if (item.load) {
      auto *value = at.CreateCall(ops->GetOpFunc(DXIL::OpCode::EvalCentroid, call->getType()),
          {at.getInt32(unsigned(DXIL::OpCode::EvalCentroid)), call->getArgOperand(1),
           call->getArgOperand(2), call->getArgOperand(3)});
      call->replaceAllUsesWith(value);
      call->eraseFromParent();
      changed = true;
    }
  }
  return changed;
}
}

extern "C" int ScorpiInspectRasterizerRequirements(const void *code, size_t size,
    const void *metadata, size_t metadata_size, uint32_t *requirements) {
  if (!code || !size || !requirements || (!metadata && metadata_size))
    return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  *requirements = 0;
  try {
    DxcMallocScope malloc_scope;
    const hlsl::DxilProgramHeader *program = nullptr;
    if (const auto *container = hlsl::IsDxilContainerLike(code, size)) {
      if (!hlsl::IsValidDxilContainer(container, size))
        return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
      program = hlsl::GetDxilProgramHeader(container, hlsl::DFCC_DXIL);
      if (!program) {
        auto *part = hlsl::GetDxilPartByType(container,
            static_cast<hlsl::DxilFourCC>(hlsl::DXBC_GenericShaderEx));
        if (!part) part = hlsl::GetDxilPartByType(container,
            static_cast<hlsl::DxilFourCC>(hlsl::DXBC_GenericShader));
        if (!part) return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
        *requirements = InspectDxbcRasterizer(hlsl::GetDxilPartData(part), part->PartSize);
        return SCORPI_DXBC_TO_DXIL_OK;
      }
    } else if (IsDxilProgram(code, size)) {
      program = static_cast<const hlsl::DxilProgramHeader *>(code);
    } else {
      *requirements = InspectDxbcRasterizer(code, size);
      return SCORPI_DXBC_TO_DXIL_OK;
    }
    llvm::LLVMContext context;
    auto source = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(
        hlsl::GetDxilBitcodeData(program), hlsl::GetDxilBitcodeSize(program)),
        "sample-count-inspection", false);
    // Load signatures/symbols, not instruction bodies, for ordinary DXIL.
    auto parsed = llvm::getLazyBitcodeModule(std::move(source), context);
    if (!parsed) return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
    bool declared = false;
    for (auto &function : *parsed.get())
      declared |= function.getName().startswith("dx.op.renderTargetGetSampleCount") ||
          function.getName().startswith("dx.op.evalCentroid.") ||
          function.getName().startswith("dx.op.coverage.");
    const bool pixel = hlsl::GetVersionShaderType(program->ProgramVersion) ==
        hlsl::DXIL::ShaderKind::Pixel;
    if (!declared && !pixel) return SCORPI_DXBC_TO_DXIL_OK;
    if (declared) {
      if (parsed.get()->materializeAllPermanently())
        return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
      if (!RasterizerSampleCountCalls(*parsed.get()).empty())
        *requirements |= SCORPI_RASTERIZER_SAMPLE_COUNT;
    }
    if (pixel) {
      // Decoding all resource metadata can fail on incomplete DDI metadata and
      // is unrelated to interpolation. Load only the pixel entry signature.
      hlsl::DxilMDHelper helper(parsed.get().get(),
          std::unique_ptr<hlsl::DxilExtraPropertyHelper>(new hlsl::DxilExtraPropertyHelper(parsed.get().get())));
      const auto *entries = helper.GetDxilEntryPoints();
      if (!entries || entries->getNumOperands() != 1)
        return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
      llvm::Function *entry = nullptr; std::string name;
      const llvm::MDOperand *signatures = nullptr, *resources = nullptr, *properties = nullptr;
      helper.GetDxilEntryPoint(entries->getOperand(0), entry, name, signatures, resources, properties);
      hlsl::DxilEntrySignature signature(hlsl::DXIL::ShaderKind::Pixel, true);
      helper.LoadDxilSignatures(*signatures, signature);
      for (const auto &element : signature.InputSignature.GetElements())
        if (element->GetInterpolationMode()->IsAnyCentroid())
          *requirements |= SCORPI_RASTERIZER_CENTROID;
      for (auto &function : *parsed.get()) {
        if (function.use_empty()) continue;
        if (function.getName().startswith("dx.op.evalCentroid."))
          *requirements |= SCORPI_RASTERIZER_CENTROID;
        if (function.getName().startswith("dx.op.coverage."))
          *requirements |= SCORPI_RASTERIZER_INPUT_COVERAGE;
      }
    }
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (...) { return SCORPI_DXBC_TO_DXIL_INVALID_SHADER; }
}

extern "C" int ScorpiSpecializeRasterizerShader(const void *code, size_t size,
    uint32_t samples, uint32_t forced_samples, uint32_t sample_mask,
    void **binary, size_t *binary_size, uint32_t *requirements,
    char *error, size_t error_size) {
  SetError(error, error_size, "");
  if (!code || !size || !binary || !binary_size)
    return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  *binary = nullptr; *binary_size = 0;
  if (requirements) *requirements = 0;
  if (!samples || samples > 32 || (samples & (samples - 1)))
    return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  try {
    using namespace llvm;
    using namespace hlsl;
    DxcMallocScope malloc_scope;
    auto normalized = ConvertToDxilBinary(code, size, nullptr, 0);
    auto *container = reinterpret_cast<const DxilContainerHeader *>(normalized.data());
    if (!IsValidDxilContainer(container, normalized.size()))
      throw std::runtime_error("invalid sample-count shader container");
    auto *program = GetDxilProgramHeader(container, DFCC_DXIL);
    if (!program) throw std::runtime_error("missing sample-count DXIL program");
    LLVMContext context;
    auto source = MemoryBuffer::getMemBuffer(StringRef(GetDxilBitcodeData(program),
        GetDxilBitcodeSize(program)), "sample-count-specialization", false);
    auto parsed = parseBitcodeFile(source->getMemBufferRef(), context);
    if (!parsed) throw std::runtime_error("invalid sample-count bitcode");
    auto calls = RasterizerSampleCountCalls(*parsed.get());
    auto *dm = DxilModule::TryGetDxilModule(parsed.get().get());
    if (!dm) throw std::runtime_error("missing sample-count DXIL module");
    for (auto *call : calls) {
      call->replaceAllUsesWith(ConstantInt::get(call->getType(), samples));
      call->eraseFromParent();
    }
    bool native_centroid = false;
    const bool pixel_changed = LowerPixelRasterizer(*dm, samples, forced_samples, sample_mask, native_centroid);
    if (requirements && native_centroid) *requirements |= SCORPI_RASTERIZER_CENTROID;
    if (calls.empty() && !pixel_changed) return SCORPI_DXBC_TO_DXIL_OK;
    for (auto function = parsed.get()->begin(); function != parsed.get()->end();) {
      auto &candidate = *function++;
      if (candidate.isDeclaration() && candidate.getName().startswith("dx.op.") &&
          candidate.use_empty()) {
        dm->RemoveFunction(&candidate);
        candidate.eraseFromParent();
      }
    }
    if (verifyModule(*parsed.get())) throw std::runtime_error("invalid sample-count specialization IR");
    dm->CollectShaderFlagsForModule();
    dm->GetSerializedViewIdState().clear();
    DxilModule::ClearDxilMetadata(*parsed.get()); dm->EmitDxilMetadata();
    CComPtr<AbstractMemoryStream> bitcode, output;
    IFT(CreateMemoryStream(DxcGetThreadMallocNoRef(), &bitcode));
    IFT(CreateMemoryStream(DxcGetThreadMallocNoRef(), &output));
    { raw_stream_ostream stream(bitcode.p); WriteBitcodeToFile(parsed.get().get(), stream); }
    SerializeDxilContainerForModule(dm, bitcode, nullptr, output, StringRef(), SerializeDxilFlags::None);
    auto *result = std::malloc(output->GetPtrSize());
    if (!result) throw std::runtime_error("sample-count output allocation failed");
    std::memcpy(result, output->GetPtr(), output->GetPtrSize());
    auto *header = static_cast<DxilContainerHeader *>(result);
    constexpr uint32_t offset = uint32_t(offsetof(DxilContainerHeader, Version));
    ComputeHashRetail(static_cast<const uint8_t *>(result) + offset,
        header->ContainerSizeInBytes - offset, header->Hash.Digest);
    *binary = result; *binary_size = output->GetPtrSize();
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (const std::exception &exception) {
    SetError(error, error_size, exception.what()); return SCORPI_DXBC_TO_DXIL_CONVERSION_FAILED;
  } catch (...) {
    SetError(error, error_size, "unexpected sample-count specialization failure");
    return SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR;
  }
}

extern "C" int ScorpiCreateAlphaToCoverageShader(const void *code, size_t size,
    uint32_t samples, void **binary, size_t *binary_size,
    char *error, size_t error_size) {
  SetError(error, error_size, "");
  if (!binary || !binary_size) return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  *binary = nullptr; *binary_size = 0;
  try {
    using namespace llvm;
    using namespace hlsl;
    DxcMallocScope malloc_scope;
    auto normalized = ConvertToDxilBinary(code, size, nullptr, 0);
    auto *container = reinterpret_cast<const DxilContainerHeader *>(normalized.data());
    if (!IsValidDxilContainer(container, normalized.size()))
      throw std::runtime_error("invalid alpha-to-coverage shader container");
    auto *program = GetDxilProgramHeader(container, DFCC_DXIL);
    if (!program) throw std::runtime_error("missing alpha-to-coverage DXIL program");
    LLVMContext context;
    auto source = MemoryBuffer::getMemBuffer(StringRef(GetDxilBitcodeData(program),
        GetDxilBitcodeSize(program)), "alpha-to-coverage", false);
    auto parsed = parseBitcodeFile(source->getMemBufferRef(), context);
    if (!parsed) throw std::runtime_error("invalid alpha-to-coverage bitcode");
    auto *dm = DxilModule::TryGetDxilModule(parsed.get().get());
    bool changed = false;
    std::string why;
    if (!dm || !scorpi::LowerAlphaToCoverage(*dm, samples, why, changed))
      throw std::runtime_error(why.empty() ? "invalid alpha-to-coverage lowering" : why);
    if (!changed) return SCORPI_DXBC_TO_DXIL_OK;
    dm->GetSerializedViewIdState().clear();
    DxilModule::ClearDxilMetadata(*parsed.get()); dm->EmitDxilMetadata();
    CComPtr<AbstractMemoryStream> bitcode, output;
    IFT(CreateMemoryStream(DxcGetThreadMallocNoRef(), &bitcode));
    IFT(CreateMemoryStream(DxcGetThreadMallocNoRef(), &output));
    { raw_stream_ostream stream(bitcode.p); WriteBitcodeToFile(parsed.get().get(), stream); }
    SerializeDxilContainerForModule(dm, bitcode, nullptr, output, StringRef(), SerializeDxilFlags::None);
    auto *result = std::malloc(output->GetPtrSize());
    if (!result) throw std::runtime_error("alpha-to-coverage output allocation failed");
    std::memcpy(result, output->GetPtr(), output->GetPtrSize());
    auto *header = static_cast<DxilContainerHeader *>(result);
    constexpr uint32_t offset = uint32_t(offsetof(DxilContainerHeader, Version));
    ComputeHashRetail(static_cast<const uint8_t *>(result) + offset,
        header->ContainerSizeInBytes - offset, header->Hash.Digest);
    *binary = result; *binary_size = output->GetPtrSize();
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (const std::exception &exception) {
    SetError(error, error_size, exception.what()); return SCORPI_DXBC_TO_DXIL_CONVERSION_FAILED;
  } catch (...) {
    SetError(error, error_size, "unexpected alpha-to-coverage conversion failure");
    return SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR;
  }
}

static int CreateRasterReplayShader(
    const void *code, size_t size, uint32_t register_space,
    uint32_t sample_count, uint32_t sample_mask, void **binary,
    size_t *binary_size, char *error_message, size_t error_message_size, bool vertex, bool generated = false, bool geometry_link = false, bool attachments = false, bool alpha_to_coverage = false, uint32_t *requirements = nullptr, bool pixel_draw_ids = false) {
  SetError(error_message, error_message_size, "");
  if (!binary || !binary_size) return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  *binary = nullptr;
  *binary_size = 0;
  try {
    DxcMallocScope malloc_scope;
    auto normalized = ConvertToDxilBinary(code, size, nullptr, 0);
    auto *container = reinterpret_cast<const hlsl::DxilContainerHeader *>(normalized.data());
    if (!hlsl::IsValidDxilContainer(container, normalized.size()))
      throw std::runtime_error("invalid replay shader container");
    auto *program = hlsl::GetDxilProgramHeader(container, hlsl::DFCC_DXIL);
    if (!program) throw std::runtime_error("replay shader has no DXIL program");
    llvm::LLVMContext context;
    auto buffer = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(hlsl::GetDxilBitcodeData(program),
                        hlsl::GetDxilBitcodeSize(program)), "raster-replay", false);
    auto parsed = llvm::parseBitcodeFile(buffer->getMemBufferRef(), context);
    if (!parsed) throw std::runtime_error("invalid replay shader bitcode");
    auto *dxil = hlsl::DxilModule::TryGetDxilModule(parsed.get().get());
    std::string error;
    if (!dxil || !(geometry_link ? scorpi::LowerGeometryPrimitiveId(*dxil, error)
        : (vertex || pixel_draw_ids) ? scorpi::PrepareRasterReplayDrawIds(*dxil, register_space, error)
        : scorpi::PrepareRasterReplayPixelShader(*dxil, register_space, sample_count, sample_mask, error, generated, attachments, alpha_to_coverage, requirements)))
      throw std::runtime_error(error.empty() ? "missing DXIL module" : error);
    // The replay transforms change signatures. Serialized ViewID state still
    // describes the source signature and cannot be reused for the new PSV.
    dxil->GetSerializedViewIdState().clear();
    hlsl::DxilModule::ClearDxilMetadata(*parsed.get());
    dxil->EmitDxilMetadata();
    CComPtr<hlsl::AbstractMemoryStream> bitcode, output;
    IFT(hlsl::CreateMemoryStream(DxcGetThreadMallocNoRef(), &bitcode));
    IFT(hlsl::CreateMemoryStream(DxcGetThreadMallocNoRef(), &output));
    {
      raw_stream_ostream stream(bitcode.p);
      llvm::WriteBitcodeToFile(parsed.get().get(), stream);
    }
    hlsl::SerializeDxilContainerForModule(dxil, bitcode, nullptr, output,
        llvm::StringRef(), hlsl::SerializeDxilFlags::None);
    void *result = std::malloc(output->GetPtrSize());
    if (!result) throw std::runtime_error("failed to allocate replay shader");
    std::memcpy(result, output->GetPtr(), output->GetPtrSize());
    auto *header = static_cast<hlsl::DxilContainerHeader *>(result);
    constexpr uint32_t hashOffset =
        static_cast<uint32_t>(offsetof(hlsl::DxilContainerHeader, Version));
    ComputeHashRetail(static_cast<const uint8_t *>(result) + hashOffset,
        header->ContainerSizeInBytes - hashOffset, header->Hash.Digest);
    *binary = result;
    *binary_size = output->GetPtrSize();
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (const std::exception &e) {
    SetError(error_message, error_message_size, e.what());
    return SCORPI_DXBC_TO_DXIL_CONVERSION_FAILED;
  } catch (...) {
    SetError(error_message, error_message_size, "unexpected raster replay conversion failure");
    return SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR;
  }
}

extern "C" int ScorpiCreateRasterReplayPixelShader(
    const void *code, size_t size, uint32_t space, uint32_t samples,
    uint32_t mask, void **binary, size_t *binary_size, char *error, size_t error_size) {
  return CreateRasterReplayShader(code, size, space, samples, mask, binary,
      binary_size, error, error_size, false);
}

extern "C" int ScorpiCreateRasterReplayVertexShader(
    const void *code, size_t size, uint32_t space, void **binary,
    size_t *binary_size, char *error, size_t error_size) {
  return CreateRasterReplayShader(code, size, space, 0, 0, binary,
      binary_size, error, error_size, true);
}

extern "C" int ScorpiCreateRasterReplayPixelDrawIds(
    const void *code, size_t size, uint32_t space, void **binary,
    size_t *binary_size, char *error, size_t error_size) {
  return CreateRasterReplayShader(code, size, space, 0, 0, binary,
      binary_size, error, error_size, false, false, false, false, false, nullptr, true);
}

extern "C" int ScorpiCreateMsaa8PixelShader(
    const void *code, size_t size, uint32_t space, uint32_t mask, int generated,
    void **binary, size_t *binary_size, char *error, size_t error_size) {
  return CreateRasterReplayShader(code, size, space, 8, mask, binary,
      binary_size, error, error_size, false, generated != 0, false, true);
}

extern "C" int ScorpiCreateMsaa8PixelShaderEx(
    const void *code, size_t size, uint32_t space, uint32_t mask, int generated,
    uint32_t options, void **binary, size_t *binary_size, char *error, size_t error_size) {
  return ScorpiCreateMsaa8PixelShaderVariant(code, size, space, mask, generated,
      options, binary, binary_size, error, error_size, nullptr);
}

extern "C" int ScorpiCreateMsaa8PixelShaderVariant(
    const void *code, size_t size, uint32_t space, uint32_t mask, int generated,
    uint32_t options, void **binary, size_t *binary_size, char *error, size_t error_size, uint32_t *requirements) {
  if (requirements) *requirements = SCORPI_MSAA8_REPLAY_COVERAGE;
  if (options & ~SCORPI_MSAA8_ALPHA_TO_COVERAGE)
    return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  return CreateRasterReplayShader(code, size, space, 8, mask, binary,
      binary_size, error, error_size, false, generated != 0, false, true,
      (options & SCORPI_MSAA8_ALPHA_TO_COVERAGE) != 0, requirements);
}

extern "C" int ScorpiCreateGeneratedRasterReplayPixelShader(
    const void *code, size_t size, uint32_t space, uint32_t samples,
    uint32_t mask, void **binary, size_t *binary_size, char *error, size_t error_size) {
  return CreateRasterReplayShader(code, size, space, samples, mask, binary,
      binary_size, error, error_size, false, true);
}

extern "C" int ScorpiLowerGeometryPrimitiveId(
    const void *code, size_t size, void **binary, size_t *binary_size, char *error, size_t error_size) {
  return CreateRasterReplayShader(code, size, 0, 0, 0, binary,
      binary_size, error, error_size, false, false, true);
}
namespace {
bool DynamicMeshOutput(const llvm::Instruction &instruction) {
  const auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
  if (!call || !call->getCalledFunction() || call->getNumArgOperands() != 6) return false;
  const auto name = call->getCalledFunction()->getName();
  return (name.startswith("dx.op.storeVertexOutput.") || name.startswith("dx.op.storePrimitiveOutput.")) &&
      !llvm::isa<llvm::ConstantInt>(call->getArgOperand(2));
}

bool HasDynamicMeshOutputs(const llvm::Module &module) {
  for (const auto &function : module) for (const auto &block : function)
    for (const auto &instruction : block) if (DynamicMeshOutput(instruction)) return true;
  return false;
}

class MeshOutputUnroll final : public llvm::LoopPass {
public:
  static char ID;
  MeshOutputUnroll() : LoopPass(ID) {}
  bool runOnLoop(llvm::Loop *loop, llvm::LPPassManager &manager) override {
    bool needed = false;
    size_t instructions = 0;
    for (const auto *block : loop->getBlocks()) for (const auto &instruction : *block) {
      needed |= DynamicMeshOutput(instruction);
      ++instructions;
    }
    if (!needed) return false;
    auto &evolution = getAnalysis<llvm::ScalarEvolution>();
    const unsigned trips = evolution.getSmallConstantTripCount(loop);
    // Bound compiler/code-size growth. Unresolved rows are rejected before MSC.
    if (!trips || instructions > 4096 / trips) return false;
    auto &loops = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();
    auto &assumptions = getAnalysis<llvm::AssumptionCacheTracker>().getAssumptionCache(*loop->getHeader()->getParent());
    return llvm::UnrollLoop(loop, trips, trips, false, false, 1,
        &loops, this, &manager, &assumptions);
  }
  void getAnalysisUsage(llvm::AnalysisUsage &usage) const override {
    usage.addRequired<llvm::LoopInfoWrapperPass>();
    usage.addRequired<llvm::ScalarEvolution>();
    usage.addRequired<llvm::AssumptionCacheTracker>();
    usage.addRequired<llvm::DominatorTreeWrapperPass>();
    usage.addRequiredID(llvm::LoopSimplifyID);
    usage.addRequiredID(llvm::LCSSAID);
    usage.addPreserved<llvm::LoopInfoWrapperPass>();
    usage.addPreserved<llvm::ScalarEvolution>();
    usage.addPreserved<llvm::DominatorTreeWrapperPass>();
    usage.addPreservedID(llvm::LoopSimplifyID);
    usage.addPreservedID(llvm::LCSSAID);
  }
};
char MeshOutputUnroll::ID;
}

extern "C" int ScorpiInspectMeshOutputRows(const void *code, size_t size, int *required) {
  if (!required) return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  *required = 0;
  try {
    DxcMallocScope malloc_scope;
    auto normalized = ConvertToDxilBinary(code, size, nullptr, 0);
    auto *container = reinterpret_cast<const hlsl::DxilContainerHeader *>(normalized.data());
    if (!hlsl::IsValidDxilContainer(container, normalized.size())) return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
    auto *program = hlsl::GetDxilProgramHeader(container, hlsl::DFCC_DXIL);
    if (!program) return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
    llvm::LLVMContext context;
    auto source = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(
        hlsl::GetDxilBitcodeData(program), hlsl::GetDxilBitcodeSize(program)), "mesh-output-inspect", false);
    auto parsed = llvm::parseBitcodeFile(source->getMemBufferRef(), context);
    if (!parsed) return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
    auto *dm = hlsl::DxilModule::TryGetDxilModule(parsed.get().get());
    if (!dm) return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
    *required = dm->GetShaderModel()->IsMS() && HasDynamicMeshOutputs(*parsed.get());
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (...) { return SCORPI_DXBC_TO_DXIL_INVALID_SHADER; }
}

static int InspectShaderLinkage(
    const void *code, size_t size, const void *metadata, size_t metadata_size,
    ScorpiShaderLinkageElement **elements, size_t *count,
    char *error, size_t error_size, bool includeSystem) {
  SetError(error, error_size, "");
  if (!code || !size || !elements || !count)
    return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  *elements = nullptr; *count = 0;
  try {
    DxcMallocScope malloc_scope;
    auto normalized = ConvertToDxilBinary(code, size, metadata, metadata_size);
    auto *container = reinterpret_cast<const hlsl::DxilContainerHeader *>(normalized.data());
    if (!hlsl::IsValidDxilContainer(container, normalized.size())) throw std::runtime_error("invalid linkage container");
    auto *program = hlsl::GetDxilProgramHeader(container, hlsl::DFCC_DXIL);
    if (!program) throw std::runtime_error("missing linkage program");
    llvm::LLVMContext context;
    auto source = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(
        hlsl::GetDxilBitcodeData(program), hlsl::GetDxilBitcodeSize(program)), "rasterizer-linkage", false);
    auto parsed = llvm::getLazyBitcodeModule(std::move(source), context);
    if (!parsed) throw std::runtime_error("invalid linkage bitcode");
    // Resource metadata and instruction bodies are unrelated to the interface.
    hlsl::DxilMDHelper helper(parsed.get().get(),
        std::unique_ptr<hlsl::DxilExtraPropertyHelper>(new hlsl::DxilExtraPropertyHelper(parsed.get().get())));
    const auto *entries = helper.GetDxilEntryPoints();
    if (!entries || entries->getNumOperands() != 1) throw std::runtime_error("expected one linkage entry");
    llvm::Function *entry = nullptr; std::string name;
    const llvm::MDOperand *signatures = nullptr, *resources = nullptr, *properties = nullptr;
    helper.GetDxilEntryPoint(entries->getOperand(0), entry, name, signatures, resources, properties);
    hlsl::ShaderFlags flags;
    if (properties) if (auto *node = llvm::dyn_cast_or_null<llvm::MDNode>(properties->get()))
      for (unsigned i = 0; i + 1 < node->getNumOperands(); i += 2) {
        auto *tag = llvm::mdconst::dyn_extract<llvm::ConstantInt>(node->getOperand(i));
        auto *value = llvm::mdconst::dyn_extract<llvm::ConstantInt>(node->getOperand(i+1));
        if (tag && value && tag->getZExtValue() == hlsl::DxilMDHelper::kDxilShaderFlagsTag)
          flags.SetShaderFlagsRaw(value->getZExtValue());
      }
    const auto kind = hlsl::GetVersionShaderType(program->ProgramVersion);
    hlsl::DxilEntrySignature signature(kind, !flags.GetUseNativeLowPrecision());
    helper.LoadDxilSignatures(*signatures, signature);
    std::vector<ScorpiShaderLinkageElement> linkage;
    std::vector<std::string> names;
    auto reflect = [&](const hlsl::DxilSignature &sig, bool output) {
      for (const auto &element : sig.GetElements()) {
        if (element->GetOutputStream() || (!includeSystem &&
            element->GetKind() != hlsl::DXIL::SemanticKind::Arbitrary)) continue;
        auto type = element->GetCompType();
        if (flags.GetLowPrecisionPresent() && !flags.GetUseNativeLowPrecision() &&
            (type == hlsl::CompType::getF16() || type == hlsl::CompType::getI16()))
          type = type.GetBaseCompType();
        uint32_t selected = type == hlsl::CompType::getF32() ? 3 :
            type == hlsl::CompType::getI32() ? 2 : type == hlsl::CompType::getU32() ? 1 : 0;
        const bool flat = !output && kind == hlsl::DXIL::ShaderKind::Pixel && selected == 3 &&
            element->GetInterpolationMode()->IsConstant();
        if (flat) selected = 1;
        if (!selected) continue;
        for (unsigned row = 0; row < element->GetRows(); ++row) {
          int reg = element->GetStartRow() + int(row);
          if (reg < 0 || reg >= 32) continue;
          names.emplace_back(element->GetName());
          linkage.push_back({nullptr, element->GetSemanticIndexVec().at(row), uint32_t(reg),
              element->GetStartCol(), element->GetCols(), selected, uint32_t(output) |
              (element->GetKind() != hlsl::DXIL::SemanticKind::Arbitrary ? 2u : 0u), uint32_t(flat)});
        }
      }
    };
    reflect(signature.InputSignature, false);
    reflect(signature.OutputSignature, true);
    if (linkage.empty()) return SCORPI_DXBC_TO_DXIL_OK;
    size_t bytes = linkage.size() * sizeof(ScorpiShaderLinkageElement);
    for (const auto &name : names) bytes += name.size() + 1;
    auto *result = static_cast<ScorpiShaderLinkageElement *>(std::malloc(bytes));
    if (!result) throw std::runtime_error("linkage allocation failed");
    std::memcpy(result, linkage.data(), linkage.size() * sizeof(*result));
    char *name_cursor = reinterpret_cast<char *>(result + linkage.size());
    for (size_t i = 0; i < names.size(); ++i) {
      result[i].semantic_name = name_cursor;
      std::memcpy(name_cursor, names[i].c_str(), names[i].size() + 1);
      name_cursor += names[i].size() + 1;
    }
    *elements = result; *count = linkage.size();
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (const std::exception &e) {
    SetError(error, error_size, e.what());
    return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
  } catch (...) {
    SetError(error, error_size, "unexpected linkage inspection exception");
    return SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR;
  }
}

extern "C" int ScorpiInspectShaderLinkage(const void *code, size_t size,
    const void *metadata, size_t metadata_size,
    ScorpiShaderLinkageElement **elements, size_t *count,
    char *error, size_t error_size) {
  return InspectShaderLinkage(code, size, metadata, metadata_size, elements, count, error, error_size, false);
}

extern "C" int ScorpiInspectRegisterLinkage(const void *code, size_t size,
    const void *metadata, size_t metadata_size,
    ScorpiShaderLinkageElement **elements, size_t *count,
    char *error, size_t error_size) {
  return InspectShaderLinkage(code, size, metadata, metadata_size, elements, count, error, error_size, true);
}

extern "C" int ScorpiLinkRegisterInputs(
    const void *code, size_t size, const ScorpiShaderLinkageElement *upstream,
    size_t upstream_count, void **binary, size_t *binary_size,
    char *error, size_t error_size) {
  SetError(error, error_size, "");
  if (!code || !size || !binary || !binary_size || (!upstream && upstream_count))
    return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  *binary = nullptr; *binary_size = 0;
  try {
    using namespace llvm;
    using namespace hlsl;
    DxcMallocScope malloc_scope;
    auto normalized = ConvertToDxilBinary(code, size, nullptr, 0);
    auto *container = reinterpret_cast<const DxilContainerHeader *>(normalized.data());
    if (!IsValidDxilContainer(container, normalized.size()))
      throw std::runtime_error("invalid register-linkage container");
    auto *program = GetDxilProgramHeader(container, DFCC_DXIL);
    if (!program) throw std::runtime_error("missing register-linkage program");
    LLVMContext context;
    auto source = MemoryBuffer::getMemBuffer(StringRef(
        GetDxilBitcodeData(program), GetDxilBitcodeSize(program)), "register-linkage", false);
    auto parsed = parseBitcodeFile(source->getMemBufferRef(), context);
    if (!parsed) throw std::runtime_error("invalid register-linkage bitcode");
    auto *dm = DxilModule::TryGetDxilModule(parsed.get().get());
    if (!dm || !(dm->GetShaderModel()->IsPS() || dm->GetShaderModel()->IsGS()))
      throw std::runtime_error("register input linkage requires a pixel or geometry shader");
    ScorpiWidenMinimumPrecisionIO(*dm);
    std::vector<std::unique_ptr<DxilSignatureElement>> original;
    for (const auto &e : dm->GetInputSignature().GetElements())
      original.emplace_back(new DxilSignatureElement(*e));
    auto oldElement = [&](unsigned id) -> const DxilSignatureElement & {
      for (const auto &e : original) if (e->GetID() == id) return *e;
      throw std::runtime_error("invalid register-linkage input ID");
    };
    struct Group {
      std::string name;
      CompType type;
      InterpolationMode interpolation;
      unsigned reg, col, cols, id;
      DXIL::SemanticKind kind;
      std::vector<unsigned> indices;
    };
    std::vector<ScorpiShaderLinkageElement> layout;
    for (size_t i = 0; i < upstream_count; ++i) {
      const auto &e = upstream[i];
      if (!e.semantic_name || !(e.output & 1u) || e.register_index >= 32 ||
          e.first_component >= 4 || !e.component_count ||
          e.component_count > 4 - e.first_component ||
          e.component_type < 1 || e.component_type > 3)
        throw std::runtime_error("invalid upstream register layout");
      layout.push_back(e);
    }
    std::vector<Group> groups;
    const bool geometry = dm->GetShaderModel()->IsGS();
    for (const auto &out : layout) {
      const bool system = (out.output & 2u) != 0;
      if (system && !geometry) continue;
      CompType type = CompType::getInvalid();
      InterpolationMode interpolation(DXIL::InterpolationMode::Undefined);
      for (const auto &in : original) {
        if (in->GetKind() != DXIL::SemanticKind::Arbitrary || !in->IsAllocated() ||
            out.register_index < unsigned(in->GetStartRow()) ||
            out.register_index >= unsigned(in->GetStartRow()) + in->GetRows() ||
            out.first_component >= unsigned(in->GetStartCol()) + in->GetCols() ||
            unsigned(in->GetStartCol()) >= out.first_component + out.component_count)
          continue;
        if (in->GetCompType().GetSizeInBits() != 32)
          throw std::runtime_error("register input linkage requires 32-bit components");
        if (type.IsInvalid()) {
          type = in->GetCompType(); interpolation = *in->GetInterpolationMode();
        } else if (!(type == in->GetCompType()) ||
                   interpolation.GetKind() != in->GetInterpolationMode()->GetKind()) {
          throw std::runtime_error("mixed types or interpolation within upstream input element");
        }
      }
      if (geometry && (system || type.IsInvalid())) {
        type = out.component_type == 1 ? CompType::getU32() :
            out.component_type == 2 ? CompType::getI32() :
            out.component_type == 3 ? CompType::getF32() : CompType::getInvalid();
      }
      if (type.IsInvalid()) continue;
      const auto kind = system ? Semantic::GetByName(out.semantic_name)->GetKind() : DXIL::SemanticKind::Arbitrary;
      if (!groups.empty()) {
        auto &last = groups.back();
        if (last.name == out.semantic_name && last.type == type && last.kind == kind &&
            last.interpolation.GetKind() == interpolation.GetKind() &&
            last.reg + last.indices.size() == out.register_index &&
            last.col == out.first_component && last.cols == out.component_count) {
          last.indices.push_back(out.semantic_index);
          continue;
        }
      }
      groups.push_back({out.semantic_name, type, interpolation, out.register_index,
                        out.first_component, out.component_count, 0, kind, {out.semantic_index}});
    }
    auto &signature = dm->GetInputSignature();
    signature.ClearElements();
    unsigned occupied[32] = {};
    for (const auto &g : groups)
      for (unsigned row = 0; row < g.indices.size(); ++row) {
        const unsigned mask = ((1u << g.cols) - 1) << g.col;
        if (occupied[g.reg + row] & mask)
          throw std::runtime_error("overlapping upstream input elements");
        occupied[g.reg + row] |= mask;
      }
    std::map<unsigned, unsigned> systemIDs;
    for (const auto &e : original) if (e->GetKind() != DXIL::SemanticKind::Arbitrary) {
      if (geometry && e->IsAllocated() && std::any_of(groups.begin(), groups.end(), [&](const Group &g) {
            return g.kind == e->GetKind() && g.name == e->GetName();
          })) continue;
      std::unique_ptr<DxilSignatureElement> copy(new DxilSignatureElement(*e));
      // System values are linked by their system semantic. The consumer may
      // originally have packed them in a register now used by an upstream
      // arbitrary element; choose free internal input storage in that case.
      if (copy->IsAllocated()) {
        bool allocated = false;
        for (unsigned row = 0; row + copy->GetRows() <= 32 && !allocated; ++row)
          for (unsigned col = 0; col + copy->GetCols() <= 4 && !allocated; ++col) {
            unsigned mask = ((1u << copy->GetCols()) - 1) << col;
            bool free = true;
            for (unsigned r = 0; r < copy->GetRows(); ++r)
              free &= !(occupied[row + r] & mask);
            if (!free) continue;
            copy->SetStartRow(row); copy->SetStartCol(col);
            for (unsigned r = 0; r < copy->GetRows(); ++r) occupied[row + r] |= mask;
            allocated = true;
          }
        if (!allocated) throw std::runtime_error("linked system inputs exceed signature storage");
      }
      systemIDs[e->GetID()] = signature.AppendElement(std::move(copy));
    }
    for (auto &g : groups) {
      auto e = signature.CreateElement();
      e->Initialize(g.name, g.type, g.interpolation, g.indices.size(), g.cols,
                    g.reg, g.col, DxilSignatureElement::kUndefinedID, g.indices);
      e->SetKind(g.kind);
      e->SetUsageMask((1u << g.cols) - 1);
      g.id = signature.AppendElement(std::move(e));
    }
    struct Location { unsigned id, row, col; bool valid; };
    auto locate = [&](unsigned reg, unsigned col) -> Location {
      for (const auto &g : groups)
        if (reg >= g.reg && reg < g.reg + g.indices.size() &&
            col >= g.col && col < g.col + g.cols)
          return {g.id, reg - g.reg, col - g.col, true};
      return {0, 0, 0, false};
    };
    std::vector<CallInst *> loads;
    for (auto &f : *parsed.get()) for (auto &block : f) for (auto &inst : block)
      if (auto *call = dyn_cast<CallInst>(&inst))
        if (call->getCalledFunction() && call->getCalledFunction()->getName().startswith("dx.op.loadInput."))
          loads.push_back(call);
    for (auto *old : loads) {
      auto *id = dyn_cast<ConstantInt>(old->getArgOperand(1));
      auto *column = dyn_cast<ConstantInt>(old->getArgOperand(3));
      if (!id || !column) throw std::runtime_error("nonconstant register-linkage ID or column");
      const auto &in = oldElement(id->getZExtValue());
      if (in.GetKind() != DXIL::SemanticKind::Arbitrary && systemIDs.count(in.GetID())) {
        old->setArgOperand(1, ConstantInt::get(old->getArgOperand(1)->getType(), systemIDs.at(in.GetID())));
        continue;
      }
      if (column->getZExtValue() >= in.GetCols())
        throw std::runtime_error("register-linkage column outside signature");
      IRBuilder<> at(old);
      auto *i32 = at.getInt32Ty();
      const unsigned col = unsigned(in.GetStartCol()) + column->getZExtValue();
      auto read = [&](Location location, Value *row) -> Value * {
        if (!location.valid) return UndefValue::get(old->getType());
        auto *type = signature.GetElement(location.id).GetCompType().IsFloatTy() ?
            at.getFloatTy() : static_cast<Type *>(i32);
        auto *call = at.CreateCall(dm->GetOP()->GetOpFunc(DXIL::OpCode::LoadInput, type),
            {old->getArgOperand(0), at.getInt32(location.id), row,
             at.getInt8(location.col), old->getArgOperand(4)});
        call->setDebugLoc(old->getDebugLoc());
        return call->getType() == old->getType() ? static_cast<Value *>(call) :
            at.CreateBitCast(call, old->getType());
      };
      Value *value = nullptr;
      auto *oldRow = old->getArgOperand(2);
      if (auto *row = dyn_cast<ConstantInt>(oldRow)) {
        auto location = locate(unsigned(in.GetStartRow()) + row->getZExtValue(), col);
        if (in.GetKind() != DXIL::SemanticKind::Arbitrary) {
          location = {0, 0, 0, false};
          if (row->getZExtValue() < in.GetSemanticIndexVec().size())
            for (const auto &g : groups) {
              if (g.kind != in.GetKind() || g.name != in.GetName()) continue;
              const auto index = in.GetSemanticIndexVec()[row->getZExtValue()];
              const auto found = std::find(g.indices.begin(), g.indices.end(), index);
              if (found != g.indices.end() && col >= g.col && col < g.col + g.cols)
                location = {g.id, unsigned(found - g.indices.begin()), col - g.col, true};
            }
        }
        value = read(location, at.getInt32(location.row));
      } else {
        // Preserve dynamic array access directly when the linked rows still
        // form one rectangle. Only irregular layouts require selection.
        auto first = locate(in.GetStartRow(), col);
        bool rectangular = first.valid;
        for (unsigned row = 1; row < in.GetRows() && rectangular; ++row) {
          auto next = locate(unsigned(in.GetStartRow()) + row, col);
          rectangular = next.valid && next.id == first.id &&
              next.col == first.col && next.row == first.row + row;
        }
        if (rectangular) {
          auto *row = first.row ? at.CreateAdd(oldRow, at.getInt32(first.row)) : oldRow;
          auto &e = signature.GetElement(first.id);
          e.SetDynIdxCompMask(e.GetDynIdxCompMask() | (1u << first.col));
          value = read(first, row);
        } else {
          std::function<Value *(unsigned, unsigned)> select = [&](unsigned begin, unsigned end) -> Value * {
            if (end - begin == 1) {
              auto location = locate(unsigned(in.GetStartRow()) + begin, col);
              return read(location, at.getInt32(location.row));
            }
            unsigned mid = begin + (end - begin) / 2;
            auto *left = select(begin, mid); auto *right = select(mid, end);
            return at.CreateSelect(at.CreateICmpULT(oldRow, at.getInt32(mid)), left, right);
          };
          if (!in.GetRows()) throw std::runtime_error("empty register input array");
          value = select(0, in.GetRows());
        }
      }
      old->replaceAllUsesWith(value); old->eraseFromParent();
    }
    if (verifyModule(*parsed.get())) throw std::runtime_error("invalid register-linkage IR");
    dm->GetSerializedViewIdState().clear();
    DxilModule::ClearDxilMetadata(*parsed.get()); dm->EmitDxilMetadata();
    CComPtr<AbstractMemoryStream> bitcode, output;
    IFT(CreateMemoryStream(DxcGetThreadMallocNoRef(), &bitcode));
    IFT(CreateMemoryStream(DxcGetThreadMallocNoRef(), &output));
    { raw_stream_ostream stream(bitcode.p); WriteBitcodeToFile(parsed.get().get(), stream); }
    SerializeDxilContainerForModule(dm, bitcode, nullptr, output,
        StringRef(), SerializeDxilFlags::None);
    auto *result = std::malloc(output->GetPtrSize());
    if (!result) throw std::runtime_error("register-linkage allocation failed");
    std::memcpy(result, output->GetPtr(), output->GetPtrSize());
    auto *header = static_cast<DxilContainerHeader *>(result);
    constexpr uint32_t hashOffset = uint32_t(offsetof(DxilContainerHeader, Version));
    ComputeHashRetail(static_cast<const uint8_t *>(result) + hashOffset,
        header->ContainerSizeInBytes - hashOffset, header->Hash.Digest);
    *binary = result; *binary_size = output->GetPtrSize();
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (const std::exception &e) {
    SetError(error, error_size, e.what()); return SCORPI_DXBC_TO_DXIL_CONVERSION_FAILED;
  } catch (...) {
    SetError(error, error_size, "unexpected register-linkage exception");
    return SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR;
  }
}

extern "C" int ScorpiRetypeShaderOutputs(
    const void *code, size_t size, const uint16_t component_types[32],
    void **binary, size_t *binary_size, char *error, size_t error_size) {
  SetError(error, error_size, "");
  if (!component_types || !binary || !binary_size)
    return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  *binary = nullptr; *binary_size = 0;
  try {
    DxcMallocScope malloc_scope;
    auto normalized = ConvertToDxilBinary(code, size, nullptr, 0);
    auto *container = reinterpret_cast<const hlsl::DxilContainerHeader *>(normalized.data());
    if (!hlsl::IsValidDxilContainer(container, normalized.size()))
      throw std::runtime_error("invalid output-linkage container");
    auto *program = hlsl::GetDxilProgramHeader(container, hlsl::DFCC_DXIL);
    if (!program) throw std::runtime_error("missing output-linkage program");
    llvm::LLVMContext context;
    auto source = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(
        hlsl::GetDxilBitcodeData(program), hlsl::GetDxilBitcodeSize(program)), "output-linkage", false);
    auto parsed = llvm::parseBitcodeFile(source->getMemBufferRef(), context);
    if (!parsed) throw std::runtime_error("invalid output-linkage bitcode");
    auto *dm = hlsl::DxilModule::TryGetDxilModule(parsed.get().get());
    if (!dm || !(dm->GetShaderModel()->IsVS() || dm->GetShaderModel()->IsDS() ||
                 dm->GetShaderModel()->IsGS()))
      throw std::runtime_error("output linkage requires a rasterizer producer");
    const bool widened = ScorpiWidenMinimumPrecisionIO(*dm);
    std::map<unsigned, llvm::Type *> types;
    for (auto &element : dm->GetOutputSignature().GetElements()) {
      // System values and non-rasterized streams retain their ABI.
      if (element->GetKind() != hlsl::DXIL::SemanticKind::Arbitrary ||
          element->GetOutputStream() != 0) continue;
      unsigned selected = 0;
      for (unsigned row = 0; row < element->GetRows(); ++row) {
        const int reg = element->GetStartRow() + int(row);
        if (reg < 0 || reg >= 32) continue;
        for (unsigned col = element->GetStartCol();
             col < element->GetStartCol() + element->GetCols(); ++col) {
          const unsigned type = (component_types[reg] >> (4 * col)) & 15;
          if (!type) continue;
          if (type > 3 || (selected && selected != type))
            throw std::runtime_error("mixed component types within one linkage element");
          selected = type;
        }
      }
      if (!selected) continue;
      const auto type = selected == 1 ? hlsl::CompType::getU32() :
          selected == 2 ? hlsl::CompType::getI32() : hlsl::CompType::getF32();
      if (element->GetCompType() == type) continue;
      if (element->GetCompType().GetSizeInBits() != 32)
        throw std::runtime_error("output linkage requires 32-bit components");
      // D3D11.3 7.8.1: instructions interpret untyped 32-bit registers.
      // Match Metal's typed interface while preserving those register bits.
      element->SetCompType(type);
      types[element->GetID()] = selected == 3 ? llvm::Type::getFloatTy(context) :
          llvm::Type::getInt32Ty(context);
    }
    if (types.empty() && !widened) return SCORPI_DXBC_TO_DXIL_OK;
    std::vector<llvm::CallInst *> stores;
    for (auto &f : *parsed.get()) for (auto &block : f) for (auto &inst : block)
      if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst))
        if (call->getCalledFunction() &&
            call->getCalledFunction()->getName().startswith("dx.op.storeOutput."))
          stores.push_back(call);
    for (auto *call : stores) {
      auto *id = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(1));
      if (!id) throw std::runtime_error("dynamic output linkage ID");
      auto type = types.find(unsigned(id->getZExtValue()));
      if (type == types.end()) continue;
      llvm::IRBuilder<> at(call);
      auto *value = call->getArgOperand(4);
      if (!value->getType()->isIntegerTy(32) && !value->getType()->isFloatTy())
        throw std::runtime_error("unsupported output linkage store type");
      auto *store = at.CreateCall(dm->GetOP()->GetOpFunc(hlsl::DXIL::OpCode::StoreOutput, type->second),
          {call->getArgOperand(0), call->getArgOperand(1), call->getArgOperand(2),
           call->getArgOperand(3), at.CreateBitCast(value, type->second)});
      store->setDebugLoc(call->getDebugLoc()); call->eraseFromParent();
    }
    if (llvm::verifyModule(*parsed.get())) throw std::runtime_error("invalid output linkage IR");
    dm->GetSerializedViewIdState().clear();
    hlsl::DxilModule::ClearDxilMetadata(*parsed.get()); dm->EmitDxilMetadata();
    CComPtr<hlsl::AbstractMemoryStream> bitcode, output;
    IFT(hlsl::CreateMemoryStream(DxcGetThreadMallocNoRef(), &bitcode));
    IFT(hlsl::CreateMemoryStream(DxcGetThreadMallocNoRef(), &output));
    { raw_stream_ostream stream(bitcode.p); llvm::WriteBitcodeToFile(parsed.get().get(), stream); }
    hlsl::SerializeDxilContainerForModule(dm, bitcode, nullptr, output,
        llvm::StringRef(), hlsl::SerializeDxilFlags::None);
    auto *result = std::malloc(output->GetPtrSize());
    if (!result) throw std::runtime_error("output linkage allocation failed");
    std::memcpy(result, output->GetPtr(), output->GetPtrSize());
    auto *header = static_cast<hlsl::DxilContainerHeader *>(result);
    constexpr uint32_t hashOffset = uint32_t(offsetof(hlsl::DxilContainerHeader, Version));
    ComputeHashRetail(static_cast<const uint8_t *>(result) + hashOffset,
        header->ContainerSizeInBytes - hashOffset, header->Hash.Digest);
    *binary = result; *binary_size = output->GetPtrSize();
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (const std::exception &exception) {
    SetError(error, error_size, exception.what());
    return SCORPI_DXBC_TO_DXIL_CONVERSION_FAILED;
  } catch (...) {
    SetError(error, error_size, "unexpected output linkage exception");
    return SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR;
  }
}

extern "C" int ScorpiUnrollMeshOutputLoops(
    const void *code, size_t size, void **binary, size_t *binary_size,
    char *error, size_t error_size) {
  SetError(error, error_size, "");
  if (!binary || !binary_size) return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  *binary = nullptr; *binary_size = 0;
  try {
    DxcMallocScope malloc_scope;
    auto normalized = ConvertToDxilBinary(code, size, nullptr, 0);
    auto *container = reinterpret_cast<const hlsl::DxilContainerHeader *>(normalized.data());
    if (!hlsl::IsValidDxilContainer(container, normalized.size()))
      throw std::runtime_error("invalid mesh shader container");
    auto *program = hlsl::GetDxilProgramHeader(container, hlsl::DFCC_DXIL);
    if (!program) throw std::runtime_error("missing mesh shader program");
    llvm::LLVMContext context;
    auto source = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(
        hlsl::GetDxilBitcodeData(program), hlsl::GetDxilBitcodeSize(program)), "mesh-output-loops", false);
    auto parsed = llvm::parseBitcodeFile(source->getMemBufferRef(), context);
    if (!parsed) throw std::runtime_error("invalid mesh shader bitcode");
    auto *dm = hlsl::DxilModule::TryGetDxilModule(parsed.get().get());
    if (!dm) throw std::runtime_error("missing mesh shader module");
    if (!dm->GetShaderModel()->IsMS()) return SCORPI_DXBC_TO_DXIL_OK;
    if (!HasDynamicMeshOutputs(*parsed.get())) return SCORPI_DXBC_TO_DXIL_OK;
    auto &registry = *llvm::PassRegistry::getPassRegistry();
    llvm::initializeCore(registry);
    llvm::initializeAnalysis(registry);
    llvm::initializeTransformUtils(registry);
    llvm::initializeScalarOpts(registry);
    llvm::legacy::PassManager passes;
    passes.add(new MeshOutputUnroll());
    passes.run(*parsed.get());
    if (HasDynamicMeshOutputs(*parsed.get())) throw std::runtime_error("mesh output row cannot be resolved at compile time");
    if (llvm::verifyModule(*parsed.get())) throw std::runtime_error("invalid mesh output unroll IR");
    CComPtr<hlsl::AbstractMemoryStream> bitcode, output;
    IFT(hlsl::CreateMemoryStream(DxcGetThreadMallocNoRef(), &bitcode));
    IFT(hlsl::CreateMemoryStream(DxcGetThreadMallocNoRef(), &output));
    { raw_stream_ostream stream(bitcode.p); llvm::WriteBitcodeToFile(parsed.get().get(), stream); }
    hlsl::SerializeDxilContainerForModule(dm, bitcode, nullptr, output,
        llvm::StringRef(), hlsl::SerializeDxilFlags::None);
    auto *result = std::malloc(output->GetPtrSize());
    if (!result) throw std::runtime_error("failed to allocate unrolled mesh shader");
    std::memcpy(result, output->GetPtr(), output->GetPtrSize());
    auto *header = static_cast<hlsl::DxilContainerHeader *>(result);
    constexpr uint32_t hashOffset = static_cast<uint32_t>(offsetof(hlsl::DxilContainerHeader, Version));
    ComputeHashRetail(static_cast<const uint8_t *>(result) + hashOffset,
        header->ContainerSizeInBytes - hashOffset, header->Hash.Digest);
    *binary = result; *binary_size = output->GetPtrSize();
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (const std::exception &exception) {
    SetError(error, error_size, exception.what());
    return SCORPI_DXBC_TO_DXIL_CONVERSION_FAILED;
  }
}

namespace {
using TextureKind = hlsl::DXIL::ResourceKind;
TextureKind OriginalTextureKind(llvm::Value *value, hlsl::DxilModule &dm,
                                unsigned depth = 0) {
  if (!value || depth > 12) return TextureKind::Invalid;
  auto *call = llvm::dyn_cast<llvm::CallInst>(value);
  if (call && call->getNumArgOperands() > 2) {
    auto *opcode = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0));
    if (opcode) switch (static_cast<hlsl::DXIL::OpCode>(opcode->getZExtValue())) {
    case hlsl::DXIL::OpCode::AnnotateHandle:
      if (auto *properties = llvm::dyn_cast<llvm::Constant>(call->getArgOperand(2)))
        return hlsl::resource_helper::loadPropsFromConstant(*properties).getResourceKind();
      break;
    case hlsl::DXIL::OpCode::CreateHandle: {
      auto *type = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(1));
      auto *id = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(2));
      if (!type || !id) break;
      const auto index = id->getZExtValue();
      if (type->getZExtValue() == unsigned(hlsl::DXIL::ResourceClass::SRV) &&
          index < dm.GetSRVs().size()) return dm.GetSRV(unsigned(index)).GetKind();
      if (type->getZExtValue() == unsigned(hlsl::DXIL::ResourceClass::UAV) &&
          index < dm.GetUAVs().size()) return dm.GetUAV(unsigned(index)).GetKind();
      break;
    }
    default: break;
    }
  }
  if (auto *select = llvm::dyn_cast<llvm::SelectInst>(value)) {
    auto a = OriginalTextureKind(select->getTrueValue(), dm, depth + 1);
    auto b = OriginalTextureKind(select->getFalseValue(), dm, depth + 1);
    return a == b ? a : TextureKind::Invalid;
  }
  if (auto *phi = llvm::dyn_cast<llvm::PHINode>(value)) {
    TextureKind result = TextureKind::Invalid;
    for (auto &incoming : phi->incoming_values()) {
      auto kind = OriginalTextureKind(incoming, dm, depth + 1);
      if (result != TextureKind::Invalid && result != kind) return TextureKind::Invalid;
      result = kind;
    }
    return result;
  }
  return TextureKind::Invalid;
}
}

extern "C" int ScorpiLowerTypedBufferViews(const void *code, size_t size,
    const ScorpiTypedBufferBinding *bindings, size_t binding_count,
    uint32_t metadata_space, uint32_t self_offset,
    void **binary, size_t *binary_size, char *error, size_t error_size) {
  return ScorpiLowerTypedBufferViewsWithOptions(code, size, bindings, binding_count,
      metadata_space, self_offset, 0, binary, binary_size, error, error_size);
}

extern "C" int ScorpiInspectVertexInputs(const void *code, size_t size,
    ScorpiVertexInput *inputs, size_t *count, char *error, size_t error_size) {
  SetError(error, error_size, "");
  if (!code || !size || !count) return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  try {
    EnsureDxcStaticRuntimeInitialized();
    DxcMallocScope malloc_scope;
    auto normalized = NormalizeExistingDxil(code, size);
    if (normalized.empty()) throw std::runtime_error("expected DXIL program or container");
    auto *container = reinterpret_cast<const hlsl::DxilContainerHeader *>(normalized.data());
    auto *program = hlsl::GetDxilProgramHeader(container, hlsl::DFCC_DXIL);
    if (!program) throw std::runtime_error("missing DXIL program");
    llvm::LLVMContext context;
    auto buffer = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(
        hlsl::GetDxilBitcodeData(program), hlsl::GetDxilBitcodeSize(program)), "vertex-input-reflection", false);
    auto parsed = llvm::parseBitcodeFile(buffer->getMemBufferRef(), context);
    if (!parsed) throw std::runtime_error("invalid DXIL bitcode");
    auto *dm = hlsl::DxilModule::TryGetDxilModule(parsed.get().get());
    if (!dm || !dm->GetShaderModel()->IsVS()) throw std::runtime_error("expected vertex shader");
    std::vector<ScorpiVertexInput> reflected;
    for (const auto &element : dm->GetInputSignature().GetElements()) {
      if (element->GetStartRow() < 0) continue; // hardware-generated input
      const auto &indices = element->GetSemanticIndexVec();
      for (unsigned row = 0; row < element->GetRows(); ++row) {
        if (row >= indices.size()) throw std::runtime_error("missing input semantic index");
        ScorpiVertexInput input{};
        input.register_index = element->GetStartRow() + row;
        input.semantic_index = indices[row];
        const char *name = element->GetName();
        if (!name || std::strlen(name) >= sizeof(input.name))
          throw std::runtime_error("input semantic name too long");
        std::strcpy(input.name, name);
        reflected.push_back(input);
      }
    }
    const size_t capacity = *count;
    *count = reflected.size();
    if (inputs) {
      if (capacity < reflected.size()) throw std::runtime_error("input reflection capacity exceeded");
      std::copy(reflected.begin(), reflected.end(), inputs);
    }
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (const std::exception &e) {
    SetError(error, error_size, e.what());
    return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
  } catch (...) {
    SetError(error, error_size, "unexpected vertex reflection exception");
    return SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR;
  }
}

static bool SampleOnlyHandle(llvm::Value *handle) {
  llvm::SmallPtrSet<llvm::Value *, 16> visited;
  std::vector<llvm::Value *> pending{handle};
  bool sampled = false;
  while (!pending.empty()) {
    auto *value = pending.back(); pending.pop_back();
    if (!visited.insert(value).second) continue;
    for (auto *user : value->users()) {
      if (llvm::isa<llvm::PHINode>(user) || llvm::isa<llvm::SelectInst>(user)) {
        pending.push_back(user); continue;
      }
      auto *call = llvm::dyn_cast<llvm::CallInst>(user);
      if (!call || !call->getCalledFunction() ||
          !call->getCalledFunction()->getName().startswith("dx.op.") ||
          call->getNumArgOperands() < 2 || call->getArgOperand(1) != value) return false;
      auto *opcode = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0));
      if (!opcode) return false;
      using Op = hlsl::DXIL::OpCode;
      switch (Op(opcode->getZExtValue())) {
      case Op::AnnotateHandle: pending.push_back(call); break;
      case Op::Sample: case Op::SampleBias: case Op::SampleLevel: case Op::SampleGrad:
      case Op::SampleCmp: case Op::SampleCmpLevelZero: case Op::SampleCmpLevel:
      case Op::SampleCmpBias: case Op::SampleCmpGrad:
      case Op::TextureGather: case Op::TextureGatherCmp:
        sampled = true; break;
      default: return false;
      }
    }
  }
  return sampled;
}

static void SetSampleOnly(ScorpiSamplingBinding &, bool) {}
static void SetSampleOnly(ScorpiDescriptorAccess &r, bool value) { r.sample_only = value; }
static uint32_t SampleOnly(const ScorpiSamplingBinding &) { return 0; }
static uint32_t SampleOnly(const ScorpiDescriptorAccess &r) { return r.sample_only; }

template <typename Binding>
static int InspectBindings(const void *code, size_t size,
    const void *metadata, size_t metadata_size, Binding *bindings,
    size_t *count, bool access_only) {
  if (!code || !size || !count) return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  try {
    DxcMallocScope malloc_scope;
    auto normalized = ConvertToDxilBinary(code, size, metadata, metadata_size);
    auto *container = reinterpret_cast<const hlsl::DxilContainerHeader *>(normalized.data());
    auto *program = hlsl::GetDxilProgramHeader(container, hlsl::DFCC_DXIL);
    if (!program) return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
    llvm::LLVMContext context;
    auto buffer = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(
        hlsl::GetDxilBitcodeData(program), hlsl::GetDxilBitcodeSize(program)), "sampling-reflection", false);
    auto parsed = llvm::parseBitcodeFile(buffer->getMemBufferRef(), context);
    if (!parsed) return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
    auto *dm = hlsl::DxilModule::TryGetDxilModule(parsed.get().get());
    if (!dm) return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
    std::vector<Binding> ranges;
    const auto declarations = [&] {
      for (const auto &r : dm->GetSRVs())
        ranges.push_back({0, r->GetSpaceID(), r->GetLowerBound(), r->GetRangeSize()});
      for (const auto &r : dm->GetUAVs())
        ranges.push_back({1, r->GetSpaceID(), r->GetLowerBound(), r->GetRangeSize()});
      for (const auto &r : dm->GetSamplers())
        ranges.push_back({3, r->GetSpaceID(), r->GetLowerBound(), r->GetRangeSize()});
    };
    if (!access_only) declarations();
    bool unknown_handle = false;
    // SamplingRequirements queries whole declarations for dynamic handles,
    // but single registers for constant handles. Expose both observations so
    // the renderer can identify equivalent policies without keying on every
    // descriptor's position within a bindless array. Keep the declarations:
    // texture-layout lowering also observes their union.
    for (auto &function : *parsed.get()) for (auto &block : function)
      for (auto &instruction : block) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
        if (!call || !call->getCalledFunction() ||
            !call->getCalledFunction()->getName().startswith("dx.op.")) continue;
        if (call->getNumArgOperands() < 4) {
          unknown_handle |= call->getCalledFunction()->getName().startswith("dx.op.createHandle");
          continue;
        }
        auto *opcode = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0));
        if (!opcode) continue;
        uint32_t cls = 0, space = 0;
        uint64_t first = 0, end = 0;
        llvm::Value *index = nullptr;
        if (opcode->getZExtValue() == unsigned(hlsl::DXIL::OpCode::CreateHandle)) {
          auto *type = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(1));
          auto *id = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(2));
          if (!type || !id) { unknown_handle = true; continue; }
          cls = uint32_t(type->getZExtValue());
          const hlsl::DxilResourceBase *resource = nullptr;
          const auto slot = id->getZExtValue();
          // Range IDs are metadata IDs, not necessarily vector positions.
          if (cls == 0) for (const auto &r : dm->GetSRVs()) if (r->GetID() == slot) resource = r.get();
          if (cls == 1) for (const auto &r : dm->GetUAVs()) if (r->GetID() == slot) resource = r.get();
          if (cls == 3) for (const auto &r : dm->GetSamplers()) if (r->GetID() == slot) resource = r.get();
          if (!resource) { if (cls != 2) unknown_handle = true; continue; }
          space = resource->GetSpaceID(); first = resource->GetLowerBound();
          end = first + resource->GetRangeSize(); index = call->getArgOperand(3);
        } else if (opcode->getZExtValue() == unsigned(hlsl::DXIL::OpCode::CreateHandleFromBinding)) {
          auto *binding = llvm::dyn_cast<llvm::Constant>(call->getArgOperand(1));
          if (!binding) { unknown_handle = true; continue; }
          const auto r = hlsl::resource_helper::loadBindingFromConstant(*binding);
          cls = r.resourceClass; space = r.spaceID; first = r.rangeLowerBound;
          end = uint64_t(r.rangeUpperBound) + 1; index = call->getArgOperand(2);
          if (!access_only && (cls == 0 || cls == 1 || cls == 3) && end > first)
            ranges.push_back({cls, space, uint32_t(first),
                uint32_t(std::min<uint64_t>(end - first, UINT32_MAX))});
        } else if (call->getCalledFunction()->getName().startswith("dx.op.createHandle")) {
          // Do not narrow declarations for an unrecognized handle producer.
          unknown_handle = true;
        }
        auto *constant = llvm::dyn_cast_or_null<llvm::ConstantInt>(index);
        if (!index || (cls != 0 && cls != 1 && cls != 3)) continue;
        const auto before = ranges.size();
        if (constant && constant->getZExtValue() >= first && constant->getZExtValue() < end)
          ranges.push_back({cls, space, uint32_t(constant->getZExtValue()), 1});
        else if (access_only && end > first)
          ranges.push_back({cls, space, uint32_t(first),
              uint32_t(std::min<uint64_t>(end - first, UINT32_MAX))});
        if (access_only && ranges.size() != before)
          SetSampleOnly(ranges.back(), cls == 0 && SampleOnlyHandle(call));
      }
    if (access_only && unknown_handle) declarations();
    const auto identity = [](const Binding &r) {
      return std::array<uint32_t, 5>{r.resource_class, r.space, r.base_register, r.count, SampleOnly(r)};
    };
    std::sort(ranges.begin(), ranges.end(), [&](const auto &a, const auto &b) { return identity(a) < identity(b); });
    ranges.erase(std::unique(ranges.begin(), ranges.end(), [&](const auto &a, const auto &b) {
      return identity(a) == identity(b);
    }), ranges.end());
    const auto capacity = *count;
    *count = ranges.size();
    if (bindings) {
      if (capacity < ranges.size()) return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
      std::copy(ranges.begin(), ranges.end(), bindings);
    }
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (...) { return SCORPI_DXBC_TO_DXIL_INVALID_SHADER; }
}

extern "C" int ScorpiInspectSamplingBindings(const void *code, size_t size,
    const void *metadata, size_t metadata_size, ScorpiSamplingBinding *bindings, size_t *count) {
  return InspectBindings(code, size, metadata, metadata_size, bindings, count, false);
}

extern "C" int ScorpiInspectDescriptorAccess(const void *code, size_t size,
    const void *metadata, size_t metadata_size, ScorpiDescriptorAccess *bindings, size_t *count) {
  return InspectBindings(code, size, metadata, metadata_size, bindings, count, true);
}

extern "C" int ScorpiLowerTypedBufferViewsWithOptions(const void *code, size_t size,
    const ScorpiTypedBufferBinding *bindings, size_t binding_count,
    uint32_t metadata_space, uint32_t self_offset, uint32_t options,
    void **binary, size_t *binary_size, char *error, size_t error_size) {
  if (options & ~(SCORPI_SHADER_NATIVE_TEXTURES | SCORPI_SHADER_NO_MINMAX_FILTERING))
    return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  return ScorpiLowerTypedBufferViewsWithSamplingPolicy(code, size, bindings, binding_count,
      metadata_space, self_offset, options, nullptr, 0, binary, binary_size, error, error_size);
}

extern "C" int ScorpiLowerTypedBufferViewsWithSamplingPolicy(const void *code, size_t size,
    const ScorpiTypedBufferBinding *bindings, size_t binding_count,
    uint32_t metadata_space, uint32_t self_offset, uint32_t options,
    const ScorpiSamplingRequirement *policy, size_t policy_count,
    void **binary, size_t *binary_size, char *error, size_t error_size) {
  SetError(error, error_size, "");
  if (!binary || !binary_size || !code || !size || (!bindings && binding_count) ||
      (!policy && policy_count) ||
      (options & ~(SCORPI_SHADER_NATIVE_TEXTURES | SCORPI_SHADER_NO_MINMAX_FILTERING | SCORPI_SHADER_BINDING_POLICY | SCORPI_SHADER_NATIVE_TYPED_BUFFERS | SCORPI_SHADER_TYPED_DIMENSIONS_ONLY | SCORPI_SHADER_FLUSH_FP32_COMPARISON_INPUTS | SCORPI_SHADER_FLUSH_FP32_ARITHMETIC | SCORPI_SHADER_FLAT_FLOAT_INPUT_BITS)))
    return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  *binary = nullptr; *binary_size = 0;
  try {
    DxcMallocScope malloc_scope;
    auto normalized = ConvertToDxilBinary(code, size, nullptr, 0);
    auto *container = reinterpret_cast<const hlsl::DxilContainerHeader *>(normalized.data());
    if (!hlsl::IsValidDxilContainer(container, normalized.size()))
      throw std::runtime_error("invalid typed-buffer shader container");
    auto *program = hlsl::GetDxilProgramHeader(container, hlsl::DFCC_DXIL);
    if (!program) throw std::runtime_error("typed-buffer shader has no DXIL program");
    llvm::LLVMContext context;
    auto source = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(
        hlsl::GetDxilBitcodeData(program), hlsl::GetDxilBitcodeSize(program)), "typed-buffer", false);
    auto parsed = llvm::parseBitcodeFile(source->getMemBufferRef(), context);
    if (!parsed) throw std::runtime_error("invalid typed-buffer shader bitcode");
    auto *dm = hlsl::DxilModule::TryGetDxilModule(parsed.get().get());
    if (!dm) throw std::runtime_error("missing typed-buffer shader module");
    const bool trace = std::getenv("SCORPI_SHADER_TRACE") != nullptr;
    auto instructions = [&]() { size_t n = 0; for (const auto &f : *parsed.get()) for (const auto &b : f) n += b.size(); return n; };
    const size_t before = trace ? instructions() : 0;
    if (!ScorpiLowerTypedBuffers(*dm, bindings, binding_count, metadata_space, self_offset, options, policy, policy_count))
      return SCORPI_DXBC_TO_DXIL_OK;
    if (trace) std::fprintf(stderr, "shader lowering options=%u requirements=%zu instructions_before=%zu instructions_after=%zu\n",
        options, policy_count, before, instructions());
    if (llvm::verifyModule(*parsed.get())) throw std::runtime_error("invalid typed-buffer lowering IR");
    hlsl::DxilModule::ClearDxilMetadata(*parsed.get());
    dm->EmitDxilMetadata();
    CComPtr<hlsl::AbstractMemoryStream> bitcode, output;
    IFT(hlsl::CreateMemoryStream(DxcGetThreadMallocNoRef(), &bitcode));
    IFT(hlsl::CreateMemoryStream(DxcGetThreadMallocNoRef(), &output));
    { raw_stream_ostream stream(bitcode.p); llvm::WriteBitcodeToFile(parsed.get().get(), stream); }
    hlsl::SerializeDxilContainerForModule(dm, bitcode, nullptr, output,
        llvm::StringRef(), hlsl::SerializeDxilFlags::None);
    auto *result = std::malloc(output->GetPtrSize());
    if (!result) throw std::runtime_error("failed to allocate typed-buffer shader");
    std::memcpy(result, output->GetPtr(), output->GetPtrSize());
    auto *header = static_cast<hlsl::DxilContainerHeader *>(result);
    constexpr uint32_t hashOffset = static_cast<uint32_t>(offsetof(hlsl::DxilContainerHeader, Version));
    ComputeHashRetail(static_cast<const uint8_t *>(result) + hashOffset,
        header->ContainerSizeInBytes - hashOffset, header->Hash.Digest);
    *binary = result; *binary_size = output->GetPtrSize();
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (const std::exception &exception) {
    SetError(error, error_size, exception.what());
    return SCORPI_DXBC_TO_DXIL_CONVERSION_FAILED;
  } catch (...) {
    SetError(error, error_size, "typed-buffer lowering failed");
    return SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR;
  }
}

extern "C" int ScorpiInspectAtomic64Requirements(const void *code, size_t size, uint32_t *required) {
  if (!code || !size || !required) return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  *required = 0;
  try {
    DxcMallocScope malloc_scope;
    auto *container = hlsl::IsDxilContainerLike(code, size);
    const hlsl::DxilProgramHeader *program = nullptr;
    if (container) {
      if (!hlsl::IsValidDxilContainer(container, size)) return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
      program = hlsl::GetDxilProgramHeader(container, hlsl::DFCC_DXIL);
      if (!program) return SCORPI_DXBC_TO_DXIL_OK;
      // Ordinary container shaders avoid another LLVM parse.
      if (auto *part = hlsl::GetDxilPartByType(container, hlsl::DFCC_FeatureInfo)) {
        if (part->PartSize >= sizeof(hlsl::DxilShaderFeatureInfo)) {
          hlsl::DxilShaderFeatureInfo features;
          std::memcpy(&features, hlsl::GetDxilPartData(part), sizeof(features));
          constexpr uint64_t atomicFeatures = hlsl::DXIL::ShaderFeatureInfo_Int64Ops |
              hlsl::DXIL::ShaderFeatureInfo_AtomicInt64OnTypedResource |
              hlsl::DXIL::ShaderFeatureInfo_AtomicInt64OnGroupShared |
              hlsl::DXIL::ShaderFeatureInfo_AtomicInt64OnHeapResource;
          if (!(features.FeatureFlags & atomicFeatures))
            return SCORPI_DXBC_TO_DXIL_OK;
        }
      }
    } else if (IsDxilProgram(code, size)) {
      // Windows shader DDIs supply the DXIL program without its container.
      // Classify it before pipeline/root creation just as we do after
      // normalization; otherwise lowering and binding disagree on the ABI.
      program = static_cast<const hlsl::DxilProgramHeader *>(code);
    } else {
      // Shader model 5 token streams cannot express i64 atomics.
      return SCORPI_DXBC_TO_DXIL_OK;
    }
    llvm::LLVMContext context;
    auto source = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(
        hlsl::GetDxilBitcodeData(program), hlsl::GetDxilBitcodeSize(program)), "atomic64-inspection", false);
    auto parsed = llvm::parseBitcodeFile(source->getMemBufferRef(), context);
    if (!parsed) return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
    // Opcode inspection needs only instructions. Loading a DxilModule also
    // decodes resource metadata, which can be incomplete in converted DXBC
    // and is not needed to determine whether any i64 atomics are present.
    *required = ScorpiClassifyAtomic64(*parsed.get());
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (...) { return SCORPI_DXBC_TO_DXIL_INVALID_SHADER; }
}

extern "C" int ScorpiInspectAtomic64(const void *code, size_t size, int *required) {
  if (!required) return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  uint32_t requirements=0;
  const auto status=ScorpiInspectAtomic64Requirements(code,size,&requirements);
  *required=requirements!=0;return status;
}

extern "C" int ScorpiEmulateAtomic64(const void *code, size_t size, uint32_t lock_space,
    const ScorpiTypedBufferBinding *bindings, size_t binding_count,
    void **binary, size_t *binary_size, char *error, size_t error_size) {
  return ScorpiEmulateAtomic64WithOptions(code,size,lock_space,bindings,binding_count,0,binary,binary_size,error,error_size);
}

extern "C" int ScorpiEmulateAtomic64WithOptions(const void *code, size_t size, uint32_t lock_space,
    const ScorpiTypedBufferBinding *bindings, size_t binding_count, int native_texture_minmax,
    void **binary, size_t *binary_size, char *error, size_t error_size) {
  SetError(error, error_size, "");
  if (!binary || !binary_size || !code || !size || (!bindings && binding_count))
    return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  *binary = nullptr; *binary_size = 0;
  try {
    DxcMallocScope malloc_scope;
    auto normalized = ConvertToDxilBinary(code, size, nullptr, 0);
    auto *container = reinterpret_cast<const hlsl::DxilContainerHeader *>(normalized.data());
    auto *program = hlsl::GetDxilProgramHeader(container, hlsl::DFCC_DXIL);
    if (!program) throw std::runtime_error("atomic shader has no DXIL program");
    llvm::LLVMContext context;
    auto source = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(
        hlsl::GetDxilBitcodeData(program), hlsl::GetDxilBitcodeSize(program)), "atomic64", false);
    auto parsed = llvm::parseBitcodeFile(source->getMemBufferRef(), context);
    if (!parsed) throw std::runtime_error("invalid atomic shader bitcode");
    auto *dm = hlsl::DxilModule::TryGetDxilModule(parsed.get().get());
    if (!dm) throw std::runtime_error("missing atomic shader module");
    if (!ScorpiLowerAtomic64(*dm, lock_space, bindings, binding_count, native_texture_minmax!=0)) return SCORPI_DXBC_TO_DXIL_OK;
    if (llvm::verifyModule(*parsed.get(), &llvm::errs())) throw std::runtime_error("invalid atomic lowering IR");
    hlsl::DxilModule::ClearDxilMetadata(*parsed.get()); dm->EmitDxilMetadata();
    CComPtr<hlsl::AbstractMemoryStream> bitcode, output;
    IFT(hlsl::CreateMemoryStream(DxcGetThreadMallocNoRef(), &bitcode));
    IFT(hlsl::CreateMemoryStream(DxcGetThreadMallocNoRef(), &output));
    { raw_stream_ostream stream(bitcode.p); llvm::WriteBitcodeToFile(parsed.get().get(), stream); }
    hlsl::SerializeDxilContainerForModule(dm, bitcode, nullptr, output,
        llvm::StringRef(), hlsl::SerializeDxilFlags::None);
    auto *result = std::malloc(output->GetPtrSize());
    if (!result) throw std::runtime_error("failed to allocate atomic shader");
    std::memcpy(result, output->GetPtr(), output->GetPtrSize());
    auto *header = static_cast<hlsl::DxilContainerHeader *>(result);
    constexpr uint32_t hashOffset = static_cast<uint32_t>(offsetof(hlsl::DxilContainerHeader, Version));
    ComputeHashRetail(static_cast<const uint8_t *>(result) + hashOffset,
        header->ContainerSizeInBytes - hashOffset, header->Hash.Digest);
    *binary = result; *binary_size = output->GetPtrSize();
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (const std::exception &exception) {
    SetError(error, error_size, exception.what());
    return SCORPI_DXBC_TO_DXIL_CONVERSION_FAILED;
  } catch (...) {
    SetError(error, error_size, "atomic lowering failed");
    return SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR;
  }
}

extern "C" int ScorpiPromoteTexture1D(const void *code, size_t size,
    void **binary, size_t *binary_size, char *error, size_t error_size) {
  return ScorpiLowerTextureLayouts(code,size,0,nullptr,0,binary,binary_size,error,error_size);
}

extern "C" int ScorpiLowerTextureLayouts(const void *code, size_t size, int split_msaa8,
    const ScorpiSamplingRequirement *policy, size_t policy_count,
    void **binary, size_t *binary_size, char *error, size_t error_size) {
  SetError(error, error_size, "");
  if (!binary || !binary_size || !code || !size || (policy_count && !policy))
    return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  *binary = nullptr; *binary_size = 0;
  try {
    DxcMallocScope malloc_scope;
    auto normalized = ConvertToDxilBinary(code, size, nullptr, 0);
    auto *container = reinterpret_cast<const hlsl::DxilContainerHeader *>(normalized.data());
    if (!hlsl::IsValidDxilContainer(container, normalized.size()))
      throw std::runtime_error("invalid texture shader container");
    auto *program = hlsl::GetDxilProgramHeader(container, hlsl::DFCC_DXIL);
    if (!program) throw std::runtime_error("texture shader has no DXIL program");
    llvm::LLVMContext context;
    auto source = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(
        hlsl::GetDxilBitcodeData(program), hlsl::GetDxilBitcodeSize(program)),
        "texture-1d", false);
    auto parsed = llvm::parseBitcodeFile(source->getMemBufferRef(), context);
    if (!parsed) throw std::runtime_error("invalid texture shader bitcode");
    auto *dm = hlsl::DxilModule::TryGetDxilModule(parsed.get().get());
    if (!dm) throw std::runtime_error("missing texture shader module");
    auto promoted = [](TextureKind kind) {
      return kind == TextureKind::Texture1D ? TextureKind::Texture2D :
          kind == TextureKind::Texture1DArray ? TextureKind::Texture2DArray : kind;
    };
    bool changed = false;
    if (split_msaa8) {
      std::string message;
      if (!scorpi::PrepareMsaa8TextureLoads(*dm,message,changed,policy,policy_count)) throw std::runtime_error(message);
    }
    for (const auto &resource : dm->GetSRVs()) changed |= promoted(resource->GetKind()) != resource->GetKind();
    for (const auto &resource : dm->GetUAVs()) changed |= promoted(resource->GetKind()) != resource->GetKind();
    if (!changed) return SCORPI_DXBC_TO_DXIL_OK;

    std::vector<llvm::CallInst *> annotated;
    for (auto &function : *parsed.get()) for (auto &block : function)
      for (auto &instruction : block) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
        if (!call || !call->getNumArgOperands()) continue;
        auto *opcodeValue = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0));
        if (!opcodeValue || !call->getCalledFunction() ||
            !call->getCalledFunction()->getName().startswith("dx.op.")) continue;
        const auto opcode = static_cast<hlsl::DXIL::OpCode>(opcodeValue->getZExtValue());
        if (opcode == hlsl::DXIL::OpCode::AnnotateHandle) {
          annotated.push_back(call);
          continue;
        }
        if (call->getNumArgOperands() < 2) continue;
        const auto kind = OriginalTextureKind(call->getArgOperand(1), *dm);
        if (kind != TextureKind::Texture1D && kind != TextureKind::Texture1DArray) continue;
        const bool array = kind == TextureKind::Texture1DArray;
        const auto zero = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 0);
        const auto half = llvm::ConstantFP::get(llvm::Type::getFloatTy(context), 0.5);
        switch (opcode) {
        case hlsl::DXIL::OpCode::Sample:
        case hlsl::DXIL::OpCode::SampleBias:
        case hlsl::DXIL::OpCode::SampleLevel:
        case hlsl::DXIL::OpCode::SampleGrad:
        case hlsl::DXIL::OpCode::SampleCmp:
        case hlsl::DXIL::OpCode::SampleCmpLevelZero:
        case hlsl::DXIL::OpCode::TextureGather:
        case hlsl::DXIL::OpCode::TextureGatherCmp:
          if (array) call->setArgOperand(5, call->getArgOperand(4));
          call->setArgOperand(4, half);
          call->setArgOperand(8, zero);
          if (opcode == hlsl::DXIL::OpCode::SampleGrad) {
            call->setArgOperand(11, llvm::ConstantFP::get(llvm::Type::getFloatTy(context), 0));
            call->setArgOperand(14, llvm::ConstantFP::get(llvm::Type::getFloatTy(context), 0));
          }
          break;
        case hlsl::DXIL::OpCode::TextureLoad:
          if (array) call->setArgOperand(5, call->getArgOperand(4));
          call->setArgOperand(4, zero);
          call->setArgOperand(7, zero);
          break;
        case hlsl::DXIL::OpCode::TextureStore:
          if (array) call->setArgOperand(4, call->getArgOperand(3));
          call->setArgOperand(3, zero);
          break;
        case hlsl::DXIL::OpCode::AtomicBinOp:
          if (array) call->setArgOperand(5, call->getArgOperand(4));
          call->setArgOperand(4, zero);
          break;
        case hlsl::DXIL::OpCode::AtomicCompareExchange:
          if (array) call->setArgOperand(4, call->getArgOperand(3));
          call->setArgOperand(3, zero);
          break;
        case hlsl::DXIL::OpCode::CalculateLOD:
          if (array) call->setArgOperand(5, call->getArgOperand(4));
          call->setArgOperand(4, half);
          break;
        case hlsl::DXIL::OpCode::GetDimensions: {
          std::vector<llvm::ExtractValueInst *> extracts;
          for (auto *user : call->users()) {
            auto *extract = llvm::dyn_cast<llvm::ExtractValueInst>(user);
            if (!extract) throw std::runtime_error("unsupported 1D dimensions consumer");
            extracts.push_back(extract);
          }
          for (auto *extract : extracts) {
            if (extract->getNumIndices() != 1) throw std::runtime_error("unsupported 1D dimensions result");
            unsigned index = *extract->idx_begin();
            // DXBC resinfo requires zero for dimensions that do not exist.
            // The physical 2D height and mip count must not leak into them.
            if (index == 2 || (!array && index == 1)) {
              extract->replaceAllUsesWith(zero);
              extract->eraseFromParent();
              continue;
            }
            if (array && index == 1) index = 2;
            if (index == *extract->idx_begin()) continue;
            auto *replacement = llvm::ExtractValueInst::Create(call, {index}, "", extract);
            extract->replaceAllUsesWith(replacement);
            extract->eraseFromParent();
          }
          break;
        }
        default:
          throw std::runtime_error("unsupported 1D texture DXIL operation");
        }
      }
    for (auto *call : annotated)
      if (auto *constant = llvm::dyn_cast<llvm::Constant>(call->getArgOperand(2))) {
        auto properties = hlsl::resource_helper::loadPropsFromConstant(*constant);
        const auto kind = promoted(properties.getResourceKind());
        if (kind != properties.getResourceKind()) {
          properties.setResourceKind(kind);
          call->setArgOperand(2, hlsl::resource_helper::getAsConstant(
              properties, dm->GetOP()->GetResourcePropertiesType(), *dm->GetShaderModel()));
        }
      }
    for (auto &resource : dm->GetSRVs()) resource->SetKind(promoted(resource->GetKind()));
    for (auto &resource : dm->GetUAVs()) resource->SetKind(promoted(resource->GetKind()));
    if (llvm::verifyModule(*parsed.get())) throw std::runtime_error("invalid promoted texture IR");
    hlsl::DxilModule::ClearDxilMetadata(*parsed.get());
    dm->EmitDxilMetadata();
    CComPtr<hlsl::AbstractMemoryStream> bitcode, output;
    IFT(hlsl::CreateMemoryStream(DxcGetThreadMallocNoRef(), &bitcode));
    IFT(hlsl::CreateMemoryStream(DxcGetThreadMallocNoRef(), &output));
    { raw_stream_ostream stream(bitcode.p); llvm::WriteBitcodeToFile(parsed.get().get(), stream); }
    hlsl::SerializeDxilContainerForModule(dm, bitcode, nullptr, output,
        llvm::StringRef(), hlsl::SerializeDxilFlags::None);
    auto *result = std::malloc(output->GetPtrSize());
    if (!result) throw std::runtime_error("failed to allocate promoted texture shader");
    std::memcpy(result, output->GetPtr(), output->GetPtrSize());
    auto *header = static_cast<hlsl::DxilContainerHeader *>(result);
    constexpr uint32_t hashOffset = static_cast<uint32_t>(offsetof(hlsl::DxilContainerHeader, Version));
    ComputeHashRetail(static_cast<const uint8_t *>(result) + hashOffset,
        header->ContainerSizeInBytes - hashOffset, header->Hash.Digest);
    *binary = result; *binary_size = output->GetPtrSize();
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (const std::exception &e) {
    SetError(error, error_size, e.what());
    return SCORPI_DXBC_TO_DXIL_CONVERSION_FAILED;
  } catch (...) {
    SetError(error, error_size, "unexpected 1D texture promotion failure");
    return SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR;
  }
}

static int CreateGeometryCaptureBinary(
    const void *code, size_t size, uint32_t register_space,
    uint32_t vertices_per_primitive, int capture, uint32_t *vertex_stride, void **binary,
    size_t *binary_size, char *error_message, size_t error_message_size,
    ScorpiGeometryCaptureLayout *layout, bool compacted = false, unsigned tessellation_mode = 0,
    uint32_t rasterized_stream = 0) {
  SetError(error_message, error_message_size, "");
  if (!binary || !binary_size || !vertex_stride) return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  *binary = nullptr;
  *binary_size = 0;
  const char *phase = "parse source";
  try {
    DxcMallocScope malloc_scope;
    auto normalized = ConvertToDxilBinary(code, size, nullptr, 0);
    auto *container = reinterpret_cast<const hlsl::DxilContainerHeader *>(normalized.data());
    if (!hlsl::IsValidDxilContainer(container, normalized.size()))
      throw std::runtime_error("invalid replay shader container");
    auto *program = hlsl::GetDxilProgramHeader(container, hlsl::DFCC_DXIL);
    if (!program) throw std::runtime_error("replay shader has no DXIL program");
    llvm::LLVMContext context;
    auto buffer = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(hlsl::GetDxilBitcodeData(program),
                        hlsl::GetDxilBitcodeSize(program)), "raster-replay", false);
    auto parsed = llvm::parseBitcodeFile(buffer->getMemBufferRef(), context);
    if (!parsed) throw std::runtime_error("invalid replay shader bitcode");
    auto *dxil = hlsl::DxilModule::TryGetDxilModule(parsed.get().get());
    std::string error;
    if (!dxil) throw std::runtime_error("missing DXIL module");
    phase = "inline legacy subroutines";
    ScorpiInlineDxbcSubroutines(*dxil);
    phase = "generate capture IR";
    std::unique_ptr<llvm::Module> generated;
    if (tessellation_mode == 7) {
      if(!scorpi::PrepareTessellationPixelPatchId(*dxil,register_space,error))throw std::runtime_error(error);
      generated=std::move(parsed.get());
    } else if (tessellation_mode == 5) {
      if(!scorpi::PrepareTessellationPatchBias(*dxil,register_space,error))throw std::runtime_error(error);
      generated=std::move(parsed.get());
    } else if (tessellation_mode == 1) {
      if(!scorpi::PrepareTessellationPatchBias(*dxil,register_space,error))throw std::runtime_error(error);
      if (!scorpi::PrepareTessellationCaptureDomain(*dxil,*vertex_stride,error,register_space)) throw std::runtime_error(error);
      generated=std::move(parsed.get());
    } else if (tessellation_mode == 4) {
      generated=scorpi::CreateGeometryCaptureModule(*dxil,register_space,vertices_per_primitive,true,
          *vertex_stride,error,nullptr,false,false,true);
    } else if (tessellation_mode == 3) {
      ScorpiGeometryCaptureLayout replay{};
      replay.max_vertices=vertices_per_primitive;replay.instance_count=1;replay.vertices_per_primitive=vertices_per_primitive;
      replay.stream_mask=1;
      for(const auto &e:dxil->GetOutputSignature().GetElements()) replay.vertex_stride+=e->GetRows()*e->GetCols()*4;
      generated=scorpi::CreateGeometryCaptureModule(*dxil,register_space,vertices_per_primitive,false,
          *vertex_stride,error,&replay,true,true);
    } else if (tessellation_mode == 2 || tessellation_mode == 6) {
      generated=scorpi::CreateGeometryCaptureModule(*dxil,register_space,vertices_per_primitive,true,
          *vertex_stride,error,nullptr,false,true,false,tessellation_mode == 6);
    } else if (layout) {
      if (!scorpi::GetGeometryCaptureLayout(*dxil, *layout, error)) throw std::runtime_error(error);
      *vertex_stride = layout->vertex_stride;
      if (capture) {
        if (!scorpi::PrepareGeometryShaderCapture(*dxil, register_space, *layout, error,
            capture == 2 || capture == 3, capture == 1 || capture == 2)) throw std::runtime_error(error);
        generated = std::move(parsed.get());
      } else generated = scorpi::CreateGeometryCaptureModule(*dxil, register_space,
          layout->vertices_per_primitive, false, *vertex_stride, error, layout,
          compacted, false, false, false, rasterized_stream);
    } else generated = scorpi::CreateGeometryCaptureModule(*dxil, register_space,
        vertices_per_primitive, capture != 0, *vertex_stride, error);
    if (!generated) throw std::runtime_error(error);
    dxil = &generated->GetOrCreateDxilModule();
    hlsl::DxilModule::ClearDxilMetadata(*generated);
    phase = "emit capture metadata";
    dxil->EmitDxilMetadata();
    CComPtr<hlsl::AbstractMemoryStream> bitcode, output;
    IFT(hlsl::CreateMemoryStream(DxcGetThreadMallocNoRef(), &bitcode));
    IFT(hlsl::CreateMemoryStream(DxcGetThreadMallocNoRef(), &output));
    {
      raw_stream_ostream stream(bitcode.p);
      llvm::WriteBitcodeToFile(generated.get(), stream);
    }
    phase = "serialize capture container";
    hlsl::SerializeDxilContainerForModule(dxil, bitcode, nullptr, output,
        llvm::StringRef(), hlsl::SerializeDxilFlags::None);
    void *result = std::malloc(output->GetPtrSize());
    if (!result) throw std::runtime_error("failed to allocate replay shader");
    std::memcpy(result, output->GetPtr(), output->GetPtrSize());
    auto *header = static_cast<hlsl::DxilContainerHeader *>(result);
    constexpr uint32_t hashOffset =
        static_cast<uint32_t>(offsetof(hlsl::DxilContainerHeader, Version));
    ComputeHashRetail(static_cast<const uint8_t *>(result) + hashOffset,
        header->ContainerSizeInBytes - hashOffset, header->Hash.Digest);
    *binary = result;
    *binary_size = output->GetPtrSize();
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (const std::exception &e) {
    SetError(error_message, error_message_size, (std::string(phase) + ": " + e.what()).c_str());
    return SCORPI_DXBC_TO_DXIL_CONVERSION_FAILED;
  } catch (...) {
    SetError(error_message, error_message_size, "unexpected raster replay conversion failure");
    return SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR;
  }
}

extern "C" int ScorpiCreateTessellationCaptureShader(
    const void *code, size_t size, uint32_t space, uint32_t vertices, uint32_t mode,
    uint32_t *stride, void **binary, size_t *binary_size, char *error, size_t error_size) {
  if(mode < 1 || mode > 7) return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  return CreateGeometryCaptureBinary(code,size,space,vertices,1,stride,binary,binary_size,error,error_size,nullptr,false,mode);
}

extern "C" int ScorpiCreateGeometryCaptureShader(
    const void *code, size_t size, uint32_t space, uint32_t vertices,
    int capture, uint32_t *stride, void **binary, size_t *binary_size,
    char *error, size_t error_size) {
  return CreateGeometryCaptureBinary(code, size, space, vertices, capture,
      stride, binary, binary_size, error, error_size, nullptr);
}

extern "C" int ScorpiCreateGeometryShaderCapture(
    const void *code, size_t size, uint32_t space, int capture,
    ScorpiGeometryCaptureLayout *layout, void **binary, size_t *binary_size,
    char *error, size_t error_size) {
  if (!layout) return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  uint32_t stride = 0;
  return CreateGeometryCaptureBinary(code, size, space, 0, capture,
      &stride, binary, binary_size, error, error_size, layout);
}

extern "C" int ScorpiCreateCompactedGeometryReplayShader(
    const void *code, size_t size, uint32_t space, ScorpiGeometryCaptureLayout *layout,
    void **binary, size_t *binary_size, char *error, size_t error_size) {
  if (!layout) return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  uint32_t stride = 0;
  return CreateGeometryCaptureBinary(code, size, space, 0, 0, &stride,
      binary, binary_size, error, error_size, layout, true);
}

extern "C" int ScorpiCreateCompactedGeometryReplayShaderForStream(
    const void *code, size_t size, uint32_t space, uint32_t rasterized_stream,
    ScorpiGeometryCaptureLayout *layout, void **binary, size_t *binary_size,
    char *error, size_t error_size) {
  if (!layout) return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  uint32_t stride = 0;
  return CreateGeometryCaptureBinary(code, size, space, 0, 0, &stride,
      binary, binary_size, error, error_size, layout, true, 0, rasterized_stream);
}

extern "C" int ScorpiGetGeometryCaptureElements(const void *code, size_t size,
    ScorpiGeometryCaptureElement *elements, uint32_t capacity, uint32_t *count,
    char *error, size_t error_size) {
  SetError(error, error_size, "");
  if (!count || (capacity && !elements)) return SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT;
  *count = 0;
  try {
    DxcMallocScope malloc_scope;
    auto binary = ConvertToDxilBinary(code, size, nullptr, 0);
    auto *container = reinterpret_cast<const hlsl::DxilContainerHeader *>(binary.data());
    if (!hlsl::IsValidDxilContainer(container, binary.size())) throw std::runtime_error("invalid capture container");
    auto *program = hlsl::GetDxilProgramHeader(container, hlsl::DFCC_DXIL);
    if (!program) throw std::runtime_error("missing capture DXIL program");
    llvm::LLVMContext context;
    auto buffer = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(
        hlsl::GetDxilBitcodeData(program), hlsl::GetDxilBitcodeSize(program)), "capture-signature", false);
    auto parsed = llvm::parseBitcodeFile(buffer->getMemBufferRef(), context);
    if (!parsed) throw std::runtime_error("invalid capture bitcode");
    auto *dm = hlsl::DxilModule::TryGetDxilModule(parsed.get().get());
    if (!dm) throw std::runtime_error("missing capture module");
    std::vector<ScorpiGeometryCaptureElement> rows;
    uint32_t offset = 0;
    for (const auto &element : dm->GetOutputSignature().GetElements()) {
      const auto &e = *element;
      if (e.GetCompType().Is64Bit() || e.GetCompType().Is16Bit())
        throw std::runtime_error("capture requires 32-bit signature storage");
      if (std::strlen(e.GetName()) >= sizeof(ScorpiGeometryCaptureElement::semantic_name))
        throw std::runtime_error("capture semantic name too long");
      for (uint32_t row = 0; row < e.GetRows(); ++row) {
        ScorpiGeometryCaptureElement out{};
        std::memcpy(out.semantic_name, e.GetName(), std::strlen(e.GetName()));
        out.semantic_index = e.GetSemanticIndexVec().at(row);
        out.register_index = e.GetStartRow() + row;
        out.start_component = e.GetStartCol(); out.component_count = e.GetCols();
        out.byte_offset = offset; out.stream = e.GetOutputStream();
        rows.push_back(out); offset += e.GetCols() * 4;
      }
    }
    *count = static_cast<uint32_t>(rows.size());
    if (capacity < rows.size()) {
      if (!capacity) return SCORPI_DXBC_TO_DXIL_OK;
      throw std::runtime_error("capture element capacity too small");
    }
    std::copy(rows.begin(), rows.end(), elements);
    return SCORPI_DXBC_TO_DXIL_OK;
  } catch (const std::exception &e) { SetError(error, error_size, e.what()); }
  catch (...) { SetError(error, error_size, "unexpected capture signature exception"); }
  return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
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
    if (dxil->GetShaderModel()->IsHS()) return SCORPI_COVERAGE_REPLAY_SAFE_HULL;
    if (dxil->GetShaderModel()->IsDS()) return SCORPI_COVERAGE_REPLAY_SAFE_DOMAIN;
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

// A symbolic bit map is stronger than evaluating a few test inputs: it proves
// both union and parity are preserved for every coverage mask and primitive.
extern "C" uint32_t ScorpiInspectCoverageTransform(const void *code, size_t size,
    const void *metadata, size_t metadata_size, uint32_t samples,
    uint32_t output_masks[16]) {
  if (!output_masks || (samples != 4 && samples != 8 && samples != 16)) return 0;
  try {
    DxcMallocScope malloc_scope;
    auto binary = ConvertToDxilBinary(code, size, metadata, metadata_size);
    auto *container = reinterpret_cast<const hlsl::DxilContainerHeader *>(binary.data());
    if (!hlsl::IsValidDxilContainer(container, binary.size())) return 0;
    auto *program = hlsl::GetDxilProgramHeader(container, hlsl::DFCC_DXIL);
    if (!program) return 0;
    llvm::LLVMContext context;
    auto buffer = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(
        hlsl::GetDxilBitcodeData(program), hlsl::GetDxilBitcodeSize(program)), "coverage-transform", false);
    auto parsed = llvm::parseBitcodeFile(buffer->getMemBufferRef(), context);
    if (!parsed) return 0;
    auto *dxil = hlsl::DxilModule::TryGetDxilModule(parsed.get().get());
    if (!dxil || !dxil->GetShaderModel()->IsPS() || !dxil->GetUAVs().empty() ||
        dxil->m_ShaderFlags.GetResourceDescriptorHeapIndexing()) return 0;
    const auto &outputs = dxil->GetOutputSignature().GetElements();
    if (outputs.size() != 1 || outputs[0]->GetKind() != hlsl::DXIL::SemanticKind::Target ||
        outputs[0]->GetSemanticStartIndex() != 0 || outputs[0]->GetRows() != 1 ||
        outputs[0]->GetCompType().GetKind() != hlsl::DXIL::ComponentType::U32) return 0;
    auto *entry = dxil->GetEntryFunction();
    if (!entry || entry->size() != 1) return 0;
    using Bits = std::array<int, 32>; // -1 means zero; otherwise input bit index.
    Bits zero; zero.fill(-1);
    Bits red = zero;
    std::vector<std::pair<llvm::Value *, Bits>> values;
    auto find = [&](llvm::Value *value, Bits &bits) {
      if (auto *c = llvm::dyn_cast<llvm::ConstantInt>(value)) {
        if (c->isZero()) { bits = zero; return true; }
        return false;
      }
      for (const auto &known : values) if (known.first == value) { bits = known.second; return true; }
      return false;
    };
    auto constant = [](llvm::Value *value, uint64_t expected) {
      auto *c = llvm::dyn_cast<llvm::ConstantInt>(value);
      return c && c->getZExtValue() == expected;
    };
    unsigned components = 0;
    bool coverage = false;
    for (auto &inst : entry->front()) {
      if (llvm::isa<llvm::ReturnInst>(inst)) continue;
      if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
        if (!call->getCalledFunction()) return 0;
        auto name = call->getCalledFunction()->getName();
        if (name == "dx.op.coverage.i32" && call->getNumArgOperands() == 1 &&
            constant(call->getArgOperand(0), 91)) {
          Bits bits = zero;
          for (unsigned bit = 0; bit < samples; ++bit) bits[bit] = int(bit);
          values.emplace_back(call, bits); coverage = true;
        } else if (name == "dx.op.storeOutput.i32" && call->getNumArgOperands() == 5 &&
            constant(call->getArgOperand(0), 5) && constant(call->getArgOperand(1), 0) &&
            constant(call->getArgOperand(2), 0)) {
          auto *column = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(3));
          Bits bits;
          if (!column || column->getZExtValue() > 3 || !find(call->getArgOperand(4), bits)) return 0;
          unsigned index = unsigned(column->getZExtValue());
          if (components & (1u << index)) return 0;
          components |= 1u << index;
          if (!index) red = bits;
        } else return 0;
        continue;
      }
      auto *op = llvm::dyn_cast<llvm::BinaryOperator>(&inst);
      if (!op || !op->getType()->isIntegerTy(32)) return 0;
      Bits a, b, result = zero;
      auto *amount = llvm::dyn_cast<llvm::ConstantInt>(op->getOperand(1));
      if ((op->getOpcode() == llvm::Instruction::Shl || op->getOpcode() == llvm::Instruction::LShr) &&
          amount && amount->getZExtValue() < 32 && find(op->getOperand(0), a)) {
        int shift = int(amount->getZExtValue());
        for (int bit = 0; bit < 32; ++bit) {
          int source = bit + (op->getOpcode() == llvm::Instruction::Shl ? -shift : shift);
          if (source >= 0 && source < 32) result[bit] = a[source];
        }
      } else if (op->getOpcode() == llvm::Instruction::And && amount && find(op->getOperand(0), a)) {
        for (unsigned bit = 0; bit < 32; ++bit)
          if (amount->getZExtValue() & (1ull << bit)) result[bit] = a[bit];
      } else if ((op->getOpcode() == llvm::Instruction::Or || op->getOpcode() == llvm::Instruction::Xor) &&
          find(op->getOperand(0), a) && find(op->getOperand(1), b)) {
        for (unsigned bit = 0; bit < 32; ++bit) {
          if (a[bit] >= 0 && b[bit] >= 0 && a[bit] != b[bit]) return 0;
          result[bit] = op->getOpcode() == llvm::Instruction::Xor && a[bit] == b[bit] ?
              -1 : std::max(a[bit], b[bit]);
        }
      } else return 0;
      values.emplace_back(op, result);
    }
    if (!coverage || !(components & 1)) return 0;
    for (int bit = 0; bit < 4; ++bit) if (red[bit] != bit) return 0;
    uint32_t masks[16] = {};
    for (unsigned bit = 0; bit < 32; ++bit)
      if (red[bit] >= 0) masks[red[bit]] |= uint32_t(1) << bit;
    std::copy(masks, masks + 16, output_masks);
    return 1;
  } catch (...) { return 0; }
}
