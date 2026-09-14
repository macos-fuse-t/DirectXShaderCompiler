#include "llvm/Support/Debug.h" // Must be included first.

#define SCORPI_DXBC_TO_DXIL_BUILD_DLL
#include "ScorpiDxbcToDxil.h"
#include "ScorpiRasterReplay.h"
#include "ScorpiGeometryCapture.h"

#include "DxbcConverterImpl.h"
#include "dxc/DxilHash/DxilHash.h"
#include "dxc/DxilContainer/DxilContainer.h"
#include "dxc/DxilContainer/DxilContainerAssembler.h"
#include "dxc/Support/dxcapi.impl.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilShaderModel.h"
#include "dxc/DXIL/DxilSignature.h"
#include "dxc/DXIL/DxilSignatureElement.h"
#include "dxc/DXIL/DxilResource.h"
#include "dxc/DXIL/DxilResourceProperties.h"
#include "dxc/DXIL/DxilOperations.h"
#include "llvm/IR/Verifier.h"

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

  if (metadata->flags & ~SCORPI_SHADER_SIGNATURE_STREAM_OUTPUT_ONLY)
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

static int CreateRasterReplayShader(
    const void *code, size_t size, uint32_t register_space,
    uint32_t sample_count, uint32_t sample_mask, void **binary,
    size_t *binary_size, char *error_message, size_t error_message_size, bool vertex, bool generated = false, bool geometry_link = false) {
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
        : vertex ? scorpi::PrepareRasterReplayVertexShader(*dxil, register_space, error)
        : scorpi::PrepareRasterReplayPixelShader(*dxil, register_space, sample_count, sample_mask, error, generated)))
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

extern "C" int ScorpiPromoteTexture1D(const void *code, size_t size,
    void **binary, size_t *binary_size, char *error, size_t error_size) {
  SetError(error, error_size, "");
  if (!binary || !binary_size || !code || !size)
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
            if (array && index == 1) index = 2;
            else if (array && index == 2) index = 3;
            else if (!array && index == 1) index = 3;
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
