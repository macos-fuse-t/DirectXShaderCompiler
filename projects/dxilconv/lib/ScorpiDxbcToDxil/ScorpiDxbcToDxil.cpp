#include "ScorpiDxbcToDxil.h"

#include "DxbcConverterImpl.h"
#include "dxc/DxilContainer/DxilContainer.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class DxcMallocScope {
public:
  DxcMallocScope() {
    if (FAILED(DxcInitThreadMalloc())) {
      throw std::runtime_error("failed to initialize DXC allocator");
    }
    DxcSetThreadMallocToDefault();
  }

  ~DxcMallocScope() { DxcCleanupThreadMalloc(); }

  DxcMallocScope(const DxcMallocScope &) = delete;
  DxcMallocScope &operator=(const DxcMallocScope &) = delete;
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

class ScorpiDxbcConverter : public hlsl::DxbcConverter {
public:
  HRESULT ConvertRawInDriver(const UINT32 *bytecode, const void *input,
                             uint32_t input_count, const void *output,
                             uint32_t output_count, const void *patch,
                             uint32_t patch_count, IDxcBlob **dxil,
                             LPWSTR *diagnostics) {
    DxcThreadMalloc thread_malloc(nullptr);
    HRESULT hr = S_OK;
    try {
      ConvertInDriverImpl(
          bytecode,
          static_cast<const D3D12DDIARG_SIGNATURE_ENTRY_0012 *>(input),
          input_count,
          static_cast<const D3D12DDIARG_SIGNATURE_ENTRY_0012 *>(output),
          output_count,
          static_cast<const D3D12DDIARG_SIGNATURE_ENTRY_0012 *>(patch),
          patch_count, nullptr, dxil, diagnostics);
    }
    CATCH_CPP_ASSIGN_HRESULT();
    return hr;
  }
};

struct ScorpiShaderMetadata {
  uint32_t version;
  uint32_t flags;
  uint32_t signature_entry_size;
  uint32_t input_count;
  uint32_t output_count;
  uint32_t patch_count;
  uint32_t input_offset;
  uint32_t output_offset;
  uint32_t patch_offset;
  uint32_t reserved;
};

struct SignatureView {
  const void *input = nullptr;
  const void *output = nullptr;
  const void *patch = nullptr;
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

const void *SignaturePointer(const uint8_t *data, size_t data_size,
                             uint32_t offset, uint32_t count,
                             uint32_t entry_size) {
  if (count == 0) {
    return nullptr;
  }

  const uint64_t bytes = static_cast<uint64_t>(count) * entry_size;
  if (offset > data_size || bytes > data_size - offset) {
    throw std::runtime_error("signature range exceeds signature dump");
  }
  return data + offset;
}

SignatureView ParseSignatures(const void *signature_dump,
                              size_t signature_dump_size) {
  SignatureView view = {};

  if (signature_dump == nullptr || signature_dump_size == 0) {
    return view;
  }
  if (signature_dump_size < sizeof(ScorpiShaderMetadata)) {
    throw std::runtime_error("signature dump is missing metadata");
  }

  const auto *data = static_cast<const uint8_t *>(signature_dump);
  const auto *metadata = reinterpret_cast<const ScorpiShaderMetadata *>(data);
  if (metadata->version != 1) {
    throw std::runtime_error("unsupported signature metadata version");
  }
  if (metadata->signature_entry_size !=
      sizeof(D3D12DDIARG_SIGNATURE_ENTRY_0012)) {
    throw std::runtime_error("signature entry size does not match DXC");
  }

  view.input_count = metadata->input_count;
  view.output_count = metadata->output_count;
  view.patch_count = metadata->patch_count;
  view.input = SignaturePointer(data, signature_dump_size,
                                metadata->input_offset,
                                metadata->input_count,
                                metadata->signature_entry_size);
  view.output = SignaturePointer(data, signature_dump_size,
                                 metadata->output_offset,
                                 metadata->output_count,
                                 metadata->signature_entry_size);
  view.patch = SignaturePointer(data, signature_dump_size,
                                metadata->patch_offset,
                                metadata->patch_count,
                                metadata->signature_entry_size);
  return view;
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

D3D10_SB_TOKENIZED_PROGRAM_TYPE ShaderTypeFromTokenStream(
    const uint32_t *token_stream) {
  return DECODE_D3D10_SB_TOKENIZED_PROGRAM_TYPE(token_stream[0]);
}

const D3D12DDIARG_SIGNATURE_ENTRY_0012 *SignatureEntries(
    const void *entries) {
  return static_cast<const D3D12DDIARG_SIGNATURE_ENTRY_0012 *>(entries);
}

bool IsEmptySignatureRecord(const D3D12DDIARG_SIGNATURE_ENTRY_0012 &entry) {
  return entry.Mask == 0 ||
         (entry.SystemValue == D3D10_SB_NAME_UNDEFINED &&
          entry.RegisterComponentType == D3D10_SB_REGISTER_COMPONENT_UNKNOWN);
}

bool IsOrdinaryLinkageRecord(const D3D12DDIARG_SIGNATURE_ENTRY_0012 &entry) {
  return entry.SystemValue == D3D10_SB_NAME_UNDEFINED &&
         entry.RegisterComponentType != D3D10_SB_REGISTER_COMPONENT_UNKNOWN;
}

bool ShouldKeepDdiSignatureRecord(
    D3D10_SB_TOKENIZED_PROGRAM_TYPE shader_type, bool output_signature,
    const D3D12DDIARG_SIGNATURE_ENTRY_0012 &entry) {
  if (IsEmptySignatureRecord(entry)) {
    return false;
  }
  if (!IsOrdinaryLinkageRecord(entry)) {
    return true;
  }

  // Ordinary DDI entries do not carry semantic names. If they are fed directly
  // to ConvertInDriver, input and output signatures synthesize different
  // arbitrary names, breaking VS/PS linkage. Let the converter rebuild ordinary
  // linkage records from DXBC declarations instead. PS outputs are special:
  // ConvertInDriver can infer SV_Target/depth/coverage from the output register
  // and declaration state, so those records are preserved.
  return output_signature && shader_type == D3D10_SB_PIXEL_SHADER;
}

std::vector<D3D12DDIARG_SIGNATURE_ENTRY_0012> FilterSignatureRecords(
    D3D10_SB_TOKENIZED_PROGRAM_TYPE shader_type, bool output_signature,
    const void *records, uint32_t count) {
  std::vector<D3D12DDIARG_SIGNATURE_ENTRY_0012> filtered;
  const auto *entries = SignatureEntries(records);
  for (uint32_t index = 0; index < count; ++index) {
    if (ShouldKeepDdiSignatureRecord(shader_type, output_signature,
                                     entries[index])) {
      filtered.emplace_back(entries[index]);
    }
  }
  return filtered;
}

SignatureView NormalizeSignaturesForRawTokenConversion(
    const uint32_t *token_stream, const SignatureView &signatures,
    SignatureStorage *storage) {
  const D3D10_SB_TOKENIZED_PROGRAM_TYPE shader_type =
      ShaderTypeFromTokenStream(token_stream);
  storage->input = FilterSignatureRecords(shader_type, false, signatures.input,
                                          signatures.input_count);
  storage->output = FilterSignatureRecords(shader_type, true, signatures.output,
                                           signatures.output_count);
  storage->patch = FilterSignatureRecords(shader_type, false, signatures.patch,
                                          signatures.patch_count);
  return storage->View();
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

int WriteExistingDxilFile(const void *shader_code, size_t shader_code_size,
                          const char *output_path) {
  if (IsDxilContainer(shader_code, shader_code_size)) {
    WriteFile(output_path, shader_code, shader_code_size);
    return SCORPI_DXBC_TO_DXIL_OK;
  }

  if (IsDxilProgram(shader_code, shader_code_size)) {
    const auto *program =
        static_cast<const hlsl::DxilProgramHeader *>(shader_code);
    const size_t program_size =
        static_cast<size_t>(program->SizeInUint32) * sizeof(uint32_t);
    const std::vector<uint8_t> dxil_container =
        WrapDxilContainer(std::vector<uint8_t>(
            static_cast<const uint8_t *>(shader_code),
            static_cast<const uint8_t *>(shader_code) + program_size));
    WriteFile(output_path, dxil_container.data(), dxil_container.size());
    return SCORPI_DXBC_TO_DXIL_OK;
  }

  return SCORPI_DXBC_TO_DXIL_INVALID_SHADER;
}

int ConvertToDxilFile(const void *shader_code, size_t shader_code_size,
                      const void *signature_dump,
                      size_t signature_dump_size, const char *output_path) {
  if (output_path == nullptr || output_path[0] == '\0') {
    throw std::invalid_argument("output path is empty");
  }
  if (shader_code == nullptr || shader_code_size == 0) {
    throw std::invalid_argument("shader code is empty");
  }

  const int existing_status =
      WriteExistingDxilFile(shader_code, shader_code_size, output_path);
  if (existing_status == SCORPI_DXBC_TO_DXIL_OK) {
    return existing_status;
  }

  const uint32_t *token_stream =
      ValidateShaderTokenStream(shader_code, shader_code_size);
  const SignatureView parsed_signatures =
      ParseSignatures(signature_dump, signature_dump_size);
  SignatureStorage normalized_signatures;
  const SignatureView signatures = NormalizeSignaturesForRawTokenConversion(
      token_stream, parsed_signatures, &normalized_signatures);

  DxcMallocScope dxc_malloc;

  ScorpiDxbcConverter converter;
  DxcBlobScope dxil;
  LPWSTR diagnostics = nullptr;
  const HRESULT status = converter.ConvertRawInDriver(
      token_stream, signatures.input, signatures.input_count,
      signatures.output, signatures.output_count, signatures.patch,
      signatures.patch_count, dxil.Out(), &diagnostics);

  if (FAILED(status)) {
    return SCORPI_DXBC_TO_DXIL_CONVERSION_FAILED;
  }
  if (dxil.Get() == nullptr) {
    throw std::runtime_error("conversion succeeded without DXIL blob");
  }

  const std::vector<uint8_t> dxil_program =
      WrapDxilProgram(dxil.Get()->GetBufferPointer(), dxil.Get()->GetBufferSize(),
                      token_stream[0]);
  const std::vector<uint8_t> dxil_container =
      WrapDxilContainer(dxil_program);

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
