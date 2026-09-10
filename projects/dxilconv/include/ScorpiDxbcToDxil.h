#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(SCORPI_DXBC_TO_DXIL_STATIC)
#define SCORPI_DXBC_TO_DXIL_EXPORT
#elif defined(_WIN32) && defined(SCORPI_DXBC_TO_DXIL_BUILD_DLL)
#define SCORPI_DXBC_TO_DXIL_EXPORT __declspec(dllexport)
#elif defined(_WIN32)
#define SCORPI_DXBC_TO_DXIL_EXPORT __declspec(dllimport)
#else
#define SCORPI_DXBC_TO_DXIL_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum ScorpiDxbcToDxilStatus {
  SCORPI_DXBC_TO_DXIL_OK = 0,
  SCORPI_DXBC_TO_DXIL_INVALID_ARGUMENT = 1,
  SCORPI_DXBC_TO_DXIL_INVALID_SHADER = 2,
  SCORPI_DXBC_TO_DXIL_INVALID_SIGNATURES = 3,
  SCORPI_DXBC_TO_DXIL_CONVERSION_FAILED = 4,
  SCORPI_DXBC_TO_DXIL_IO_FAILED = 5,
  SCORPI_DXBC_TO_DXIL_INTERNAL_ERROR = 6,
};

enum {
  SCORPI_SHADER_SIGNATURE_METADATA_VERSION = 2,
};

typedef struct ScorpiShaderSignatureMetadata {
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
} ScorpiShaderSignatureMetadata;

typedef struct ScorpiShaderSignatureEntry {
  uint32_t system_value;
  uint32_t register_index;
  uint32_t mask;
  uint32_t stream;
  uint32_t component_type;
  uint32_t min_precision;
} ScorpiShaderSignatureEntry;

SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateDxilBinaryFile(const void *shader_code,
                           size_t shader_code_size,
                           const void *signature_dump,
                           size_t signature_dump_size,
                           const char *output_path,
                           char *error_message,
                           size_t error_message_size);

SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateDxilBinary(const void *shader_code,
                       size_t shader_code_size,
                       const void *signature_dump,
                       size_t signature_dump_size,
                       void **dxil_binary,
                       size_t *dxil_binary_size,
                       char *error_message,
                       size_t error_message_size);

enum {
  SCORPI_COVERAGE_MASK_PIXEL = 1,
  SCORPI_COVERAGE_REPLAY_SAFE_VERTEX = 2,
  SCORPI_COVERAGE_REPLAY_SAFE_HULL = 4,
  SCORPI_COVERAGE_REPLAY_SAFE_DOMAIN = 8,
};
// Returns zero for unsupported or invalid shaders. Accepts DXBC or DXIL.
SCORPI_DXBC_TO_DXIL_EXPORT uint32_t
ScorpiInspectCoverageShader(const void *code, size_t size,
                           const void *metadata, size_t metadata_size);

SCORPI_DXBC_TO_DXIL_EXPORT void
ScorpiFreeDxilBinary(void *dxil_binary);

#ifdef __cplusplus
}
#endif

#undef SCORPI_DXBC_TO_DXIL_EXPORT
