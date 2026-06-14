#pragma once

#include <stddef.h>

#if defined(_WIN32)
#define SCORPI_DXBC_TO_DXIL_EXPORT __declspec(dllexport)
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

SCORPI_DXBC_TO_DXIL_EXPORT void
ScorpiFreeDxilBinary(void *dxil_binary);

#ifdef __cplusplus
}
#endif

#undef SCORPI_DXBC_TO_DXIL_EXPORT
