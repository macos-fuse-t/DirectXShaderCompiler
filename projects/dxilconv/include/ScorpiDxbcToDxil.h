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
  // No output is rasterized. Missing arbitrary GS output types represent raw
  // 32-bit stream-output storage, not an interpolated shader interface.
  SCORPI_SHADER_SIGNATURE_STREAM_OUTPUT_ONLY = 1,
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

typedef struct ScorpiRasterReplayParameters {
  uint32_t width, height, coverage_base_element, sample_base;
  int32_t origin_x, origin_y;
  uint32_t primitive_id, reserved;
  // Offsets within the original draw when capture/replay splits its work.
  uint32_t vertex_bias, instance_bias, reserved_draw[2];
} ScorpiRasterReplayParameters;

SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateRasterReplayVertexShader(const void *code, size_t size,
                                   uint32_t register_space, void **binary,
                                   size_t *binary_size, char *error_message,
                                   size_t error_message_size);

// Internal renderer ABI. Produces an attachment-free PS variant with private
// b0=ScorpiRasterReplayParameters and raw coverage SRV t0 in
// register_space. Coverage belongs to one generated primitive, after all sample
// groups have been accumulated. Inactive replay lanes retain derivative work
// but cannot write application UAVs. The caller owns the returned binary.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateRasterReplayPixelShader(const void *code, size_t size,
                                  uint32_t register_space, uint32_t sample_count,
                                  uint32_t sample_mask, void **binary,
                                  size_t *binary_size, char *error_message,
                                  size_t error_message_size);

// Internal geometry capture ABI. capture=1 generates a GS that records the
// assembled VS output; capture=0 generates a VS that reads those records.
// Both use a private raw UAV u0 in register_space. Records are packed in
// output-signature element/row/component order, with vertex_stride bytes.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateGeometryCaptureShader(const void *code, size_t size,
                                 uint32_t register_space, uint32_t vertices_per_primitive,
                                 int capture, uint32_t *vertex_stride, void **binary,
                                 size_t *binary_size, char *error_message,
                                 size_t error_message_size);

// Generated-geometry capture is bounded by the GS declaration. Each instance
// owns a 16-byte header (emitted vertex count at byte zero), followed by
// max_vertices records. A record is {primitiveID, stripVertexIndex, payload};
// payload uses output-signature element/row/component order. Replay parameters
// primitive_id/reserved select the instance header/first vertex record in bytes.
typedef struct ScorpiGeometryCaptureLayout {
  uint32_t vertex_stride;
  uint32_t max_vertices;
  uint32_t instance_count;
  uint32_t vertices_per_primitive;
} ScorpiGeometryCaptureLayout;

SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateGeometryShaderCapture(const void *code, size_t size,
                                 uint32_t register_space, int capture,
                                 ScorpiGeometryCaptureLayout *layout,
                                 void **binary, size_t *binary_size,
                                 char *error_message, size_t error_message_size);

// Batched generated replay reads a compact table in private UAV u0. Each
// 16-byte entry is {vertexRecord0, vertexRecord1, vertexRecord2, PrimitiveID}.
// Parameters.reserved selects its first entry in bytes. Coverage uses a distinct
// width*height slice per raster primitive in the batch.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateGeneratedRasterReplayPixelShader(const void *code, size_t size,
    uint32_t register_space, uint32_t samples, uint32_t sample_mask,
    void **binary, size_t *binary_size, char *error, size_t error_size);
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateCompactedGeometryReplayShader(const void *code, size_t size,
    uint32_t register_space, ScorpiGeometryCaptureLayout *layout,
    void **binary, size_t *binary_size, char *error, size_t error_size);

// Optional backend adaptation: carry an application GS PrimitiveID as a flat
// user varying when a backend cannot link the native primitive system value.
// Apply consistently to GS output and PS input; no logical shader value changes.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiLowerGeometryPrimitiveId(const void *code, size_t size, void **binary,
    size_t *binary_size, char *error, size_t error_size);

// One output-signature row and its location in the capture payload. This is
// logical shader metadata, independent of any backend's stream-output encoding.
typedef struct ScorpiGeometryCaptureElement {
  char semantic_name[64];
  uint32_t semantic_index, register_index, start_component, component_count;
  uint32_t byte_offset, stream;
} ScorpiGeometryCaptureElement;
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiGetGeometryCaptureElements(const void *code, size_t size,
    ScorpiGeometryCaptureElement *elements, uint32_t capacity, uint32_t *count,
    char *error, size_t error_size);

SCORPI_DXBC_TO_DXIL_EXPORT void
ScorpiFreeDxilBinary(void *dxil_binary);

#ifdef __cplusplus
}
#endif

#undef SCORPI_DXBC_TO_DXIL_EXPORT
