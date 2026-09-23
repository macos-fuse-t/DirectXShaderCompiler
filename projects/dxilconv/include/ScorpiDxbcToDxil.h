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
  // The runtime supplies the rasterized stream's typed signature; other GS
  // streams carry raw SO values. Bits 2-3 identify the rasterized stream.
  SCORPI_SHADER_SIGNATURE_RASTERIZED_STREAM = 2,
  SCORPI_SHADER_SIGNATURE_RASTERIZED_STREAM_SHIFT = 2,
  SCORPI_SHADER_SIGNATURE_RASTERIZED_STREAM_MASK = 12,
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
  uint32_t vertex_bias, instance_bias, visibility_enabled, reserved_draw;
} ScorpiRasterReplayParameters;

SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateRasterReplayVertexShader(const void *code, size_t size,
                                   uint32_t register_space, void **binary,
                                   size_t *binary_size, char *error_message,
                                   size_t error_message_size);

// Internal renderer ABI. Produces an attachment-free PS variant with private
// b0=ScorpiRasterReplayParameters, raw coverage SRV t0 and raw visibility UAV u1 in
// register_space. Coverage belongs to one generated primitive, after all sample
// groups have been accumulated. Inactive replay lanes retain derivative work
// but cannot write application UAVs. When visibility_enabled is nonzero, u1
// receives surviving sample masks at the same element indices as t0. The caller
// clears u1 before replay and accumulates its popcounts after all sample groups.
// The caller owns the returned binary.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateRasterReplayPixelShader(const void *code, size_t size,
                                  uint32_t register_space, uint32_t sample_count,
                                  uint32_t sample_mask, void **binary,
                                  size_t *binary_size, char *error_message,
                                  size_t error_message_size);

// Private tessellation capture linkage. mode 1 rewrites DS outputs to uint4
// key/payload rows; mode 2 produces the matching ordered capture GS;
// mode 3 replays compacted capture through a VS with the original DS signature;
// mode 4 produces a passthrough GS; mode 5 preserves HS/patch-constant primitive
// IDs when drawing one patch at a time, using private b0 primitive_id as a bias.
// Mode 6 consumes canonical DS output without capture stores, for backends
// that capture point-domain results directly before primitive assembly.
// Mode 7 supplies the per-patch pixel PrimitiveID when there is no application GS.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateTessellationCaptureShader(const void *code, size_t size,
    uint32_t register_space, uint32_t vertices_per_primitive, uint32_t mode,
    uint32_t *vertex_stride, void **binary, size_t *binary_size,
    char *error_message, size_t error_message_size);

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
// max_vertices records per active stream. Each stream has its own header and
// contiguous records {primitiveID, stripVertexIndex, payload} within a slot.
// payload uses output-signature element/row/component order. Replay parameters
// primitive_id/reserved select the instance header/first vertex record in bytes.
typedef struct ScorpiGeometryCaptureLayout {
  uint32_t vertex_stride;
  uint32_t max_vertices;
  uint32_t instance_count;
  uint32_t vertices_per_primitive;
  uint32_t stream_mask;
} ScorpiGeometryCaptureLayout;

// Batched modes index records by input primitive, then GS instance. The
// no-raster variant keeps all captured payload but exports only dummy position
// to Metal's disabled raster stage; it does not consume pixel attribute slots.
enum ScorpiGeometryCaptureMode {
  SCORPI_GEOMETRY_REPLAY = 0,
  SCORPI_GEOMETRY_CAPTURE = 1,
  SCORPI_GEOMETRY_CAPTURE_BATCHED = 2,
  SCORPI_GEOMETRY_CAPTURE_BATCHED_NO_RASTER = 3,
  SCORPI_GEOMETRY_CAPTURE_NO_RASTER = 4
};
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
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateCompactedGeometryReplayShaderForStream(const void *code, size_t size,
    uint32_t register_space, uint32_t rasterized_stream,
    ScorpiGeometryCaptureLayout *layout, void **binary, size_t *binary_size,
    char *error, size_t error_size);

// Optional backend adaptation: carry an application GS PrimitiveID as a flat
// user varying when a backend cannot link the native primitive system value.
// Apply consistently to GS output and PS input; no logical shader value changes.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiLowerGeometryPrimitiveId(const void *code, size_t size, void **binary,
    size_t *binary_size, char *error, size_t error_size);

// Metal stores logical 1D mip chains as height-one 2D textures. Promote the
// DXIL resource kind and texture coordinates to match that physical storage.
// An unchanged shader returns a null binary and zero size.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiPromoteTexture1D(const void *code, size_t size, void **binary,
    size_t *binary_size, char *error, size_t error_size);

// Renderer-private typed-buffer view metadata. The raw SRV at t0 in space
// metadata_space points at the immutable root-argument allocation. Each table
// root contains a GPU address within that allocation; descriptors are 24 bytes,
// with the texture alignment prefix in metadata bits 32..39 (byte 20).
typedef struct ScorpiTypedBufferBinding {
  uint32_t resource_class, space, base_register, count;
  uint32_t root_offset, table_offset;
  uint32_t feedback_offset;
} ScorpiTypedBufferBinding;
// Reflection only: declared reachable SRV/sampler register ranges. No rewriting.
// Set bindings=nullptr to query the count; count is input capacity on fill.
typedef struct ScorpiSamplingBinding {
  uint32_t resource_class, space, base_register, count;
} ScorpiSamplingBinding;
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiInspectSamplingBindings(const void *code, size_t size,
    const void *metadata, size_t metadata_size,
    ScorpiSamplingBinding *bindings, size_t *count);
// Read-only input signature reflection, including raw DXIL programs without
// an ISG1 container part. No optimization, rewriting or serialization occurs.
typedef struct ScorpiVertexInput {
  uint32_t register_index, semantic_index;
  char name[64];
} ScorpiVertexInput;
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiInspectVertexInputs(const void *code, size_t size,
    ScorpiVertexInput *inputs, size_t *count, char *error, size_t error_size);
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiLowerTypedBufferViews(const void *code, size_t size,
    const ScorpiTypedBufferBinding *bindings, size_t binding_count,
    uint32_t metadata_space, uint32_t self_offset,
    void **binary, size_t *binary_size, char *error, size_t error_size);

// Only for attachment-backed logical 8x rasterization on 4x hardware.
struct ScorpiSamplingRequirement;
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateMsaa8PixelShader(const void *code, size_t size, uint32_t space,
    uint32_t mask, int generated, void **binary, size_t *binary_size, char *error, size_t error_size);
#define SCORPI_MSAA8_ALPHA_TO_COVERAGE 1u
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateMsaa8PixelShaderEx(const void *code, size_t size, uint32_t space,
    uint32_t mask, int generated, uint32_t options, void **binary,
    size_t *binary_size, char *error, size_t error_size);
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiLowerTextureLayouts(const void *code, size_t size, int split_msaa8,
    const struct ScorpiSamplingRequirement *policy, size_t policy_count,
    void **binary, size_t *binary_size, char *error_message, size_t error_message_size);

// The caller has proved all sampled bindings are ordinary, fully backed views
// with no resource MinLOD. Keep native texture operations unchanged. Shaders
// consuming residency status still use the compatibility path.
#define SCORPI_SHADER_NATIVE_TEXTURES 1u
// No reachable bound sampler uses reduction filtering. Do not emit min/max
// reduction even when another requirement (clamps/residency) needs lowering.
#define SCORPI_SHADER_NO_MINMAX_FILTERING 2u
// Exhaustive semantic exceptions for the immutable descriptor snapshot. An
// absent register is ordinary; unknown handle provenance stays conservative.
#define SCORPI_SHADER_BINDING_POLICY 4u
#define SCORPI_SAMPLING_SPARSE 1u
#define SCORPI_SAMPLING_RESOURCE_MIN_LOD 2u
#define SCORPI_SAMPLING_REDUCTION 4u
#define SCORPI_SAMPLING_TYPED_PREFIX 8u
#define SCORPI_SAMPLING_TYPED_RGB32 16u
#define SCORPI_SAMPLING_TYPED_EXTENDED_PREFIX 32u
#define SCORPI_SAMPLING_MSAA8 64u
#define SCORPI_SHADER_GUARD_TYPED_PREFIX 32u
// Renderer selection must account for every reachable typed SRV/UAV prefix.
#define SCORPI_SHADER_NATIVE_TYPED_BUFFERS 8u
// The native compiler handles typed addressing. Only repair GetDimensions
// for bindings with an alignment prefix or scalar-backed RGB32 view.
#define SCORPI_SHADER_TYPED_DIMENSIONS_ONLY 16u
typedef struct ScorpiSamplingRequirement {
  uint32_t resource_class, space, base_register, count, requirements;
} ScorpiSamplingRequirement;
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiLowerTypedBufferViewsWithSamplingPolicy(const void *code, size_t size,
    const ScorpiTypedBufferBinding *bindings, size_t binding_count,
    uint32_t metadata_space, uint32_t self_offset, uint32_t options,
    const ScorpiSamplingRequirement *policy, size_t policy_count,
    void **binary, size_t *binary_size, char *error, size_t error_size);
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiLowerTypedBufferViewsWithOptions(const void *code, size_t size,
    const ScorpiTypedBufferBinding *bindings, size_t binding_count,
    uint32_t metadata_space, uint32_t self_offset, uint32_t options,
    void **binary, size_t *binary_size, char *error, size_t error_size);

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
