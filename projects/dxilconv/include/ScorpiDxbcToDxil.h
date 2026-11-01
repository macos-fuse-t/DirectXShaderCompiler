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
  // Backend metadata v1 only: DX11 DDI stage interfaces link by register and
  // component. This is not a flag in the v2 DXBC conversion signature dump.
  SCORPI_SHADER_BACKEND_REGISTER_LINKAGE = 16,
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

// Optional execution-only input for native compilers. Preserve executable
// DXIL and resource metadata; remove source-debug information. Unchanged
// shaders return a null binary. Release output with ScorpiFreeDxilBinary.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiStripShaderDebugInfo(const void *code, size_t size,
    void **binary, size_t *binary_size, char *error, size_t error_size);

// Internal renderer linkage ABI: four DDI component types (0=unchanged,
// 1=uint32, 2=int32, 3=float32) per register, packed into four nibbles.
// Retypes stream-zero arbitrary outputs with bitcasts, never numeric conversion.
// Selected components of each signature element must agree on a 32-bit type.
// Returns a null binary when the existing output types already match.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiRetypeShaderOutputs(const void *code, size_t size,
    const uint16_t component_types[32], void **binary, size_t *binary_size,
    char *error_message, size_t error_message_size);

// Native rasterizer linkage by semantic and semantic index, independent of
// each stage's register packing. Names point into the returned allocation.
// Flat float inputs request uint32 bit transport; ordinary inputs retain their
// interpolation type. Release the array with ScorpiFreeDxilBinary.
typedef struct ScorpiShaderLinkageElement {
    const char *semantic_name;
    uint32_t semantic_index, register_index, first_component, component_count;
    uint32_t component_type, output, flat_float;
} ScorpiShaderLinkageElement;
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiInspectShaderLinkage(const void *code, size_t size,
    const void *metadata, size_t metadata_size,
    ScorpiShaderLinkageElement **elements, size_t *count,
    char *error_message, size_t error_message_size);

// Complete register layout for DDI linkage. output bit0 selects outputs;
// bit1 marks a system-value element. The legacy inspector stays arbitrary-only.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiInspectRegisterLinkage(const void *code, size_t size,
    const void *metadata, size_t metadata_size,
    ScorpiShaderLinkageElement **elements, size_t *count,
    char *error, size_t error_size);

// Link register-mapped DDI inputs to the complete upstream output layout.
// The caller must select this only for DDI linkage, not API semantic linkage.
// Geometry shaders retain unused/system fields to match the producer payload.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiLinkRegisterInputs(const void *code, size_t size,
    const ScorpiShaderLinkageElement *upstream, size_t upstream_count,
    void **binary, size_t *binary_size, char *error, size_t error_size);

// Retype only exports bound to signed integer RTVs. Float exports are raw
// register bits, not numeric float-to-int conversions. Null output means the
// shader already has the required types. Other color slots remain unchanged.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateSignedRenderTargetShader(const void *code, size_t size,
    uint32_t signed_targets, void **binary, size_t *binary_size,
    char *error, size_t error_size);

// Lower target-zero alpha to native output coverage, preserving color bits.
// Existing SV_Coverage disables alpha-to-coverage and returns null output.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateAlphaToCoverageShader(const void *code, size_t size,
    uint32_t sample_count, void **binary, size_t *binary_size,
    char *error, size_t error_size);

// Inspect the original output signature, before backend-generated masks.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiInspectOutputCoverage(const void *code, size_t size,
    const void *metadata, size_t metadata_size, int *writes_coverage);

enum {
  SCORPI_RASTERIZER_SAMPLE_COUNT = 1,
  SCORPI_RASTERIZER_CENTROID = 2,
  SCORPI_RASTERIZER_INPUT_COVERAGE = 4,
};
// Inspect once per shader. Ordinary shaders do not require a rasterizer variant.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiInspectRasterizerRequirements(const void *code, size_t size,
    const void *metadata, size_t metadata_size, uint32_t *requirements);

// Fold the logical count, normalize forced input coverage and expose centroid
// inputs as native interpolants. Returned requirements identify AIR adaptations
// still needed after native compilation. Null binary means DXIL was unchanged.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiSpecializeRasterizerShader(const void *code, size_t size,
    uint32_t sample_count, uint32_t forced_sample_count, uint32_t sample_mask,
    void **binary, size_t *binary_size, uint32_t *requirements,
    char *error, size_t error_size);

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

// Proves a coverage-only PS is a zero-preserving bit selection/replication.
// Each output bit depends on at most one coverage bit, so OR/XOR reduction
// commutes with this transform. The low four bits must preserve raw coverage
// for the four-sample scratch pass. Returns one on success, zero otherwise.
SCORPI_DXBC_TO_DXIL_EXPORT uint32_t
ScorpiInspectCoverageTransform(const void *code, size_t size,
    const void *metadata, size_t metadata_size, uint32_t samples,
    uint32_t output_masks[16]);

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

// Preserve the original draw's PrimitiveID when replay draws one primitive
// at a time. Reads primitive_id from private b0 in register_space.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateRasterReplayPixelDrawIds(const void *code, size_t size,
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

// Inspect mesh output signature rows without changing the input bytecode.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiInspectMeshOutputRows(const void *code, size_t size, int *required);
// Fully unroll bounded mesh loops that dynamically index output signature rows.
// No dispatch branches are added. Unchanged shaders return null output.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiUnrollMeshOutputLoops(const void *code, size_t size, void **binary,
    size_t *binary_size, char *error, size_t error_size);

// Metal stores logical 1D mip chains as height-one 2D textures. Promote the
// DXIL resource kind and texture coordinates to match that physical storage.
// An unchanged shader returns a null binary and zero size.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiPromoteTexture1D(const void *code, size_t size, void **binary,
    size_t *binary_size, char *error, size_t error_size);

// Private descriptor metadata ABI v4. Each heap slot contains a native buffer
// descriptor for an immutable view-local payload: a 32-byte record followed by
// sparse feedback (when needed). Copies preserve that descriptor verbatim.
// t0 is per-draw root metadata; resource and sampler payloads use disjoint ranges.
#define SCORPI_DESCRIPTOR_METADATA_ABI 4
#define SCORPI_METADATA_RECORD_BYTES 32
// Buffer lock identity is the byte offset within a 64-KiB backing tile.
// Texture identities use full-resource coordinates; word 7 carries the view
// array/depth origin instead of a buffer byte origin.
// Dropping the tile identity permits collisions and preserves sparse aliases.
#define SCORPI_ATOMIC64_TILE_BYTES 65536u
#define SCORPI_ATOMIC64_STRIPES (SCORPI_ATOMIC64_TILE_BYTES / 8u)
// Two words per stripe: native-reader gate followed by the fallback mutex.
#define SCORPI_ATOMIC64_WORDS_PER_STRIPE 2u
#define SCORPI_ATOMIC64_WAVE_LANES 32u
#define SCORPI_ATOMIC64_LOCK_WORDS ((SCORPI_ATOMIC64_STRIPES + 1u) * SCORPI_ATOMIC64_WORDS_PER_STRIPE)
#define SCORPI_ATOMIC64_EMULATED_BIT 0x80000000u
#define SCORPI_ATOMIC64_READER_MASK 0x7fffffffu
#define SCORPI_ATOMIC64_ORIGIN_WORD 7u
#define SCORPI_METADATA_RESOURCE_REGISTER 1
#define SCORPI_METADATA_RESOURCE_DESCRIPTORS 1048576
#define SCORPI_METADATA_SAMPLER_REGISTER 1048577
#define SCORPI_METADATA_SAMPLER_DESCRIPTORS 4096
#define SCORPI_METADATA_PRIVATE_PARAMETERS 4
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
// Includes declaration ranges and singleton registers used by constant
// handles. Their requirement unions completely describe policy observations;
// callers must preserve singletons when canonicalizing compilation keys.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiInspectSamplingBindings(const void *code, size_t size,
    const void *metadata, size_t metadata_size,
    ScorpiSamplingBinding *bindings, size_t *count);
// Descriptor residency reflection, not a sampling-policy key. Constant
// handles cover one register; dynamically indexed handles cover their full
// declaration. This only inspects bytecode and does not rewrite the shader.
typedef struct ScorpiDescriptorAccess {
  uint32_t resource_class, space, base_register, count;
  // True only when every reachable use of this SRV handle samples a texture.
  // Unknown uses, texture loads and all UAVs are false.
  uint32_t sample_only;
} ScorpiDescriptorAccess;
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiInspectDescriptorAccess(const void *code, size_t size,
    const void *metadata, size_t metadata_size,
    ScorpiDescriptorAccess *bindings, size_t *count);
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

// DXBC/SM5 cannot contain i64 atomics and bypasses bitcode parsing entirely.
enum ScorpiAtomic64Requirements {
    SCORPI_ATOMIC64_BUFFER = 1,
    SCORPI_ATOMIC64_NATIVE_TEXTURE_MINMAX = 2,
    SCORPI_ATOMIC64_TYPED_EMULATION = 4,
    SCORPI_ATOMIC64_GROUPSHARED = 8,
};
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiInspectAtomic64Requirements(const void *code, size_t size, uint32_t *requirements);
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiEmulateAtomic64(const void *code, size_t size, uint32_t lock_space,
    const ScorpiTypedBufferBinding *bindings, size_t binding_count,
    void **binary, size_t *binary_size, char *error, size_t error_size);
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiEmulateAtomic64WithOptions(const void *code, size_t size, uint32_t lock_space,
    const ScorpiTypedBufferBinding *bindings, size_t binding_count, int native_minmax,
    void **binary, size_t *binary_size, char *error, size_t error_size);

SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiInspectAtomic64(const void *code, size_t size, int *required);

// Only for attachment-backed logical 8x rasterization on 4x hardware.
struct ScorpiSamplingRequirement;
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateMsaa8PixelShader(const void *code, size_t size, uint32_t space,
    uint32_t mask, int generated, void **binary, size_t *binary_size, char *error, size_t error_size);
#define SCORPI_MSAA8_ALPHA_TO_COVERAGE 1u
// When requested, report whether full logical coverage must be collected.
// A zero result permits native per-group raster coverage and whole-draw batching.
#define SCORPI_MSAA8_REPLAY_COVERAGE 1u
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateMsaa8PixelShaderEx(const void *code, size_t size, uint32_t space,
    uint32_t mask, int generated, uint32_t options, void **binary,
    size_t *binary_size, char *error, size_t error_size);
// Zero requirements permits batching subject to VS/capture/ordering constraints.
SCORPI_DXBC_TO_DXIL_EXPORT int
ScorpiCreateMsaa8PixelShaderVariant(const void *code, size_t size, uint32_t space,
    uint32_t mask, int generated, uint32_t options, void **binary,
    size_t *binary_size, char *error, size_t error_size, uint32_t *requirements);
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
#define SCORPI_SAMPLING_NULL_BUFFER 128u
// Renderer selection must account for every reachable typed SRV/UAV prefix.
#define SCORPI_SHADER_NATIVE_TYPED_BUFFERS 8u
// Keep native typed addressing, repairing dimensions and logical bounds
// before an alignment prefix or scalar-backed RGB32 scaling is applied.
#define SCORPI_SHADER_TYPED_DIMENSIONS_ONLY 16u
// Some native GPUs preserve FP32 subnormals in comparisons. Canonicalize only
// comparison inputs when the caller has observed that behavior on its device.
#define SCORPI_SHADER_FLUSH_FP32_COMPARISON_INPUTS 32u
// Enforce default/FTZ arithmetic only when native device probes require it.
// Explicit preserve-mode shaders and raw data movement remain unchanged.
#define SCORPI_SHADER_FLUSH_FP32_ARITHMETIC 64u
// Match a rasterizer producer retyped with ScorpiRetypeShaderOutputs.
// Transport only flat FP32 varyings as integer bits; interpolation is unchanged.
#define SCORPI_SHADER_FLAT_FLOAT_INPUT_BITS 128u
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
