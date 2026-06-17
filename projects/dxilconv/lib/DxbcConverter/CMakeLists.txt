# Build DxbcConverter.lib.

find_package(D3D12 REQUIRED)

add_dxilconv_project_library(DxbcConverter
  DxbcConverter.cpp
  DxbcUtil.cpp
  ../ShaderBinary/ShaderBinary.cpp
)

target_link_libraries(DxbcConverter PUBLIC
  DxilConvPasses
  LLVMDxilHash
)

add_dependencies(DxbcConverter intrinsics_gen DxilConvPasses)

target_include_directories(DxbcConverter
PUBLIC
  $<BUILD_INTERFACE:${DXC_PROJECTS_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${DXC_PROJECTS_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
PRIVATE
  ${D3D12_INCLUDE_DIRS}
  ${DXILCONV_PROJECT_SOURCE_DIR}/lib/ShaderBinary
)
