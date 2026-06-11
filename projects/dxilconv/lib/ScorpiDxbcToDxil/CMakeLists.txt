set(SCORPI_DXBC_TO_DXIL_SOURCES
  ScorpiDxbcToDxil.cpp
)

add_dxilconv_project_library(ScorpiDxbcToDxil
  SHARED
  ${SCORPI_DXBC_TO_DXIL_SOURCES}
)

add_dxilconv_project_library(ScorpiDxbcToDxilStatic
  ${SCORPI_DXBC_TO_DXIL_SOURCES}
)

set_target_properties(ScorpiDxbcToDxilStatic PROPERTIES
  OUTPUT_NAME ScorpiDxbcToDxil
)

foreach(target ScorpiDxbcToDxil ScorpiDxbcToDxilStatic)
  target_link_libraries(${target} PUBLIC
    DxbcConverter
    LLVMDxcSupport
    LLVMDxilContainer
    LLVMMSSupport
    LLVMScalarOpts
    LLVMSupport
  )

  add_dependencies(${target} DxbcConverter)

  target_include_directories(${target}
  PUBLIC
    $<BUILD_INTERFACE:${DXILCONV_PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${DXC_PROJECTS_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${DXC_PROJECTS_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${DXILCONV_PROJECT_SOURCE_DIR}/lib/DxbcConverter
  )
endforeach()
