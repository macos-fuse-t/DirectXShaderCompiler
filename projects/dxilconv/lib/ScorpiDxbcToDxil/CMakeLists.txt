set(SCORPI_DXBC_TO_DXIL_SOURCES
  ScorpiDxbcToDxil.cpp
  ScorpiRasterReplay.cpp
  ScorpiGeometryCapture.cpp
  ScorpiTypedBuffers.cpp
)

add_dxilconv_project_library(ScorpiDxbcToDxil
  SHARED
  ${SCORPI_DXBC_TO_DXIL_SOURCES}
)

add_dxilconv_project_library(ScorpiDxbcToDxilStatic
  STATIC
  ${SCORPI_DXBC_TO_DXIL_SOURCES}
)

set_target_properties(ScorpiDxbcToDxilStatic PROPERTIES
  OUTPUT_NAME ScorpiDxbcToDxilStatic
)
target_compile_definitions(ScorpiDxbcToDxilStatic PUBLIC SCORPI_DXBC_TO_DXIL_STATIC)

foreach(target ScorpiDxbcToDxil ScorpiDxbcToDxilStatic)
  target_link_libraries(${target} PUBLIC
    DxbcConverter
    LLVMDxcSupport
    LLVMDxilContainer
    LLVMDxilRootSignature
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

# Package the converter's static dependency closure for the standalone MSBuild UMD.
# Keep the bundle distinct from both the wrapper archive and the DLL import library.
if(MSVC)
  set(scorpi_bundle_targets
    ScorpiDxbcToDxilStatic DxbcConverter DxilConvPasses LLVMDxcSupport
    LLVMDxilContainer LLVMDxilRootSignature LLVMMSSupport LLVMScalarOpts
    LLVMSupport LLVMDxilHash LLVMInstCombine LLVMProfileData LLVMHLSL
    LLVMBitWriter LLVMTransformUtils LLVMipa LLVMAnalysis LLVMDxcBindingTable
    LLVMDXIL LLVMBitReader LLVMCore)
  set(scorpi_bundle_inputs)
  set(SCORPI_BUNDLE_STAGING "${CMAKE_CURRENT_BINARY_DIR}/bundle" CACHE PATH
    "Local staging directory for the Windows static librarian")
  set(scorpi_bundle_local_inputs)
  foreach(dep ${scorpi_bundle_targets})
    list(APPEND scorpi_bundle_inputs "$<TARGET_FILE:${dep}>")
    list(APPEND scorpi_bundle_local_inputs
      "${SCORPI_BUNDLE_STAGING}/$<CONFIG>/$<TARGET_FILE_NAME:${dep}>")
  endforeach()
  add_custom_target(ScorpiDxbcToDxilBundle
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SCORPI_BUNDLE_STAGING}/$<CONFIG>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${scorpi_bundle_inputs}
      "${SCORPI_BUNDLE_STAGING}/$<CONFIG>"
    COMMAND ${CMAKE_AR} /NOLOGO
      "/OUT:${SCORPI_BUNDLE_STAGING}/$<CONFIG>/ScorpiDxbcToDxilBundle.lib"
      ${scorpi_bundle_local_inputs}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${SCORPI_BUNDLE_STAGING}/$<CONFIG>/ScorpiDxbcToDxilBundle.lib"
      "$<TARGET_FILE_DIR:ScorpiDxbcToDxilStatic>/ScorpiDxbcToDxilBundle.lib"
    DEPENDS ${scorpi_bundle_targets}
    VERBATIM)
endif()
