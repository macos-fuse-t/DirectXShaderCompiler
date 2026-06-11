# Builds a local Scorpi DXBC token-stream conversion probe.

add_dxilconv_project_executable(scorpi-dxbc-probe
  scorpi-dxbc-probe.cpp
)

target_link_libraries(scorpi-dxbc-probe PRIVATE
  ScorpiDxbcToDxilStatic
)

add_dependencies(scorpi-dxbc-probe ScorpiDxbcToDxilStatic)
