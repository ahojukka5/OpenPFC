# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(stability_regions stability/model.cpp)
target_link_libraries(stability_regions PRIVATE OpenPFC)
set_target_properties(stability_regions PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(21_adaptive_stepping adaptive-rk/model.cpp)

add_executable(24_adaptive_etd adaptive-etd/model.cpp)

add_executable(time output-clock/model.cpp)

foreach(target IN ITEMS 21_adaptive_stepping 24_adaptive_etd time)
  target_link_libraries(${target} PRIVATE OpenPFC nlohmann_json::nlohmann_json ${_heffte_tgt})
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
endforeach()
