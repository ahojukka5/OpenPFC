# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(periodic_differentiation periodic/model.cpp)
target_link_libraries(periodic_differentiation PRIVATE OpenPFC nlohmann_json::nlohmann_json ${_heffte_tgt})
set_target_properties(periodic_differentiation PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(periodic_collocation collocation/model.cpp)
add_executable(spectral_smoothness smoothness/model.cpp)
foreach(target IN ITEMS periodic_collocation spectral_smoothness)
  target_link_libraries(${target} PRIVATE OpenPFC nlohmann_json::nlohmann_json ${_heffte_tgt})
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
endforeach()
