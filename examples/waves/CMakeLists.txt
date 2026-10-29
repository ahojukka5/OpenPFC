# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(variable_transport variable-transport/model.cpp)
target_link_libraries(variable_transport PRIVATE OpenPFC nlohmann_json::nlohmann_json ${_heffte_tgt})
set_target_properties(variable_transport PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(chebyshev_wave_line chebyshev-line/model.cpp)
add_executable(chebyshev_wave_square chebyshev-square/model.cpp)
add_executable(mixed_wave_tank mixed-tank/model.cpp)
add_executable(kdv_integrating_factor kdv/model.cpp)
foreach(target IN ITEMS chebyshev_wave_line chebyshev_wave_square mixed_wave_tank kdv_integrating_factor)
  target_link_libraries(${target} PRIVATE OpenPFC nlohmann_json::nlohmann_json)
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
endforeach()
