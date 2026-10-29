# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(02_domain_decomposition decomposition/model.cpp)
add_executable(06_multi_index index-boxes/model.cpp)
add_executable(07_array field-storage/model.cpp)
add_executable(08_discrete_fields coordinate-fields/model.cpp)
add_executable(world_helpers_example grid-spacing/model.cpp)
add_executable(world_query_helpers_example domain-queries/model.cpp)
add_executable(16_strong_types_demo strong-arguments/model.cpp)
add_executable(world_strong_types_example anisotropic-grid/model.cpp)
foreach(target IN ITEMS 02_domain_decomposition 06_multi_index 07_array 08_discrete_fields
                        world_helpers_example world_query_helpers_example 16_strong_types_demo world_strong_types_example)
  target_link_libraries(${target} PRIVATE OpenPFC ${_heffte_tgt})
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
endforeach()

add_executable(first_domain first-domain/model.cpp)
target_link_libraries(first_domain PRIVATE OpenPFC)
set_target_properties(first_domain PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(17_custom_coordinate_system polar-patterns/model.cpp)
target_link_libraries(17_custom_coordinate_system PRIVATE OpenPFC nlohmann_json::nlohmann_json)
set_target_properties(17_custom_coordinate_system PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
