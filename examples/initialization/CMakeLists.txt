# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
add_executable(14_custom_field_initializer patterns/model.cpp)
add_executable(10_ui_register_ic catalog/model.cpp)
foreach(target IN ITEMS 14_custom_field_initializer 10_ui_register_ic)
  target_link_libraries(${target} PRIVATE OpenPFC nlohmann_json::nlohmann_json)
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
endforeach()
