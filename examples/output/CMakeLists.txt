# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(checkpoint_restart checkpoint-restart/model.cpp)
add_executable(write_results binary-field/model.cpp)
add_executable(11_write_results vtk-series/model.cpp)

foreach(target IN ITEMS checkpoint_restart write_results 11_write_results)
  target_link_libraries(${target} PRIVATE OpenPFC nlohmann_json::nlohmann_json ${_heffte_tgt})
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
endforeach()
