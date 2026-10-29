# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(mpi_worker hello/model.cpp)

add_executable(mpi_worker_inside_class session/model.cpp)

foreach(target IN ITEMS mpi_worker mpi_worker_inside_class)
  target_link_libraries(${target} PRIVATE OpenPFC nlohmann_json::nlohmann_json ${_heffte_tgt})
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
endforeach()
