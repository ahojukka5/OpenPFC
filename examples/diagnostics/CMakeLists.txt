# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(mpi_timers mpi-timing/model.cpp)

add_executable(profiling_timer_report profiling/model.cpp)

foreach(target IN ITEMS mpi_timers profiling_timer_report)
  target_link_libraries(${target} PRIVATE OpenPFC nlohmann_json::nlohmann_json ${_heffte_tgt})
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
endforeach()
