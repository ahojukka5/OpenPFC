# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(03_parallel_fft round-trip/model.cpp)
add_executable(09_parallel_fft_high_level field-layout/model.cpp)
foreach(target IN ITEMS 03_parallel_fft 09_parallel_fft_high_level)
  target_link_libraries(${target} PRIVATE OpenPFC nlohmann_json::nlohmann_json ${_heffte_tgt})
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
endforeach()

add_executable(harmonic_oscillator harmonic-oscillator/model.cpp)
add_executable(mathieu_spectrum mathieu/model.cpp)
foreach(target IN ITEMS harmonic_oscillator mathieu_spectrum)
  target_link_libraries(${target} PRIVATE OpenPFC nlohmann_json::nlohmann_json)
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
endforeach()
