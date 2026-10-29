# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Keep established executable names while giving their source a thematic home.
add_executable(19_explicit_stepper_fd finite-differences/model.cpp)
add_executable(20_explicit_stepper_spectral fourier/model.cpp)
add_executable(heat_exponential exponential/model.cpp)
add_executable(15_finite_difference_heat separated-halos/model.cpp)
foreach(target IN ITEMS 19_explicit_stepper_fd 20_explicit_stepper_spectral heat_exponential 15_finite_difference_heat)
  target_link_libraries(${target} PRIVATE OpenPFC nlohmann_json::nlohmann_json ${_heffte_tgt})
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
endforeach()

add_executable(04_diffusion_model implicit-euler/model.cpp)
add_executable(05_simulator driver-hooks/model.cpp)
foreach(target IN ITEMS 04_diffusion_model 05_simulator)
  target_link_libraries(${target} PRIVATE OpenPFC nlohmann_json::nlohmann_json)
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
endforeach()
