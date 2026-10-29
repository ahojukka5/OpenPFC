# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(polynomial_interpolation interpolation/model.cpp)
add_executable(polynomial_nodal_polynomials nodal-polynomials/model.cpp)
add_executable(polynomial_differentiation differentiation/model.cpp)
add_executable(polynomial_regularity regularity/model.cpp)
add_executable(polynomial_integration integration/model.cpp)
add_executable(polynomial_boundary_values boundary-values/model.cpp)
add_executable(polynomial_nonlinear_boundary nonlinear-boundary/model.cpp)

foreach(target polynomial_interpolation polynomial_nodal_polynomials polynomial_differentiation
               polynomial_regularity polynomial_integration polynomial_boundary_values polynomial_nonlinear_boundary)
  target_link_libraries(${target} PRIVATE OpenPFC nlohmann_json::nlohmann_json)
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
endforeach()

add_executable(allen_cahn_fixed allen-cahn-fixed/model.cpp)
add_executable(allen_cahn_driven allen-cahn-driven/model.cpp)
foreach(target IN ITEMS allen_cahn_fixed allen_cahn_driven)
  target_link_libraries(${target} PRIVATE OpenPFC nlohmann_json::nlohmann_json)
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
endforeach()

add_executable(chebyshev_eigenmodes eigenmodes/model.cpp)
add_executable(airy_eigenproblem airy/model.cpp)
add_executable(membrane_eigenmodes membrane/model.cpp)
add_executable(square_poisson poisson-square/model.cpp)
add_executable(square_helmholtz helmholtz-square/model.cpp)
add_executable(square_laplace laplace-square/model.cpp)
add_executable(clamped_beam clamped-beam/model.cpp)
add_executable(clamped_plate clamped-plate/model.cpp)
add_executable(orr_sommerfeld orr-sommerfeld/model.cpp)
add_executable(davies_pseudospectra pseudospectra/model.cpp)
add_executable(disk_eigenmodes disk-modes/model.cpp)
add_executable(disk_poisson poisson-disk/model.cpp)
foreach(target IN ITEMS chebyshev_eigenmodes airy_eigenproblem membrane_eigenmodes
                        square_poisson square_helmholtz square_laplace clamped_beam clamped_plate
                        orr_sommerfeld davies_pseudospectra disk_eigenmodes disk_poisson)
  target_link_libraries(${target} PRIVATE OpenPFC nlohmann_json::nlohmann_json)
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
endforeach()
