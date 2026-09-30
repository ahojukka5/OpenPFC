# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Issue #233. Periodic Fourier velocity. Not a copy of examples/ns2d_vorticity.

set(_flow_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_flow_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_flow_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_flow_heffte_tgt heffte)
endif()

add_executable(incompressible_flow src/main.cpp src/hit_series.cpp)
target_include_directories(incompressible_flow PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(incompressible_flow PRIVATE OpenPFC ${_flow_heffte_tgt})
install(TARGETS incompressible_flow DESTINATION bin)

if(OpenPFC_BUILD_TESTS)
  # The driver calls MPI_Init itself. Do not wrap it with srun.
  add_test(NAME incompressible-flow-smoke
           COMMAND $<TARGET_FILE:incompressible_flow>
                   --case taylor-green
                   --n 16
                   --nu 0.05
                   --dt 0.015625
                   --time 0.03125
                   --outdir ${CMAKE_CURRENT_BINARY_DIR}/smoke)
  set_tests_properties(incompressible-flow-smoke PROPERTIES
                       PASS_REGULAR_EXPRESSION "incompressible_flow status=ok"
                       TIMEOUT 120)
  add_test(NAME incompressible-flow-hit-smoke
           COMMAND $<TARGET_FILE:incompressible_flow>
                   --case decaying-hit
                   --n 16
                   --nu 0.02
                   --dt 0.015625
                   --time 0.03125
                   --seed 1
                   --outdir ${CMAKE_CURRENT_BINARY_DIR}/hit-smoke)
  set_tests_properties(incompressible-flow-hit-smoke PROPERTIES
                       PASS_REGULAR_EXPRESSION "incompressible_flow status=ok"
                       TIMEOUT 120)
endif()

if(OpenPFC_BUILD_TESTS AND TARGET Catch2::Catch2)
  add_executable(test_incompressible_flow
    tests/test_taylor_green.cpp
    tests/test_decaying_hit.cpp)
  target_include_directories(test_incompressible_flow PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_incompressible_flow PRIVATE OpenPFC Catch2::Catch2
                        ${_flow_heffte_tgt})
  add_test(NAME incompressible-flow-cpu COMMAND test_incompressible_flow)
  set_tests_properties(incompressible-flow-cpu PROPERTIES TIMEOUT 180)
endif()
