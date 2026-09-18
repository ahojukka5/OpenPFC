# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Research prototype for issues #21 (NS) and #23 (MHD).
# Not a shipped apps/ catalog entry.

set(_ns2d_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_ns2d_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_ns2d_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_ns2d_heffte_tgt heffte)
endif()

add_executable(ns2d_vorticity src/ns2d.cpp)
target_include_directories(ns2d_vorticity PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ns2d_vorticity PRIVATE OpenPFC ${_ns2d_heffte_tgt})

add_executable(mhd2d src/mhd.cpp)
target_include_directories(mhd2d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mhd2d PRIVATE OpenPFC ${_ns2d_heffte_tgt})

if(OpenPFC_BUILD_TESTS)
  # The driver calls MPI_Init itself. Do not wrap with srun/mpiexec here:
  # on LUMI login nodes srun needs a partition and fails the regex check.
  add_test(NAME ns2d-vorticity-cli-verify
           COMMAND $<TARGET_FILE:ns2d_vorticity>
                   --verify --N 16 --steps 8 --dt 0.05 --nu 0.1)
  set_tests_properties(ns2d-vorticity-cli-verify PROPERTIES
                       PASS_REGULAR_EXPRESSION "VERIFY PASS"
                       TIMEOUT 120)
  add_test(NAME mhd2d-cli-verify
           COMMAND $<TARGET_FILE:mhd2d>
                   --verify --N 16 --steps 8 --dt 0.05 --nu 0.1 --eta 0.1)
  set_tests_properties(mhd2d-cli-verify PROPERTIES
                       PASS_REGULAR_EXPRESSION "VERIFY PASS"
                       TIMEOUT 120)
endif()

if(OpenPFC_BUILD_TESTS AND TARGET Catch2::Catch2)
  add_executable(test_ns2d_vorticity tests/test_ns2d.cpp)
  target_include_directories(test_ns2d_vorticity PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_ns2d_vorticity PRIVATE OpenPFC Catch2::Catch2
                        ${_ns2d_heffte_tgt})
  add_test(NAME ns2d-vorticity-cpu COMMAND test_ns2d_vorticity)
  set_tests_properties(ns2d-vorticity-cpu PROPERTIES TIMEOUT 180)

  add_executable(test_mhd2d tests/test_mhd.cpp)
  target_include_directories(test_mhd2d PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_mhd2d PRIVATE OpenPFC Catch2::Catch2
                        ${_ns2d_heffte_tgt})
  add_test(NAME mhd2d-cpu COMMAND test_mhd2d)
  set_tests_properties(mhd2d-cpu PROPERTIES TIMEOUT 180)
endif()
