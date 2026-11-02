# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Periodic Burgers on one CPU rank. The line DFT is ComplexLineFFT, not HeFFTe.

set(BURGERS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(burgers src/burgers.cpp)
target_include_directories(burgers PRIVATE ${BURGERS_INCLUDE_DIR})
target_link_libraries(burgers PRIVATE OpenPFC::openpfc)
install(TARGETS burgers DESTINATION bin)

option(BURGERS_ENABLE_TESTS "Enable building burgers tests" ON)
if(BURGERS_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
  find_package(Catch2 QUIET)
  if(TARGET Catch2::Catch2WithMain)
    add_executable(test_burgers tests/test_burgers.cpp)
    target_include_directories(test_burgers PRIVATE ${BURGERS_INCLUDE_DIR})
    target_link_libraries(test_burgers PRIVATE OpenPFC::openpfc Catch2::Catch2WithMain)
    include(CTest)
    add_test(NAME burgers-all-tests COMMAND test_burgers)
    set_tests_properties(burgers-all-tests PROPERTIES TIMEOUT 180)
  else()
    message(STATUS "burgers: Catch2 not found, not building tests")
  endif()
endif()
