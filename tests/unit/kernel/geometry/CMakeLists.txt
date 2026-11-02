# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

file(GLOB GEOMETRY_UNIT_TEST_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
target_sources(openpfc-tests PRIVATE ${GEOMETRY_UNIT_TEST_SOURCES})
add_test(NAME reference-domains COMMAND openpfc-tests "[reference_domain]")
set_tests_properties(reference-domains PROPERTIES
  ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
  TIMEOUT 60)
