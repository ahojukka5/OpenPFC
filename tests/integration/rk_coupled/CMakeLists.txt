# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
add_executable(test_rk_coupled main.cpp)
target_link_libraries(test_rk_coupled PRIVATE OpenPFC)
add_test(NAME RK_CoupledFields COMMAND test_rk_coupled)
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
  add_executable(test_rk_coupled_hip native.hip)
  target_link_libraries(test_rk_coupled_hip PRIVATE OpenPFC openpfc_hip_kernels hip::host)
  set_target_properties(test_rk_coupled_hip PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)

endif()

foreach(_ranks 1 2 4 8)
  if(NOT OpenPFC_RUN_MPI_SUITES AND _ranks GREATER 1)
    continue()
  endif()
  if(OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER 0 AND _ranks GREATER OpenPFC_MPI_TEST_MAX_WORLD_SIZE)
    continue()
  endif()
  if(OpenPFC_RUN_MPI_SUITES)
    add_test(NAME RK_Coupled_${_ranks}ranks COMMAND ${MPIEXEC_EXECUTABLE}
      ${MPIEXEC_NUMPROC_FLAG} ${_ranks} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_rk_coupled> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(RK_Coupled_${_ranks}ranks PROPERTIES RUN_SERIAL TRUE TIMEOUT 180)
  endif()
  if(TARGET test_rk_coupled_hip)
    add_test(NAME HIP_RK_Coupled_${_ranks}ranks COMMAND ${MPIEXEC_EXECUTABLE}
      ${MPIEXEC_NUMPROC_FLAG} ${_ranks} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_rk_coupled_hip> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(HIP_RK_Coupled_${_ranks}ranks PROPERTIES RUN_SERIAL TRUE TIMEOUT 180)
  endif()
endforeach()
