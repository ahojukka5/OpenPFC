# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
add_executable(test_distributed_grain main.cpp)
target_link_libraries(test_distributed_grain PRIVATE OpenPFC)
add_test(NAME DistributedGrain COMMAND test_distributed_grain)
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
  add_executable(test_distributed_grain_hip native.hip)
  target_link_libraries(test_distributed_grain_hip PRIVATE OpenPFC openpfc_hip_kernels hip::host)
  set_target_properties(test_distributed_grain_hip PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
endif()
foreach(_ranks 1 2 4 8)
  if(NOT OpenPFC_RUN_MPI_SUITES OR (OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER 0 AND _ranks GREATER OpenPFC_MPI_TEST_MAX_WORLD_SIZE))
    continue()
  endif()
  add_test(NAME DistributedGrain_${_ranks}ranks COMMAND ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} ${_ranks} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_distributed_grain> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(DistributedGrain_${_ranks}ranks PROPERTIES RUN_SERIAL TRUE TIMEOUT 180)
  if(TARGET test_distributed_grain_hip)
    add_test(NAME HIP_DistributedGrain_${_ranks}ranks COMMAND ${MPIEXEC_EXECUTABLE}
      ${MPIEXEC_NUMPROC_FLAG} ${_ranks} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_distributed_grain_hip> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(HIP_DistributedGrain_${_ranks}ranks PROPERTIES RUN_SERIAL TRUE TIMEOUT 180)
  endif()
endforeach()
