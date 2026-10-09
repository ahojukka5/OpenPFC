# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
add_executable(test_label_contacts main.cpp)
target_link_libraries(test_label_contacts PRIVATE OpenPFC)
add_test(NAME LabelContacts COMMAND test_label_contacts)
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
  add_executable(test_label_contacts_hip native.hip)
  target_link_libraries(test_label_contacts_hip PRIVATE OpenPFC openpfc_hip_kernels hip::host)
  set_target_properties(test_label_contacts_hip PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
endif()
foreach(_ranks 1 2 4 8)
  if(NOT OpenPFC_RUN_MPI_SUITES OR (OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER 0 AND _ranks GREATER OpenPFC_MPI_TEST_MAX_WORLD_SIZE))
    continue()
  endif()
  add_test(NAME LabelContacts_${_ranks}ranks COMMAND ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} ${_ranks} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_label_contacts> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(LabelContacts_${_ranks}ranks PROPERTIES RUN_SERIAL TRUE TIMEOUT 180)
  if(TARGET test_label_contacts_hip)
    add_test(NAME HIP_LabelContacts_${_ranks}ranks COMMAND ${MPIEXEC_EXECUTABLE}
      ${MPIEXEC_NUMPROC_FLAG} ${_ranks} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_label_contacts_hip> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(HIP_LabelContacts_${_ranks}ranks PROPERTIES RUN_SERIAL TRUE TIMEOUT 180)
  endif()
endforeach()
