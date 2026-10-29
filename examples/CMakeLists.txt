# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# FFT Backend Benchmark (requires MPI)
if(NOT TARGET grain_remapping_example)
  add_executable(grain_remapping_example grain_remapping.cpp)
  target_link_libraries(grain_remapping_example PRIVATE OpenPFC)
endif()

add_executable(fft_backend_benchmark fft_backend_benchmark.cpp)
set(_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_heffte_tgt heffte)
endif()

target_link_libraries(fft_backend_benchmark PRIVATE OpenPFC ${_heffte_tgt})
if(OpenPFC_ENABLE_CUDA)
  target_compile_definitions(fft_backend_benchmark PRIVATE OpenPFC_ENABLE_CUDA)
endif()
if(OpenPFC_ENABLE_HIP)
  target_compile_definitions(fft_backend_benchmark PRIVATE OpenPFC_ENABLE_HIP)
endif()
if(OpenPFC_ENABLE_HIP_SPECTRAL)
  target_compile_definitions(fft_backend_benchmark PRIVATE OpenPFC_ENABLE_HIP_SPECTRAL)
endif()

add_executable(fft_kspace_helpers_example fft_kspace_helpers_example.cpp)
target_link_libraries(fft_kspace_helpers_example PRIVATE OpenPFC ${_heffte_tgt})

add_executable(diffusion_model_with_custom_initial_condition diffusion_model_with_custom_initial_condition.cpp)
target_link_libraries(diffusion_model_with_custom_initial_condition PRIVATE OpenPFC ${_heffte_tgt})

add_executable(18_sobel_edge_detection 18_sobel_edge_detection.cpp)
target_link_libraries(18_sobel_edge_detection PRIVATE OpenPFC ${_heffte_tgt})

# Teaching chapters own their implementations; established binary names remain.
add_subdirectory(geometry)
add_subdirectory(fourier)
add_subdirectory(heat)
add_subdirectory(differentiation)
add_subdirectory(interpolation)
add_subdirectory(waves)
add_subdirectory(integration)
add_subdirectory(polynomials)
add_subdirectory(coupling)
add_subdirectory(parallel)
add_subdirectory(diagnostics)
add_subdirectory(output)
add_subdirectory(quadrature)
add_subdirectory(elliptic)
add_subdirectory(initialization)
add_subdirectory(configuration)
add_subdirectory(phase-fields)

if(OpenPFC_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(_lesson_check
      "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_lessons.py"
      --fd $<TARGET_FILE:19_explicit_stepper_fd>
      --separated $<TARGET_FILE:15_finite_difference_heat>
      --fourier $<TARGET_FILE:20_explicit_stepper_spectral>
      --exponential $<TARGET_FILE:heat_exponential>
      --differentiation $<TARGET_FILE:periodic_differentiation>
      --output "${PROJECT_SOURCE_DIR}/results/example-checks")
  add_test(NAME example-lessons COMMAND ${_lesson_check})
  set_tests_properties(example-lessons PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 180)
  set(_foundations_check
      "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_foundations.py"
      --bin-dir "${CMAKE_BINARY_DIR}/examples"
      --output "${PROJECT_SOURCE_DIR}/results/example-checks")
  add_test(NAME example-foundations COMMAND ${_foundations_check})
  set_tests_properties(example-foundations PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 180)
  set(_dynamics_check
      "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_dynamics.py"
      --bin-dir "${CMAKE_BINARY_DIR}/examples"
      --output "${PROJECT_SOURCE_DIR}/results/example-checks")
  add_test(NAME example-dynamics COMMAND ${_dynamics_check})
  set_tests_properties(example-dynamics PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 180)
  add_test(NAME example-polynomials
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_polynomials.py"
                   --bin-dir "${CMAKE_BINARY_DIR}/examples"
                   --output "${PROJECT_SOURCE_DIR}/results/example-checks")
  set_tests_properties(example-polynomials PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 180)
  set(_workflows_check "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_workflows.py"
                       --bin-dir "${CMAKE_BINARY_DIR}/examples"
                       --output "${PROJECT_SOURCE_DIR}/results/example-checks")
  add_test(NAME example-workflows COMMAND ${_workflows_check})
  set_tests_properties(example-workflows PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_executable(test_example_eigensystem tests/eigensystem.cpp)
  target_link_libraries(test_example_eigensystem PRIVATE OpenPFC nlohmann_json::nlohmann_json)
  add_test(NAME example-eigensystem-reference COMMAND test_example_eigensystem)
  add_test(NAME example-spectra
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_spectra.py"
                   --bin-dir "${CMAKE_BINARY_DIR}/examples"
                   --output "${PROJECT_SOURCE_DIR}/results/example-checks")
  set_tests_properties(example-spectra PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300)
  add_test(NAME example-clamped
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_clamped.py"
                   --bin-dir "${CMAKE_BINARY_DIR}/examples"
                   --output "${PROJECT_SOURCE_DIR}/results/example-checks")
  set_tests_properties(example-clamped PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 180)
  add_test(NAME example-orr-sommerfeld
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_orr_sommerfeld.py"
                   --bin-dir "${CMAKE_BINARY_DIR}/examples"
                   --output "${PROJECT_SOURCE_DIR}/results/example-checks")
  set_tests_properties(example-orr-sommerfeld PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 180)
  add_test(NAME example-pseudospectra
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_pseudospectra.py"
                   --bin-dir "${CMAKE_BINARY_DIR}/examples"
                   --output "${PROJECT_SOURCE_DIR}/results/example-checks")
  set_tests_properties(example-pseudospectra PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 180)
  add_test(NAME example-disk
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_disk.py"
                   --bin-dir "${CMAKE_BINARY_DIR}/examples"
                   --output "${PROJECT_SOURCE_DIR}/results/example-checks")
  set_tests_properties(example-disk PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 180)
  set(_coordinates_check "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_coordinates.py"
                         --bin-dir "${CMAKE_BINARY_DIR}/examples"
                         --output "${PROJECT_SOURCE_DIR}/results/example-checks")
  add_test(NAME example-coordinates COMMAND ${_coordinates_check})
  set_tests_properties(example-coordinates PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 120)
  add_test(NAME example-evolution
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_evolution.py"
                   --bin-dir "${CMAKE_BINARY_DIR}/examples"
                   --output "${PROJECT_SOURCE_DIR}/results/example-checks")
  set_tests_properties(example-evolution PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 180)
  set(_everyday_check "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_everyday.py"
                      --bin-dir "${CMAKE_BINARY_DIR}/examples"
                      --output "${PROJECT_SOURCE_DIR}/results/example-checks")
  add_test(NAME example-everyday COMMAND ${_everyday_check})
  set_tests_properties(example-everyday PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 240)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-coordinates-mpi COMMAND ${_coordinates_check}
             --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-coordinates-mpi PROPERTIES
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
                         PROCESSORS 2 TIMEOUT 180)
    add_test(NAME example-everyday-mpi COMMAND ${_everyday_check}
             --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-everyday-mpi PROPERTIES
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
                         PROCESSORS 2 TIMEOUT 300)
    add_test(NAME example-lessons-mpi COMMAND ${_lesson_check}
             --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-lessons-mpi PROPERTIES
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
                         PROCESSORS 2 TIMEOUT 240)
    add_test(NAME example-foundations-mpi COMMAND ${_foundations_check}
             --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-foundations-mpi PROPERTIES
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
                         PROCESSORS 2 TIMEOUT 240)
    add_test(NAME example-dynamics-mpi COMMAND ${_dynamics_check}
             --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-dynamics-mpi PROPERTIES
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
                         PROCESSORS 2 TIMEOUT 240)
    add_test(NAME example-workflows-mpi COMMAND ${_workflows_check}
             --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-workflows-mpi PROPERTIES
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
                         PROCESSORS 2 TIMEOUT 360)
  endif()
endif()

add_executable(23_halo_microtiming 23_halo_microtiming.cpp)
target_link_libraries(23_halo_microtiming PRIVATE OpenPFC nlohmann_json::nlohmann_json
                                                  ${_heffte_tgt})
if(OpenPFC_ENABLE_CUDA)
  target_compile_definitions(23_halo_microtiming PRIVATE OpenPFC_ENABLE_CUDA)
endif()
if(OpenPFC_ENABLE_HIP)
  target_compile_definitions(23_halo_microtiming PRIVATE OpenPFC_ENABLE_HIP)
endif()
if(OpenPFC_BUILD_TESTS AND MPIEXEC_EXECUTABLE)
  add_test(NAME halo-microtiming-host-2rank
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                   $<TARGET_FILE:23_halo_microtiming> --nx 16 --iters 4 --warmup 1
                   --output ${CMAKE_CURRENT_BINARY_DIR}/halo_microtiming_smoke.json)
  set_tests_properties(halo-microtiming-host-2rank PROPERTIES
                       ENVIRONMENT
                       "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
endif()

add_executable(25_halo_bandwidth 25_halo_bandwidth.cpp)
target_link_libraries(25_halo_bandwidth PRIVATE OpenPFC ${_heffte_tgt})
if(OpenPFC_ENABLE_CUDA)
  target_compile_definitions(25_halo_bandwidth PRIVATE OpenPFC_ENABLE_CUDA)
endif()
if(OpenPFC_ENABLE_HIP)
  target_compile_definitions(25_halo_bandwidth PRIVATE OpenPFC_ENABLE_HIP)
endif()
if(OpenPFC_BUILD_TESTS AND MPIEXEC_EXECUTABLE)
  add_test(NAME halo-bandwidth-host-2rank
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                   $<TARGET_FILE:25_halo_bandwidth> --sizes 32 --iters 2 --warmup 1)
  set_tests_properties(halo-bandwidth-host-2rank PROPERTIES
                       PASS_REGULAR_EXPRESSION "Uni_GBps"
                       ENVIRONMENT
                       "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
endif()

# Issue #21 research prototype under examples/, not an apps/ executable.
add_subdirectory(ns2d_vorticity)

# Issue #229 verification driver under examples/, not an apps/ executable.
add_subdirectory(taylor_green3d)

if(NOT TARGET coupled_fields_example)
  add_subdirectory(coupled_fields)
endif()
