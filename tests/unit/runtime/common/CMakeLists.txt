# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

target_sources(openpfc-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_backend_from_string.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mpi_timer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpu_affinity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_launch_rank.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_grain_transfer.cpp
)

# The field transaction is header-only and instantiated by each device compiler.
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
    add_executable(test_grain_transfer_cuda grain_transfer.cu)
    target_include_directories(test_grain_transfer_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(test_grain_transfer_cuda PRIVATE Catch2::Catch2WithMain
        openpfc_gpu_compile_defs CUDA::cudart)
    set_target_properties(test_grain_transfer_cuda PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME CUDA_GrainTransfer COMMAND test_grain_transfer_cuda)
endif()

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
    add_executable(test_grain_transfer_hip grain_transfer.hip)
    target_include_directories(test_grain_transfer_hip PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(test_grain_transfer_hip PRIVATE Catch2::Catch2WithMain
        openpfc_gpu_compile_defs hip::host)
    set_target_properties(test_grain_transfer_hip PROPERTIES
        HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
    add_test(NAME HIP_GrainTransfer COMMAND test_grain_transfer_hip)
endif()
