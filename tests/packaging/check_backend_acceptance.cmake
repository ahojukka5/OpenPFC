# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

function(check_command)
  execute_process(COMMAND ${ARGV} RESULT_VARIABLE status
    OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "Backend acceptance failed (${status}): ${output}\n${error}")
  endif()
endfunction()

# Qualification requires the actual configure-time source identity. A dirty
# development build remains usable, but cannot produce a qualified receipt.
file(READ "${build}/share/openpfc/${configuration}/build-manifest.json" manifest)
string(JSON state GET "${manifest}" source_state)
if(NOT state STREQUAL "clean")
  message("NOT_QUALIFIED: SDK source state is ${state}")
  return()
endif()
if(NOT EXISTS "${launcher}")
  message("NOT_QUALIFIED: MPI launcher unavailable")
  return()
endif()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef label)
set(root "${build}/backend-acceptance-${label}")
check_command("${CMAKE_COMMAND}" --install "${build}" --config "${configuration}"
  --prefix "${root}/install")
file(RENAME "${root}/install" "${root}/relocated")
file(COPY "${source}/tests/packaging/backend_acceptance" DESTINATION "${root}")
set(link_args)
if(link_coverage)
  list(APPEND link_args "-DCMAKE_EXE_LINKER_FLAGS=--coverage")
endif()
check_command("${CMAKE_COMMAND}" -S "${root}/backend_acceptance" -B "${root}/consumer"
  -G Ninja -Dbackend=cpu "-DCMAKE_PREFIX_PATH=${root}/relocated"
  "-DCMAKE_C_COMPILER=${cc}" "-DCMAKE_CXX_COMPILER=${cxx}"
  "-DCMAKE_BUILD_TYPE=${configuration}" "-DHeffte_DIR=${Heffte_DIR}"
  "-Dnlohmann_json_DIR=${nlohmann_json_DIR}" "-Dtomlplusplus_DIR=${tomlplusplus_DIR}"
  ${link_args})
check_command("${CMAKE_COMMAND}" --build "${root}/consumer" --parallel 2)
set(launch_command "\"${launcher}\"")
foreach(flag IN LISTS launcher_flags)
  string(APPEND launch_command " \"${flag}\"")
endforeach()
# Open MPI can execute this tiny 960-cell case on workstations with fewer
# than four slots. Other MPI implementations retain their normal launcher.
execute_process(COMMAND "${launcher}" --version OUTPUT_VARIABLE mpi_version ERROR_QUIET)
if(mpi_version MATCHES "Open MPI|OpenRTE")
  string(APPEND launch_command " --oversubscribe")
endif()
execute_process(COMMAND "${python}" "${root}/backend_acceptance/run_matrix.py"
  --cpu "${root}/consumer/backend_acceptance" --launcher "${launch_command}"
  "--count-flag=${count_flag}" --output "${root}/results" --timeout 60
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 210)
file(WRITE "${root}/runner.log" "${output}\n${error}")
if(NOT status EQUAL 77)
  message(FATAL_ERROR "Expected partial CPU matrix, got ${status}; see ${root}/runner.log")
endif()
file(READ "${root}/results/matrix.json" report)
string(JSON result GET "${report}" status)
if(NOT result STREQUAL "NOT_QUALIFIED")
  message(FATAL_ERROR "CPU matrix did not retain missing GPU qualification")
endif()
foreach(index RANGE 0 8)
  string(JSON backend GET "${report}" rows ${index} backend)
  string(JSON result GET "${report}" rows ${index} status)
  if(backend STREQUAL "cpu")
    if(NOT result STREQUAL "PASS")
      message(FATAL_ERROR "CPU acceptance row ${index} failed; see ${root}/results")
    endif()
  elseif(NOT result STREQUAL "NOT_QUALIFIED")
    message(FATAL_ERROR "Absent ${backend} backend claimed qualification")
  endif()
endforeach()
message(STATUS "CPU 1/2/4 acceptance passed; CUDA/HIP remain unqualified. Receipts: ${root}/results")
