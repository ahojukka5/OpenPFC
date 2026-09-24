# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
cmake_minimum_required(VERSION 3.21)
project(openpfc_spectral_consumer LANGUAGES C CXX)
find_package(OpenPFC CONFIG REQUIRED)
add_executable(spectral_consumer main.cpp)
# A spectral translation unit includes HeFFTe. The ROCm HeFFTe build also
# needs that package's HIP usage requirements, which OpenPFC keeps private.
target_link_libraries(spectral_consumer PRIVATE OpenPFC::openpfc Heffte::Heffte)
