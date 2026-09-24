# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
cmake_minimum_required(VERSION 3.21)
project(openpfc_fd_consumer LANGUAGES C CXX)
find_package(OpenPFC CONFIG REQUIRED)
add_executable(fd_consumer main.cpp)
target_link_libraries(fd_consumer PRIVATE OpenPFC::openpfc)
