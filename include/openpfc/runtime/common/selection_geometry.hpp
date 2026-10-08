// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace pfc::core::detail {
/// Checked geometry for the signed grid dimension of a compact selection.
constexpr int selection_blocks(std::size_t count, int block_size) {
  if (block_size <= 0) throw std::invalid_argument("gather: invalid block size");
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("gather: selection exceeds launch index range");
  const auto width = static_cast<std::size_t>(block_size);
  return static_cast<int>(count / width + (count % width != 0));
}
} // namespace pfc::core::detail
