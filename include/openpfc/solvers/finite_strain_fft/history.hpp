// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file history.hpp
 * @brief Transactional per-point state for a history-dependent material.
 *
 * The lifecycle is `committed -> trial -> accept/reject -> committed`.
 * A Newton iteration may edit the trial copy. `reject` restores that copy
 * from the last accepted state, so a failed increment cannot advance the
 * committed history. The container does not know which constitutive law
 * owns the state.
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pfc::finite_strain {

template <typename State> class TransactionalHistory {
public:
  TransactionalHistory() = default;

  explicit TransactionalHistory(std::vector<State> committed)
      : committed_(std::move(committed)), trial_(committed_) {}

  [[nodiscard]] std::size_t size() const noexcept { return committed_.size(); }

  [[nodiscard]] bool trial_open() const noexcept { return open_; }

  [[nodiscard]] const State &committed(std::size_t index) const {
    return committed_.at(index);
  }

  [[nodiscard]] const State &trial(std::size_t index) const {
    return trial_.at(index);
  }

  [[nodiscard]] State &trial(std::size_t index) {
    if (!open_) {
      throw std::logic_error("material trial state is not open");
    }
    return trial_.at(index);
  }

  /// Copy the committed state into the trial buffer and open it.
  void begin() {
    trial_ = committed_;
    open_ = true;
  }

  /// Publish the trial buffer. The trial must have been opened.
  void accept() {
    if (!open_) {
      throw std::logic_error("material history cannot accept a closed trial");
    }
    committed_ = trial_;
    open_ = false;
  }

  /// Drop the trial buffer. The committed state is left unchanged.
  void reject() {
    trial_ = committed_;
    open_ = false;
  }

private:
  std::vector<State> committed_;
  std::vector<State> trial_;
  bool open_ = false;
};

} // namespace pfc::finite_strain
