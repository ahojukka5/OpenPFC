// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file inverse_checkpoint.hpp
 * @brief Durable restart bundle for frozen inverse homogenization.
 *
 * A restart continues the same mathematical optimization: incoming design
 * `h`, previous accepted `h_prev`, absolute iterate index, continuation
 * freeze, tracker window/hold, and previous `J` / `C_H`. It does not
 * reset convergence history or unfreeze SIMP / `lambda_reg`. Grid size
 * and continuation length identify the frozen problem; `--max-steps` is
 * a run budget and may stay at the original scientific ceiling after a
 * walltime restart.
 */

#include <iomanip>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>

#include <inverse_homogenization/inverse_convergence.hpp>

namespace pfc::apps::inverse {

inline constexpr int kInverseCheckpointSchema = 1;
inline constexpr std::string_view kInverseCheckpointMagic =
    "OPENPFC_INVERSE_CHECKPOINT";

struct InverseCheckpoint {
  int schema{kInverseCheckpointSchema};
  int nx{0}, ny{0}, nz{0};
  int next_step{0};
  int continuation_steps{0};
  int max_steps{0};
  int conv_window{0};
  int verify_steps{0};
  int quiet_count{0};
  int verify_left{0};
  int candidate{0};
  int verified{0};
  int have_prev{0};
  int n_snap{0};
  int last_dumped{-1};
  double J_prev{0.0};
  double C_prev[36]{};
};

template <typename Tensor> inline void store_voigt6(double *out, const Tensor &C) {
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j) out[6 * i + j] = C(i, j);
}

template <typename Tensor> inline void fill_voigt6(Tensor &C, const double *in) {
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j) C(i, j) = in[6 * i + j];
}

inline void apply_tracker(const InverseCheckpoint &ck, ConvergenceTracker &tr) {
  tr.cfg.continuation_steps = ck.continuation_steps;
  tr.cfg.max_steps = ck.max_steps;
  tr.cfg.conv_window = ck.conv_window;
  tr.cfg.verify_steps = ck.verify_steps;
  tr.quiet_count = ck.quiet_count;
  tr.verify_left = ck.verify_left;
  tr.candidate = ck.candidate != 0;
  tr.verified = ck.verified != 0;
}

inline void capture_tracker(InverseCheckpoint &ck, const ConvergenceTracker &tr,
                            int next_step) {
  ck.next_step = next_step;
  ck.continuation_steps = tr.cfg.continuation_steps;
  ck.max_steps = tr.cfg.max_steps;
  ck.conv_window = tr.cfg.conv_window;
  ck.verify_steps = tr.cfg.verify_steps;
  ck.quiet_count = tr.quiet_count;
  ck.verify_left = tr.verify_left;
  ck.candidate = tr.candidate ? 1 : 0;
  ck.verified = tr.verified ? 1 : 0;
}

[[nodiscard]] inline bool checkpoint_matches_problem(const InverseCheckpoint &ck,
                                                     int nx, int ny, int nz,
                                                     int continuation_steps) {
  return ck.nx == nx && ck.ny == ny && ck.nz == nz &&
         ck.continuation_steps == continuation_steps;
}

inline bool write_checkpoint_text(std::ostream &os, const InverseCheckpoint &ck) {
  os << kInverseCheckpointMagic << ' ' << ck.schema << '\n';
  os << "nx " << ck.nx << "\nny " << ck.ny << "\nnz " << ck.nz << '\n';
  os << "next_step " << ck.next_step << '\n';
  os << "continuation_steps " << ck.continuation_steps << '\n';
  os << "max_steps " << ck.max_steps << '\n';
  os << "conv_window " << ck.conv_window << '\n';
  os << "verify_steps " << ck.verify_steps << '\n';
  os << "quiet_count " << ck.quiet_count << '\n';
  os << "verify_left " << ck.verify_left << '\n';
  os << "candidate " << ck.candidate << '\n';
  os << "verified " << ck.verified << '\n';
  os << "have_prev " << ck.have_prev << '\n';
  os << "n_snap " << ck.n_snap << '\n';
  os << "last_dumped " << ck.last_dumped << '\n';
  os << "J_prev " << std::setprecision(17) << ck.J_prev << '\n';
  os << "C_prev";
  for (double v : ck.C_prev) os << ' ' << std::setprecision(17) << v;
  os << '\n';
  return static_cast<bool>(os);
}

[[nodiscard]] inline bool read_checkpoint_text(std::istream &in,
                                               InverseCheckpoint &ck) {
  std::string magic;
  int schema = 0;
  if (!(in >> magic >> schema) || magic != kInverseCheckpointMagic ||
      schema != kInverseCheckpointSchema)
    return false;
  ck = InverseCheckpoint{};
  ck.schema = schema;
  std::string key;
  bool saw_c = false;
  while (in >> key) {
    if (key == "nx") {
      if (!(in >> ck.nx)) return false;
    } else if (key == "ny") {
      if (!(in >> ck.ny)) return false;
    } else if (key == "nz") {
      if (!(in >> ck.nz)) return false;
    } else if (key == "next_step") {
      if (!(in >> ck.next_step) || ck.next_step < 0) return false;
    } else if (key == "continuation_steps") {
      if (!(in >> ck.continuation_steps)) return false;
    } else if (key == "max_steps") {
      if (!(in >> ck.max_steps)) return false;
    } else if (key == "conv_window") {
      if (!(in >> ck.conv_window)) return false;
    } else if (key == "verify_steps") {
      if (!(in >> ck.verify_steps)) return false;
    } else if (key == "quiet_count") {
      if (!(in >> ck.quiet_count)) return false;
    } else if (key == "verify_left") {
      if (!(in >> ck.verify_left)) return false;
    } else if (key == "candidate") {
      if (!(in >> ck.candidate)) return false;
    } else if (key == "verified") {
      if (!(in >> ck.verified)) return false;
    } else if (key == "have_prev") {
      if (!(in >> ck.have_prev)) return false;
    } else if (key == "n_snap") {
      if (!(in >> ck.n_snap) || ck.n_snap < 0) return false;
    } else if (key == "last_dumped") {
      if (!(in >> ck.last_dumped)) return false;
    } else if (key == "J_prev") {
      if (!(in >> ck.J_prev)) return false;
    } else if (key == "C_prev") {
      for (double &v : ck.C_prev)
        if (!(in >> v)) return false;
      saw_c = true;
    } else {
      return false;
    }
  }
  return saw_c && ck.nx > 0 && ck.ny > 0 && ck.nz > 0;
}

[[nodiscard]] inline std::string
format_checkpoint_text(const InverseCheckpoint &ck) {
  std::ostringstream os;
  write_checkpoint_text(os, ck);
  return os.str();
}

} // namespace pfc::apps::inverse
