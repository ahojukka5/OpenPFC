// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file cli_options.hpp
 * @brief Fail-closed `--key=value` parsing for CLI-driven applications.
 *
 * @details
 * JSON/TOML sessions already live in the frontend. Standalone science
 * drivers still need named flags without inventing a second parser per
 * app. Unknown keys are an error after the driver has queried every
 * option it understands: silently ignoring `--lamda=2` would run at the
 * default and look like a verified result.
 *
 * Bare `--flag` stores `"1"`. `--help` / `-h` set @ref help and are not
 * keys. Positional tokens are rejected.
 *
 * @code
 * #include <openpfc/frontend/utils/cli_options.hpp>
 *
 * pfc::utils::CliOptions opt(argc, argv);
 * if (opt.help()) { print_usage(); return 0; }
 * const int nx = opt.integer("nx", 128);
 * opt.require_all_consumed();
 * @endcode
 */

#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace pfc::utils {

/// Parsed `--key=value` pairs with typed lookup and unknown-key detection.
class CliOptions {
public:
  CliOptions(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
      const std::string a(argv[i]);
      if (a == "-h" || a == "--help") {
        m_help = true;
        continue;
      }
      if (a.rfind("--", 0) != 0) {
        throw std::invalid_argument("unexpected positional argument: " + a);
      }
      const auto eq = a.find('=');
      if (eq == std::string::npos) {
        m_kv[a.substr(2)] = "1";
      } else {
        m_kv[a.substr(2, eq - 2)] = a.substr(eq + 1);
      }
    }
  }

  [[nodiscard]] bool help() const noexcept { return m_help; }

  /// True if @p key was on the command line. Does not mark the key used.
  [[nodiscard]] bool has(const std::string &key) const {
    return m_kv.find(key) != m_kv.end();
  }

  double real(const std::string &key, double dflt) {
    const auto it = m_kv.find(key);
    if (it == m_kv.end()) {
      return dflt;
    }
    m_used.insert(key);
    return std::atof(it->second.c_str());
  }

  int integer(const std::string &key, int dflt) {
    const auto it = m_kv.find(key);
    if (it == m_kv.end()) {
      return dflt;
    }
    m_used.insert(key);
    return std::atoi(it->second.c_str());
  }

  bool flag(const std::string &key, bool dflt) {
    const auto it = m_kv.find(key);
    if (it == m_kv.end()) {
      return dflt;
    }
    m_used.insert(key);
    return it->second != "0" && it->second != "false";
  }

  std::string text(const std::string &key, const std::string &dflt) {
    const auto it = m_kv.find(key);
    if (it == m_kv.end()) {
      return dflt;
    }
    m_used.insert(key);
    return it->second;
  }

  /// Throw if any supplied key was never queried with a typed lookup.
  void require_all_consumed() const {
    std::string bad;
    for (const auto &[k, v] : m_kv) {
      if (m_used.find(k) == m_used.end()) {
        bad += (bad.empty() ? "" : ", ") + k;
      }
    }
    if (!bad.empty()) {
      throw std::invalid_argument("unknown option(s): " + bad);
    }
  }

private:
  std::map<std::string, std::string> m_kv;
  std::set<std::string> m_used;
  bool m_help{false};
};

} // namespace pfc::utils
