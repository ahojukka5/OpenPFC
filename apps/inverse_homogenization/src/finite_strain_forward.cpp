// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file finite_strain_forward.cpp
 * @brief CLI for the 2-D finite-strain forward ladder (OpenPFC #55).
 *
 * Prints FINITE_STRAIN_CHECKSUM from the shipped run_declared_ladder()
 * entry point. Optional --evidence writes the same numbers as JSON.
 */

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <inverse_homogenization/finite_strain_forward.hpp>

namespace {

void write_evidence(const std::string &path,
                    const pfc::apps::inverse::fs::LadderReport &r) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot write evidence " + path);
  }
  out << std::setprecision(16);
  out << "{\n";
  out << "  \"schema\": \"openpfc-inverse-evidence/v1\",\n";
  out << "  \"openpfc_issue\": 55,\n";
  out << "  \"research_issue\": \"ahojukka5/research#484\",\n";
  out << "  \"question\": \"Does a 2-D plane-strain finite-strain forward "
         "path with P22=0 report a tangent Poisson that matches FD of the "
         "same shipped F22(F11)?\",\n";
  out << "  \"kinematics\": {\n";
  out << "    \"F33\": 1.0,\n";
  out << "    \"stress\": \"first-Piola-Kirchhoff\",\n";
  out << "    \"strain\": \"logarithmic\",\n";
  out << "    \"nu_t\": \"-d ln F22 / d ln F11\",\n";
  out << "    \"not_small_strain_CH\": true\n";
  out << "  },\n";
  out << "  \"lame_stiff\": {\"lambda\": " << r.lame_stiff.lambda
      << ", \"mu\": " << r.lame_stiff.mu << "},\n";
  out << "  \"lame_compliant\": {\"lambda\": " << r.lame_compliant.lambda
      << ", \"mu\": " << r.lame_compliant.mu << "},\n";
  out << "  \"plane_strain_small_nu\": " << r.small_strain_nu << ",\n";
  out << "  \"all_converged\": " << (r.all_converged ? "true" : "false") << ",\n";
  out << "  \"all_stable\": " << (r.all_stable ? "true" : "false") << ",\n";
  out << "  \"rungs\": [\n";
  for (std::size_t i = 0; i < r.rungs.size(); ++i) {
    const auto &u = r.rungs[i];
    out << "    {\"kind\": \"" << u.kind << "\", \"F11\": " << u.F11
        << ", \"F22\": " << u.F22 << ", \"P11\": " << u.P11 << ", \"P22\": " << u.P22
        << ", \"nu_t\": " << u.nu_t << ", \"residual\": " << u.residual
        << ", \"newton_iters\": " << u.newton_iters
        << ", \"converged\": " << (u.converged ? "true" : "false")
        << ", \"stable\": " << (u.stable ? "true" : "false") << "}";
    if (i + 1 < r.rungs.size()) out << ",";
    out << "\n";
  }
  out << "  ],\n";
  out << "  \"interpretation\": \"Homogeneous neo-Hookean and StVK uniaxial "
         "paths with P22=0 converge; small-strain nu_t matches "
         "lambda/(lambda+2 mu). A two-material y-laminate (E,nu)= "
         "(1,0.3) and (0.1,0.1) has a distinct nu_t from the stiff "
         "homogeneous cell. This is not small-strain C_H and not inverse "
         "design.\"\n";
  out << "}\n";
}

} // namespace

int main(int argc, char **argv) {
  std::string evidence;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--evidence" && i + 1 < argc) {
      evidence = argv[++i];
    } else if (a.rfind("--evidence=", 0) == 0) {
      evidence = a.substr(11);
    } else if (a == "--help" || a == "-h") {
      std::cout << "openpfc_finite_strain_forward [--evidence PATH]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << a << "\n";
      return 2;
    }
  }

  const auto report = pfc::apps::inverse::fs::run_declared_ladder();
  std::cout << std::setprecision(12);
  std::cout << "FINITE_STRAIN_CHECKSUM"
            << " n=" << report.rungs.size()
            << " converged=" << (report.all_converged ? 1 : 0)
            << " stable=" << (report.all_stable ? 1 : 0)
            << " nu_small=" << report.small_strain_nu << "\n";
  for (const auto &u : report.rungs) {
    std::cout << u.kind << " F11=" << u.F11 << " F22=" << u.F22 << " P22=" << u.P22
              << " nu_t=" << u.nu_t << " iters=" << u.newton_iters << "\n";
  }
  if (!evidence.empty()) {
    write_evidence(evidence, report);
    std::cout << "wrote " << evidence << "\n";
  }
  return report.all_converged && report.all_stable ? 0 : 1;
}
