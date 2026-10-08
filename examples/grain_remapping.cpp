// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "grain_remapping.hpp"
#include <iostream>

int main() {
  for (bool seam : {false, true}) {
    auto state = grain_example::initial(seam);
    for (int step = 0; step < 6; ++step)
      if (grain_example::advance(state) != pfc::grain::remapping::Status::Success)
        return 1;
    if (state.grains[0].slot == state.grains[1].slot) return 2;
  }
  std::cout << "2D grain remapping example passed\n";
}
