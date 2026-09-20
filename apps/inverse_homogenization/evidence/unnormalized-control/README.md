<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Fixed 160-state unnormalized control

This is numerical evidence for #74 and the starting comparator for the
separate longer hold #83. It is **MAX_STEPS**, not convergence.

All elasticity solves pass. The exact saved final field matches the final
accepted snapshot; threshold output is its componentwise `h > 0.5` indicator.
The binary, target and initial field hashes match normalized job 22178459.
No target, regularization, cap, projection, timestep or tolerance changes;
only the existing normalization option changes from 1 to 0.

The mean of the last 50 design RMS values is 0.000389786, versus 0.01490253
in the normalized control. One-/two-step field RMS are 0.000391139/0.000781959:
the small lag-two signature of the normalized trajectory is absent in this
window. J decreases at every recorded step at CSV precision. This does not
prove asymptotic convergence or a general descent guarantee. Grey fraction
is higher (tail mean 0.658664), so smaller updates alone are not better material.

The exact initial field and final accepted 160-state field are small compressed
fixtures, allowing replay without an ephemeral input path. `history160.csv`
preserves all objective components and convergence diagnostics. The original
run recipe and package modules are preserved; `provenance.json` binds files,
source and binary identities. Large trajectories remain in scratch and are
recreated with the pinned code, frozen input and recorded command.

A separate 2000-state experiment is predeclared in
`../2026-09-20-unnormalized-hold.md`. It must meet the original convergence
certificate; its larger budget is not a matched-budget performance comparison.

Regenerate the numerical summary with
`apps/inverse_homogenization/scripts/audit_unnormalized_control.py RUN BASELINE OUTPUT`.
The audit checks the complete 160-row run and exact final-field identity.
