<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Issue #106 pre-analysis decision tests

Frozen **before** harvesting the bulk of the campaign. Do not change
these criteria after seeing winners.

Per-allocation values stay primary. Pooled medians are a summary. A
single allocation with winner/second within 5% is not a ranking.

## Evidence that would support H1 (smooth rank/message scaling)

All of:

1. Neighbouring sampled node counts keep the same protocol order after
   three independent allocations at each count, or the order changes
   only where the pre-execution descriptor (peer count, bytes/rank,
   communicator size) also jumps.
2. A candidate model of the form
   `T_comm = a * n_messages + b * bytes + c * n_peers`
   (or the per-reshape Hockney extension in
   `heffte_comm_plan.py`) fitted **only** on completed #61 families or
   on a declared training subset of #106, leaves residuals without a
   discrete step at 124/248/496.
3. Protocol × order interaction is small relative to the protocol main
   effect (see `order.csv`).

## Evidence that would support H2 (topology-regime effect)

All of:

1. Ranking or local slope changes at the same node-count neighbourhood
   across independent allocations **and** different protocol orders.
2. The change is **not** explained by the pre-execution descriptor
   (the neighbouring counts have essentially the same bytes/peer and
   communicator size).
3. Observed placement metadata actually differs across that boundary
   (host nid span, exposed `xNNNN` features if present). Absence of
   group labels means H2 can remain **unresolved** even if rankings
   change: a change at 124 nodes is not by itself a Slingshot-group
   effect.

## Evidence that would support H3 (order/allocation noise)

Any of:

1. Independent allocations at the same node count elect different
   winners.
2. Winner/second stay within 5% after the planned three repeats.
3. Protocol order position (1–4) accounts for the apparent crossover
   in `order.csv`.

H3 is an admissible negative result and blocks a production selector
based on the old tournament.

## What the first completed measurements discriminate

The first 136-node allocation can only:

- confirm that all four protocols still admit and checksum;
- show whether `alltoall` vs `alltoallv` remain close;
- feed one row to `order.csv`.

It cannot decide H1/H2/H3. The 136-node r2/r3 jobs and the 112/124/128
controls are required before any neighbourhood is called a regime.
