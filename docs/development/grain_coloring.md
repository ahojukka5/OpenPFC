<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Deterministic graph recoloring reference

The internal header
[`grain_coloring.hpp`](../../include/openpfc/runtime/cpu/detail/grain_coloring.hpp)
provides an independent host reference for explicit small undirected graphs.
It does not own grain identities, structured-grid contact detection, field
transfers or simulation scheduling. Dense vertex indices are adapter indices,
and zero-based colors denote a caller's fixed slot budget. The `detail` path
is not a public stability promise.

`make_graph` canonicalizes edges. Directly supplied adjacency lists must be
simple and symmetric; invalid endpoints, loops, repeated neighbors and
asymmetry throw `std::invalid_argument`. Graph storage order does not affect
tie breaks. Relabeling vertices may change a legal solution because index is
the final deterministic tie break.

`global_greedy_coloring` provides two practical single-pass global policies:
DSATUR selection or static descending-degree largest-first order, both with
ascending-index ties and first available color. A greedy dead end reports
`search_limit`; it does not prove the graph impossible. These modes are
separate from the bounded complete-search oracle, so a comparison can time
the practical heuristic, any explicitly chosen fallback and label alignment
separately.

`global_coloring` uses DSATUR vertex selection: greatest number of distinct
assigned neighbor colors, greatest degree, then lowest vertex index. It
tries colors in ascending order and backtracks until the first proper
coloring is found. It neither minimizes migration nor enumerates every
proper coloring after success. Exhausted complete search returns
`infeasible`; exhausting the candidate-assignment budget returns
`search_limit`. Complete search is exponential in the worst case and is a
reference, not a scalable production optimizer.

`incremental_coloring` preserves a proper input without any search. Otherwise
it takes the first conflicting edge in canonical order, tries its first
endpoint and then its second. Candidates are ordered by number of blockers,
slot population and slot index. A candidate is reserved before recursively
moving blocking neighbors. Locked ancestors cannot be displaced, and failed
branches restore their tentative assignments. The default depth limit is
five; zero allows direct moves. This is a bounded local heuristic, with no
minimum-migration or completeness guarantee. A depth-limited dead end or an
exhausted shared attempt budget returns `search_limit`, including on graphs
that are actually infeasible. Only the global complete search can certify
infeasibility.

Every successful result passes a final proper-coloring check. Failure
returns no coloring, and both solvers leave caller inputs unchanged. A
caller must reject failure before generating field-move instructions.
The attempt counter measures candidate assignments, not time or complexity.

`migration` counts changed vertices and sums optional finite nonnegative
caller weights. These are migration proxies; actual moved values, bytes,
backgrounds and diffuse-tail errors require a separate field-transfer
contract. Numeric color permutations describe the same partition.
`align_colors` uses minimum-cost Hungarian assignment to match a new
partition's labels to existing slots before interpreting migration. It
minimizes the supplied weighted proxy over permutations; without weights it
minimizes changed vertices. It preserves proper coloring and does not change
the partition or find a better graph coloring. Its storage is quadratic and
assignment cost cubic in the palette size. Tie breaks are deterministic.

The Catch2 cases in
[`test_grain_coloring.cpp`](../../tests/unit/runtime/test_grain_coloring.cpp)
use an independent mixed-radix exhaustive oracle for every labeled simple
graph through five vertices and palettes of one to three slots. They check
feasibility, legality and lower bounds on migration without assuming the
heuristic is optimal. Explicit cases check saturated chained reassignment,
cycles, a clique, deterministic adjacency order, vertex permutation,
transactional limits, empty graphs and malformed inputs. All initial
three-slot assignments on four-vertex graphs check local-success legality.
All old/new three-vertex assignments check label alignment against every
permutation. These tests verify software behavior; they do not establish
physical correctness or speed.
