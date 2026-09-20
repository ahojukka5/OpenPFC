# Full-continuation fixed-domain temporal/resolution control

Follow-up to #92: the frozen 256 probe at identical pseudo-time 1.6 ends
MAX_STEPS in both arms, with all elasticity solves passing. dt .04 raises J
.00304835 -> .0285778 (25 positive CSV increments); dt .01 lowers J to
.00302162 (zero positive CSV increments). This is bounded stability evidence,
not convergence or a general nonlinear stability guarantee.

Run paired fresh-start 128x128x242 dx .5 and 256x256x484 dx .25 cases.
Use the exact shared initial field underlying converged 64/128, not the
warm-start state600 used in the probe. 128 initial SHA
 ecf733fbbbe8e0f831bc3daa6352090c6d1775eab82d58f4b6f28d150ff8f65f.
256 is one periodic factor-two refinement of this field, using immutable
refiner a342b407005c778703704ee0b39fcd1faa3e9877. Preserve input recipe/hashes.
Physical domain stays 64x64x121; target and material coefficients stay fixed.

Both arms dt .01, normalize0, max_delta .04, project_volume1; Esolid1/Evoid.002,
nu .3, volume .2575865186, lambda_volume1, epsilon2, elasticity limit400.
Continuation: lambda_reg .05 -> .2; SIMP1 ->2, 1197 states. This preserves
last-ramp pseudo-time (1197-1)*.01=(300-1)*.04=11.96 exactly. The frozen flag
starts one discrete step later (11.97 instead of12.00); report this discretization
difference. Matching128dt.01 isolates the temporal/continuation change before
attributing a 128-to256 difference to spatial resolution.

Convergence thresholds 2.5e-5/2.5e-7/2.5e-5; window80+hold400. These are the
probe's quarter-size per-step thresholds and fourfold observation period.
No threshold loosening; same target. maxsteps20000 is a safety work ceiling,
never a convergence certificate. Dump every80 accepted states (pseudo-time.8),
including exact terminal snapshot, continuous and strict>.5 endpoint fields.
Compare exact material/threshold C and compliance, grey/volume, topology,
percolation, opening diagnostics, iteration/pseudo-time and wall/GPU/memory.

Use qualified checkpoint binary source43d0d4224340532594f2ae6f618a7e661cbd0ff9,
SHA590a350fae0cc8d27511b415aeb1337bea72724fe0c8d8c814127b90ea674fa6.
Canonical build22183109 passes139/139 plusPython; replay22183155 passes CPU2
and HIP8-rank field/history/active-hold equivalence and fault checks. Actual
same-rank production restart must retain immutable arguments and checkpoint
history. No different-rank equivalence claimed. PR #102 is merged; this
protocol still pins executable `43d0d422` rather than floating with master.

128:2nodes16GCD,256:16nodes128GCD, accountproject_462001245, standard-g.
Initial allocation24h with stop-after1500. This puts planned interruption after
continuation. Resume same source/ranks/parameters from published CURRENT in
successive allocations, with distinct CSV/logs and shared checkpoint/field root.
Each allocation verifies CURRENT next_step/termination before resume; no
automatic restart of terminal errors or scientific retuning. If walltime hits,
inspect published checkpoint and retain only its authoritative CSV prefix.
Stop-after is operational, not scientific termination. r2 **22185790**
COMPLETED RUNNING at 3000 (`C12=+0.00959`, RMS 3.39e-5). Linear last-50
quiet then receded through 3158, 4275, 4659, 5050; projected hold end
is now ~5530, past r5 `--stop-after=5500`. That receding quiet is the
slope-to-zero signature of an RMS plateau, not a forecast of CONVERGED.
Late exponential fits of `design_rms` (last 100/200/400) asymptote at
3.09/3.07/3.02e-5, **above** `tol-design=2.5e-5`. `dC_rel` and `dJ_rel`
are already below tol. `C12>0` throughout. Grey still ~0.68, so a later
decay phase is not ruled out. Evidence:
`2026-09-20-rms-plateau.json`. Dump 0042 (step 3360) `h>0.5` solid has
4 periodic components, no x-percolation, solid fraction 0.162 against
prescribed volume 0.258; largest piece is 26% of the solid. Uniform n0
128 CONVERGED had 2 components and xyz percolation. Evidence:
`2026-09-20-128-topology.json`. Not CONVERGED.

r3 **22185871** (`dev-g`, stop-after 4500) is RUNNING. r4 **22191420**
is **afterany** r3 (`dev-g`, 5000, TimeLimit 2h50) so a wall-clock on
r3 still continues the chain. r5 **22192220** is afterany r4 on
`small-g` (5500, 2h30) because `dev-g` MaxSubmit=2 is filled by r3+r4.
Do **not** queue r6 unless the late exponential asymptote drops below
2.5e-5. Do **not** loosen tolerances. Each resume script refuses a
non-RUNNING checkpoint.

A CONVERGED checkpoint also copies the native `43d0d422`
`h_final_material.json` / `h_thresh_material.json` into campaign logs,
then triggers a PR #90 `--max-steps=0` material eval of the certified
`h_final` (dx 0.5 / 0.25, not the amplifying dx=1 128 field). That eval
is skipped if the allocation stopped RUNNING. Final consumers run only
when the full hold passes. Preserve failures; do not silently reset the
history.

Return a genuinely converged pair or a diagnosed numerical obstruction. Do not
claim asymptotic resolution credibility from only two grids. No larger grid
or different optimization method is authorized by this bounded protocol.
