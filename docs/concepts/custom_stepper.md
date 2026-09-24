<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Custom stepper lifecycle

Use this when the numerical step is yours and you still want the OpenPFC
clock, field modifiers, and save hooks.

`SpectralETDSession` remains the JSON session for spectral ETD physics.
It is not a requirement for a custom step.

## What you call

| Need | API |
|------|-----|
| Fixed step, clock already built | `pfc::sim::run` |
| Accept or reject a step | `pfc::sim::run_attempts` |
| Fields, modifiers, save, checkpoint, profiling | `pfc::sim::SimulationLifecycle` |

Header: `openpfc/kernel/simulation/simulation_lifecycle.hpp`.

## Who owns what

OpenPFC owns the clock, the accept/reject transition, named-field lookup,
and application of `FieldModifier` initial and boundary conditions.

The application owns the stepper, material laws, derived observables, and
the decision of which columns or files those observables use.

`SnapshotSeries` and `DiagnosticsSeries` are not members of the lifecycle.
Call them from the save observer. They still work with no lifecycle at all.

## Fixed and adaptive loops

`run` commits `Time::next()` before the step. A callback
`step(double t, double interval)` receives the clock interval. That
interval is `min(dt, t1 - t_before)` on the last step. A one-argument
`step(double t)` does not see it.

`run_attempts` opens `begin_attempt` with `Time::get_dt()`, which clips
to `t1` and to the next save. The stepper returns `StepDecision`. The
lifecycle's runner owns `increment_step_success` and
`increment_step_rejection`. A rejection does not call the save observer
or the checkpoint hook. An exception from the step closes the attempt
and propagates.

`SimulationLifecycle::schedule(t0, t1, dt, saveat)` builds that clock.
A non-positive `saveat` still saves the initial state and the final
state.

## Initial conditions and boundary conditions

Bind each real field by name, then add modifiers. Names are not assumed:
nothing in the lifecycle is called `psi`, `h`, or `c`.

```cpp
life.bind_field("density", density);
life.bind_field("solute", solute);
life.add_initial_condition("density", std::make_unique<pfc::Constant>(0.1));
```

Initial modifiers run once, before the first save. Boundary modifiers
run from the whole-step `apply` hook, including on an attempt that is
later rejected.

`prepare_stage(stage_time)` is the stepper's own request for boundary
data at an internal stage. It uses the same modifiers and does not save,
checkpoint, or advance `Time`.

## Save, checkpoint, profiling

```cpp
life.set_save_observer([&](const pfc::Time &now) {
  const int step = pfc::time::increment(now);
  const double t = pfc::time::current(now);
  snapshots.write(step, t);
  diagnostics.write(step, t, columns);
});
life.set_checkpoint_hook([&](const pfc::Time &now) {
  checkpoint.save(state, now);
});
life.set_profiling(&profiling_session); // optional; nullptr leaves it off
```

The checkpoint hook is the integration point for `CheckpointService`.
The lifecycle does not invent a payload. Both hooks run only after an
accepted state that `do_save()` accepts.

Profiling frames each step callback when a session is set. A rejected
attempt is framed, because the step ran. `prepare_stage` is not a frame.
Output and checkpoint progress do not move on a rejection.
