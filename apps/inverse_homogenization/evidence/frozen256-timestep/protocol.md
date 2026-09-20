# Fixed-domain 256-grid timestep qualification

Engineering/numerical gate for OpenPFC #81, not a completed resolution result.
Question: does parabolic dt scaling avoid the expected fine-grid explicit
regularization instability without changing target or energy coefficients?

Use the exact accepted step600 snapshot from n0fixed128 job22180901:
fields/n0fixed128_h_0030.bin,
SHA256 4009a9b2dfc6830c87f0a4c4b53c3fd832dcb919afadf17fa7f6d091bd28b880.
Source6deeb76cc23be832db3aa33f3215b3ba14bd414e,
immutable binary SHA b7659d22ff8ef5dd338d6e2c53b680bcd8a50254ff69206d8a084d964a7e7b5a.
The field is not converged. Periodic trilinear midpoint interpolation from
128x128x242/dx0.5 to256x256x484/dx0.25 preserves the physical64x64x121cell,
coarse samples, mean and bounds. Use refine_periodic_initial.py from immutable
source a342b407005c778703704ee0b39fcd1faa3e9877; preserve hashes and recipe.

Both arms load that same refined field. Keep the same frozen Yang target,
Esolid1/nu.3/Evoid.002/nu.3, volume.2575865186, lambda_volume1,
lambda_reg.2, epsilon2, SIMP2, normalize0, project_volume1, max_delta.04,
elasticity limit400. No continuation: this is a frozen-state stability probe,
not a substitute for the fresh full-continuation resolution campaign.

Arm A dt.04, maxsteps41, dump_every1; original stopping thresholds/window20+100.
Arm B dt.01, maxsteps161, dump_every4; thresholds divided by4 and window/hold
multiplied by4 (2.5e-5,2.5e-7,2.5e-5;80+400). This prevents a smaller per-step
motion from weakening the stopping contract. Both intended accepted endpoints
span pseudo-time1.6. MAX_STEPS is expected and is not convergence. If either
terminates early, report the reason and actual pseudo-time explicitly.

For the spectral Laplacian, diffusion-only explicit Euler factor is
 dt*lambda_reg*epsilon*3*pi^2/dx^2.
At dx.25 this is7.5799 fordt.04 and1.8950 fordt.01; the linear diffusion bound
is2. This predicts a necessary restriction, not stability of the full nonlinear
coupled update. Failure ofdt.01 is preserved; no adaptive parameter fishing.

Run sequential arms on16LUMI-Gnodes/128GCDs, accountproject_462001245.
This keeps local voxel count equal to the healthy128/16GCDcase (~247808/GCD).
Existing256/64GCD exploratory timing is~15.6s/step and memory~0.46GiB/GCD;
we test128GCD throughput rather than assuming that scaling will be efficient.
The different existing problem is not a numerical comparator. Preserve HIP_MEM,
all solve/failure logs, scheduler/runtime metadata and common binary identity.

Compare full objective components versus pseudo-time, monotonicity violations,
design/tensor increments, elasticity success, projection/clipping effects where
observable, field bounds/volume/grey, step costs and endpoint tensors. Exact
accepted fields and strict thresholds must be preserved. No canonical256
convergence claim or animation follows from this bounded probe.

Stop after both arms. If scaled dt is healthy, use evidence to freeze a full
production protocol with qualified checkpoint/restart before depending on
walltime recovery. Do not modify live64/128runs or increase any solver budget.
