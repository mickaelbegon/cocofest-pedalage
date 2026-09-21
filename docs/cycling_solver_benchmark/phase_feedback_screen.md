# Phase feedback playback screen, 2026-09-20

The new source-phase policy and bounded PW feedback remain rejected prototypes.
There is no validated five-cycle controller, no RHO fallback, and no GUI change.

## Protocol

`scripts/screen_phase_feedback_correction.py` loads archived unilateral IPOPT
cycle 0 through the provenance adapter, with explicitly declared baseline Ding
parameters, dynamic mechanics, a one-second cycle and the reduced Fourier12
profile. The source has 30 fixed stimulus intervals. Reports retain the source
hashes and declarations. The policy reads the actual plant phase and speed on
each fixed tick; Ding intervals remain exactly 1/30 second.

The gates were fixed before the screen: terminal phase error <=0.5 rad, directed
speed in [0.1,15] rad/s with monotonic phase, native Ding PW bounds and maximum
successive applied PW step 500 us (the existing playback validator setting).
The PW correction is limited to 100 us and each application also enforces slew.
This does not validate the separate 100-us-slew simulation condition.

The source controls are shooting-time indexed, while source phase knots are
nonuniform. `SourcePhasePulseWidthFeedback` preserves those actual knots and
the initial phase offset. Optional proportional phase/speed correction is
distributed with signed crank muscle effectiveness, normalized by its largest
absolute value. It is a local heuristic, not an inverse-dynamics controller
or a new fatigue forecast.

## Results

Reports: `.cache/phase-feedback-screen/h1.json` and `h5.json`.

| One-cycle case | Terminal phase error (rad) | Maximum speed (rad/s) | Gate result |
|---|---:|---:|---|
| Previous uniform-phase playback | 0.97418 | 13.1886 | Rejected: phase |
| Actual source-phase knots | 1.52287 | 11.8750 | Rejected: phase |
| Phase gain20 us/rad, speed gain5 us/(rad/s) | 0.68895 | 11.0979 | Rejected: phase |
| Phase gain50 us/rad, speed gain10 us/(rad/s) | 0.11844 | 17.7201 | Rejected: speed |
| Phase gain100 us/rad, speed gain20 us/(rad/s) | 1.73666 | 16.6464 | Rejected |
| Phase gain200 us/rad, speed gain40 us/(rad/s) | — | — | Reversed rotation |
| Archived temporal replay diagnostic | 0.12097 | 9.8276 | Passed one-cycle gates |

All five-cycle candidates fail. Uncorrected uniform-phase and actual-phase
playbacks reach phase errors30.79 and24.76 rad. Every screened corrected policy,
and temporal replay itself, reverses rotation before completing five cycles.
The one-cycle temporal replay diagnostic therefore is not a selected candidate.

## What the screen establishes

The archive starts at directed phase6.32399 rad (offset0.04080 rad from a full
revolution), with speed5.48146 rad/s, and finishes at speed9.28187 rad/s.
Repeating its PW cycle does not repeat a periodic full physical state.
Correctly translating time-indexed PW into a phase table is necessary for an
auditable policy, but does not preserve the source's timing or force transient.

Doubling Ding substeps16→32 and mechanics substeps8→16 changes source-phase
error1.522871→1.522941 rad; temporal replay changes0.120974→0.120821 rad.
Thus integration step refinement does not explain the phase-playback failure.
The split Ding/mechanics integrator is still approximate and has not been
certified against a fully coupled integrator; this screen does not establish
continuous-plant validity.

The remaining requirement is an online predictor that uses actual force and
fatigue states and predicts their delayed influence on crank speed/phase over
several stimulus intervals. An actual new OCP from the reached state is needed
for any RHO recovery claim. Increasing proportional gains alone was not robust
in the bounded screen.
