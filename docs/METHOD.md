# Load-transfer-aware elastic-joint control and wrench classification

## 1. Problem and information boundary

A binary suction command does not measure the payload fraction carried by the
arm. Support changes continuously at pickup and may reappear before release.
Changing the rigid payload model at the command edge creates feedforward steps
and confounds external-torque collision residuals. Motor-side joint compliance
also creates tracking sag even with an accurate rigid-body nominal model.

Production inputs are only motor encoder angle/derived velocity, six JTS
measurements, grip, desired trajectory, known mass/COM, and public nominal models.
The detector ignores motor torque. True link state, contact forces, and sensor
bias/drift/seed configuration are not read. True table force appears only in
analysis plots and scoring. The simulation/model sources are unchanged.

## 2. Elastic reference and actuator architectures

On elastic axes, the commanded motor reference is

\[
\theta_d=q_d+K^{-1}\tau_{J,d},\qquad
\tau_{J,d}=\tau_0+\hat\lambda(\tau_{box}-\tau_0).
\]

Reference evaluation is advanced by one sample to compensate command transport.
Filtered/clipped derivatives of the torque reference generate elastic velocity
and acceleration feedforward. Motor feedforward is

\[
\tau_{ff}=B\ddot\theta_d+\tau_{J,d}
 +b\dot\theta_d+c\tanh(\dot\theta_d/\epsilon).
\]

**A, servo:** sends position, velocity, velocity feedforward and torque
feedforward to the existing fixed cascaded drive. Drive gains are unchanged.
**B, torque:** adds motor-side PD at 18 rad/s and bounded JTS torque feedback
\(k_\tau(\tau_s-\tau_{J,d})\), with \(k_\tau=-0.10\).
The idealized inertia-shaping interpretation is \(B/(1-k_\tau)\), not a
sampled-data passivity proof. Finite gain sweeps are empirical stability evidence.

## 3. Load and table estimates

A separate slow JTS projection substate computes

\[
\lambda_{direct}=\operatorname{clip}_{[0,1]}
\frac{(\tau_s-\tau_0-\hat b)^T\Delta\tau_{box}}
     {\|\Delta\tau_{box}\|^2+\varepsilon}.
\]

This dynamically noisy estimate is bounded around a 25 ms taught-process ramp,
then low-pass filtered. Near a plausible support height, a second estimate is
\(\lambda_{support}=\operatorname{clip}(1-\hat F_{table,z}/mg,0,1)\).
The selected conservative fusion moves only 10% of the way toward the smaller
support estimate. Zero mass is explicitly guarded. This is not unrestricted
adaptive unloading; the support estimate is delayed and model dependent.

The final placement path retains a smooth 5 ms endpoint advance and scheduled
10% unloading during the last 60 ms before the fixed release. Aggressive early
settle/overtravel/two-stage approaches were rejected for force, saturation or
tracking costs. Sustained pre-release handover is **not** claimed.

## 4. Momentum observer and wrench-consistent collision evidence

The elastic link estimate combines motor velocity with filtered JTS-deflection
rate. Bias is estimated during initial rest and tracked slowly while empty,
freezing on contact suspicion or alarms. The momentum observer uses

\[
p=M(\hat q)\hat v,\quad
\beta=h(\hat q,\hat v)-\dot M\hat v,\quad
r=K_O\left[p-\int(\tau_s-\hat b-\beta+r)dt\right],\quad K_O=50\,s^{-1}.
\]

At the box bottom centre, \(\hat w=J_b^{-T}r\). A table-consistent component is
removed only with grip on, positive box mass, and bottom height within 35 mm of
a taught table height. The feasible component has world upward normal force,
small tangential force, and a COP inside the rotated box footprint. It is a
clipped feasible projection, not an optimal Euclidean projection mixing N and
N m. Remaining horizontal force, negative normal force, COP/torsion violations,
and positive normal force away from the table are independent alarm features.
A grip-off EMA-innovation channel adds transient evidence for tool contact.

Both channels use horizontally symmetric force and roll/pitch thresholds,
learned from no-collision data only. Persistence and amplitudes were selected
before collision testing; disjoint holdout cases were never used for fitting.
The residual returned to the kit remains joint torque in N m; internal wrench
attributes use world-frame N and N m about the box bottom centre.

## 5. Sharing without a runtime helper dependency

Every final file embeds the exact same `SharedEstimator` class. The controller
owns an instance; the detector adopts it through guarded `getattr(self.ctrl, …)`
when the API matches. The detector runs first. A timestamp cache makes the
controller's same-cycle call a read, not a second state advance. Without the
other component, each file creates its own estimator and works independently.

Sharing on/off has identical physical and decision outputs in the parity tests.
The shared effect is lower computation (6000 rather than 12000 estimator advances
per 6 s episode), not an additional physical-performance improvement. The support
fusion effect is reported separately from this reuse effect.

## 6. Experimental interpretation

Official comparisons use the unchanged evaluator behind an integrity gate.
Stratified robustness sweeps and physical/noise perturbations are supplemental,
not replacements for the official score. Timing includes host scheduling and
Python allocation effects; low p99 latency does not establish a hard real-time
bound. The fixed 6 s cycle and suction times remain unchanged.
