# Primary references

1. A. De Luca and R. Mattone, *Sensorless Robot Collision Detection and Hybrid
   Force/Motion Control*, ICRA 2005.
   https://www.diag.uniroma1.it/~labrob/pub/papers/ICRA05_CollDetect.pdf
   DOI: https://doi.org/10.1109/ROBOT.2005.1570247
   Generalized momentum provides an external-torque residual without requiring
   acceleration sensing; modeling and measurement error limit performance.
2. A. Albu-Schäffer, C. Ott and G. Hirzinger, *A Unified Passivity Based Control
   Framework for Position, Torque and Impedance Control of Flexible Joint
   Robots*, IJRR 26(1), 23–39, 2007.
   https://elib.dlr.de/51174/
   DOI: https://doi.org/10.1177/0278364907073776
   Supports the inertia-shaping interpretation of inner torque feedback.
   Its full passivity conditions have NOT been proved for our sampled controller.
3. MuJoCo, *Computation* (official documentation).
   https://mujoco.readthedocs.io/en/stable/computation/
   Contact/constraint calculations are a simulation model; parameter sensitivity
   tests are not physical validation. Local engine version: 3.17.0, Apache-2.0.
