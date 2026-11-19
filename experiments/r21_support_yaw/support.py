"""Physical ambiguity interval for table-supported base-yaw torque."""
import math

import numpy as np


def admissible_yaw(packet, th, tau_s, grip):
    """Return the maximum yaw magnitude explainable by projected support.

    For ||F_t|| <= mu Fz, a planar lever arm of length rho can contribute at
    most rho*mu*Fz.  The existing torsional contact model contributes at most
    r_t*Fz.  The sum is a conservative symmetric generalized-yaw interval.
    """
    if not (bool(grip) and packet.last_table_allowed):
        return 0.0
    fz = max(float(packet.last_table_wrench[2]), 0.0)
    if not math.isfinite(fz) or fz <= 0.0:
        return 0.0
    q = np.asarray(th, dtype=float) - packet.inv_k * np.asarray(tau_s, dtype=float)
    _, point, _ = packet._point_jacobian(q)
    rho = math.hypot(float(point[0]), float(point[1])) + math.hypot(packet.half_x, packet.half_y)
    bound = fz * (packet.friction_mu * rho + packet.torsion_radius)
    return bound if math.isfinite(bound) and bound >= 0.0 else 0.0


def unexplained_yaw(value, ambiguity):
    """Remove only the component inside the symmetric support interval."""
    magnitude = max(abs(float(value)) - max(float(ambiguity), 0.0), 0.0)
    return math.copysign(magnitude, float(value))
