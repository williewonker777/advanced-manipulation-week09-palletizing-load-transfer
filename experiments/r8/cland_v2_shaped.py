"""Normal-reference, zero-phase planned torque shaping refinement.

The exact derivatives of the original C2 joint reference generate a release
jerk impulse in elastic motor feedforward. Smooth the *known planned torque*
function offline (6 ms symmetric Gaussian), then use that same function for
both torque and elastic position. This is not an online EMA, a sensor filter,
a parallel residual torque, or an alteration of the kit trajectory.
"""
import numpy as np

from experiments.r8.cland_v2 import (
    Servo as V2Servo, Torque as V2Torque, derivatives, quintic, schedule,
)
from sim import config as C
from sim.nominal import Nominal


class _Shaped:
    def _build_table(self, delay):
        table = super()._build_table(delay)
        # A sufficient public-reference halo prevents the convolution boundary
        # from entering the active [3.75,4.05) command interval.
        times = np.arange(3720, 4081)*self.dt
        empty, full = Nominal(), Nominal()
        full.set_payload(self.box_mass, self.box_com)
        te, tp = [], []
        for t in times:
            index = int(round(t/self.dt))-3747
            reference = ((table['q'][index], table['qd'][index], table['qdd'][index])
                         if 0 <= index < len(table['times']) else self._base_reference(t))
            value = empty.inverse_dynamics(*reference)
            te.append(value)
            tp.append(full.inverse_dynamics(*reference)-value)
        x = np.arange(-18, 19)/6.
        kernel = np.exp(-.5*x*x)
        kernel /= kernel.sum()
        smooth = lambda values: np.column_stack([
            np.convolve(np.asarray(values)[:, j], kernel, mode='same') for j in range(6)
        ])[27:27+len(table['times'])]
        table['empty'], table['payload'] = smooth(te), smooth(tp)
        table['ed'], table['edd'] = derivatives(table['empty'])
        table['pd'], table['pdd'] = derivatives(table['payload'])
        predicted, offload = [], []
        ld, previous_sensor = 0., 1.
        u = self.parameters.u0+self.parameters.u1*self.box_mass
        for k, t in enumerate(table['times']):
            qv, qa = table['qd'][k], table['qdd'][k]
            e, p, ed, edd, pd, pdd = (table[key][k] for key in
                ('empty', 'payload', 'ed', 'edd', 'pd', 'pdd'))
            factor, fd, fdd = schedule(t, self.box_mass, self.parameters)
            sensor = 1. if t < 3.9 else .86**(1+round((t-3.9)/self.dt))
            old = ld
            ld += .04*(factor*(sensor-previous_sensor)/self.dt-ld)
            ldd = (ld-old)/self.dt
            load = factor*sensor
            velocity = np.clip(qv+self.inv_k*(ed+load*pd+(fd*sensor+ld)*p),
                               qv-.35, qv+.35)
            acceleration = np.clip(qa+self.inv_k*(edd+load*pdd
                + 2*(fd*sensor+ld)*pd+(fdd*sensor+ldd)*p), -25., 25.)
            friction = C.VISCOUS*velocity+C.COULOMB*np.tanh(velocity/C.FRICTION_EPS)
            predicted.append(self.nom.B*acceleration+e+load*p+friction)
            h, hd, hdd = quintic(t, self.parameters.end-self.parameters.Tu, self.parameters.Tu)
            offload.append(-u*h*p+self.nom.B*self.inv_k*(-u*h*pdd-2*u*hd*pd-u*hdd*p))
            previous_sensor = sensor
        table['predicted_ff'] = np.asarray(predicted)
        active = (table['times'] >= 3.75) & (table['times'] < 4.05)
        table['scan'] = np.max(np.abs(table['predicted_ff'][active, 1:3]), axis=0)
        ramp = ((table['times'][1:] >= self.parameters.end-self.parameters.Tu-.001)
                & (table['times'][1:] <= self.parameters.end+.001))
        table['scheduled_step_nm'] = float(np.max(np.abs(np.diff(offload, axis=0))[ramp]))
        return table


class Servo(_Shaped, V2Servo):
    pass


class Torque(_Shaped, V2Torque):
    pass
