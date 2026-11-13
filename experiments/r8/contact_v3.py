"""Mass-banded A3 landing prototype with one shared link-load torque.

Zero activation delegates to the exact base (servo no-advance, torque R1).
Full activation retains the A2 shaped equations. Intermediate activation blends
the planned torque functions, the load law and derivative rule, not a separate
post-release torque correction. No true plant/contact values are inputs.
"""
from dataclasses import replace

import numpy as np

import r1_servo
import r1_torque
from experiments.r8 import cland_v2 as C, cland_v2_shaped as G
from sim import config as K
from sim.nominal import Nominal


def mass_weight(mass, lo, hi):
    if not 0 <= lo < hi <= 8:
        raise ValueError('invalid mass activation band')
    return C.quintic(mass, lo, hi-lo)[0]


class _MassBand:
    mass_lo = 7.273
    mass_hi = 7.873

    def reset(self):
        self.landing_weight = mass_weight(self.box_mass, self.mass_lo, self.mass_hi)
        self._blend_filtered = None
        configured = getattr(self, '_configured_parameters', self.parameters)
        self._configured_parameters = configured
        base_delay = 0. if self.MODE == 'servo' and self.box_mass > 0 else -.005
        if self.landing_weight == 0:
            self.ENDPOINT_DELAY = base_delay
            (r1_servo if self.MODE == 'servo' else r1_torque).Controller.reset(self)
            self._a2_active = False
            self._previous_sensor = 0.
            return
        self.parameters = replace(configured,
            delay=base_delay+self.landing_weight*(configured.delay-base_delay),
            damping=self.landing_weight*configured.damping)
        super().reset()

    def _build_table(self, delay):
        table = super()._build_table(delay)
        w = self.landing_weight
        if w == 1:
            return table
        empty, full = Nominal(), Nominal()
        full.set_payload(self.box_mass, self.box_com)
        args = list(zip(table['q'], table['qd'], table['qdd']))
        raw_e = np.array([empty.inverse_dynamics(*a) for a in args])
        raw_p = np.array([full.inverse_dynamics(*a) for a in args])-raw_e
        table['empty'] = (1-w)*raw_e+w*table['empty']
        table['payload'] = (1-w)*raw_p+w*table['payload']
        table['ed'], table['edd'] = C.derivatives(table['empty'])
        table['pd'], table['pdd'] = C.derivatives(table['payload'])
        # Recompute the expected full FF using the same intermediate load and
        # derivative blend. The synthetic load decay is the offline planning
        # assumption, never an extra runtime input or a stability guarantee.
        previous_sensor, sd = 1., 0.
        previous_link = table['empty'][0]+table['payload'][0]
        filtered = table['ed'][0]+table['pd'][0]
        predictions = []
        for k, t in enumerate(table['times']):
            sensor = 1. if t < 3.9 else .86**(1+round((t-3.9)/self.dt))
            old_sd = sd
            sd += .04*((sensor-previous_sensor)/self.dt-sd)
            sdd = (sd-old_sd)/self.dt
            base, bd, bdd = sensor, sd, sdd
            if 3.84 < t < 3.9:
                h, hd, hdd = C.quintic(t, 3.84, .06)
                if 1-.1*h < sensor:
                    base, bd, bdd = 1-.1*h, -.1*hd, -.1*hdd
            factor, fd, fdd = C.schedule(t, self.box_mass, self.parameters)
            load = (1-w)*base+w*factor*sensor
            ld = (1-w)*bd+w*(fd*sensor+factor*sd)
            ldd = (1-w)*bdd+w*(fdd*sensor+2*fd*sd+factor*sdd)
            e, p, ed, edd, pd, pdd = (table[key][k] for key in
                ('empty', 'payload', 'ed', 'edd', 'pd', 'pdd'))
            link = e+load*p
            old_filtered = filtered.copy()
            filtered += .04*((link-previous_link)/self.dt-filtered)
            first = (1-w)*filtered+w*(ed+load*pd+ld*p)
            second = (1-w)*(filtered-old_filtered)/self.dt\
                +w*(edd+load*pdd+2*ld*pd+ldd*p)
            qv, qa = table['qd'][k], table['qdd'][k]
            v = np.clip(qv+self.inv_k*first, qv-.35, qv+.35)
            a = np.clip(qa+self.inv_k*second, -25., 25.)
            ff = self.nom.B*a+link+K.VISCOUS*v+K.COULOMB*np.tanh(v/K.FRICTION_EPS)
            predictions.append(ff)
            previous_sensor, previous_link = sensor, link
        table['predicted_ff'] = np.array(predictions)
        mask = (table['times'] >= 3.75) & (table['times'] < 4.05)
        table['scan'] = np.max(abs(table['predicted_ff'][mask, 1:3]), axis=0)
        return table

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        w = self.landing_weight
        if w in (0., 1.) or t < 3.75 or t >= 4.05:
            return super().compute(t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)
        k = int(round(t/self.dt))-3747
        tab = self.table
        q, qv, qa, e, p, ed, edd, pd, pdd = (tab[key][k] for key in
            ('q', 'qd', 'qdd', 'empty', 'payload', 'ed', 'edd', 'pd', 'pdd'))
        sensor = self._estimate_load(t, th, thd, tau_s, grip)
        old_sd = self._sensor_derivative
        self._sensor_derivative += .04*((sensor-self._previous_sensor)/self.dt-old_sd)
        sd, sdd = self._sensor_derivative, (self._sensor_derivative-old_sd)/self.dt
        base, bd, bdd = sensor, sd, sdd
        if 3.84 < t < 3.9:
            h, hd, hdd = C.quintic(t, 3.84, .06)
            if 1-.1*h < sensor:
                base, bd, bdd = 1-.1*h, -.1*hd, -.1*hdd
        factor, fd, fdd = C.schedule(t, self.box_mass, self.parameters)
        load = (1-w)*base+w*factor*sensor
        ld = (1-w)*bd+w*(fd*sensor+factor*sd)
        ldd = (1-w)*bdd+w*(fdd*sensor+2*fd*sd+factor*sdd)
        link = e+load*p
        if self._blend_filtered is None:
            self._blend_filtered = self.tau_dot.copy()
        old_filtered = self._blend_filtered.copy()
        self._blend_filtered += .04*((link-self.tau_link_prev)/self.dt-old_filtered)
        exact = ed+load*pd+ld*p
        second = edd+load*pdd+2*ld*pd+ldd*p
        self.tau_dot = (1-w)*self._blend_filtered+w*exact
        d2 = (1-w)*(self._blend_filtered-old_filtered)/self.dt+w*second
        velocity = np.clip(qv+self.inv_k*self.tau_dot, qv-.35, qv+.35)
        acceleration = np.clip(qa+self.inv_k*d2, -25., 25.)
        theta = q+self.inv_k*link
        self._previous_sensor = sensor
        self.tau_link_prev = link.copy()
        self.last_commanded_load = load
        friction = K.VISCOUS*velocity+K.COULOMB*np.tanh(velocity/K.FRICTION_EPS)
        ff = self.nom.B*acceleration+link+friction
        if self.MODE == 'servo':
            if self.parameters.damping:
                deflection = -self.inv_k*tau_s
                if self._deflection_previous is not None:
                    self._deflection_velocity += (1-np.exp(-30*self.dt))*(
                        (deflection-self._deflection_previous)/self.dt-self._deflection_velocity)
                self._deflection_previous = deflection.copy()
                window = C.quintic(t, 3.75, .03)[0]*(1-C.quintic(t, 3.9, .15)[0])
                velocity -= window*self.parameters.damping*self._deflection_velocity
            return theta, velocity, velocity, ff
        self.v_motor += .2*(thd-self.v_motor)
        return ff+self.nom.B*(self.wn**2*(theta-th)+2*self.wn*(velocity-self.v_motor))\
            +self.k_tau*(tau_s-link)


class Servo(_MassBand, G.Servo):
    pass


class Torque(_MassBand, G.Torque):
    pass
