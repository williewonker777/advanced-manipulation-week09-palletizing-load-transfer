"""Causal incremental normal equations for the frozen early and height decisions."""
import numpy as np

from experiments.r14_online.stream import Stream as FullStream
from experiments.r15_early_com.stream import PREFIX_CENTRES, PREFIX_MASK, Stream as EarlyStream
from tools.r13_grasp_estimator import CENTRES, DT


class Stream(EarlyStream):
    """Frozen public stream plus bounded incremental decision algebra."""

    def configure_staging(self, height_calibration, early_calibration):
        if height_calibration.get("schema") != "r14-registration-calibration.v1":
            raise ValueError("height calibration schema")
        if early_calibration.get("schema") != "r15-early-com-calibration.v1":
            raise ValueError("early calibration schema")
        self._height_calibration = height_calibration
        self._early_calibration = early_calibration
        self._prefix_slots = {int(slot): index for index, slot in enumerate(np.flatnonzero(PREFIX_MASK))}
        self._early_grams = np.zeros((5, 10, 10)); self._early_rhs = np.zeros((5, 10))
        self._early_rows = 0; self._early_result = None
        self._height_stage = 0; self._height_work = {}; self._height_result = None
        self.staging_ticks = []

    def _accumulate_early(self, slot):
        position = self._prefix_slots.get(int(slot))
        if position is None:
            return
        model = self._early_calibration["dynamics"]
        sigma = np.asarray(model["sigma"], float)
        bias = np.asarray(model["bias"])
        y = (self._y[slot]-bias[0, position]
             -(self.mass-float(model["center"]))*bias[1, position])/sigma
        eye = np.eye(6)
        design = np.concatenate((self._jac[slot], self._stiffness[slot, :, None], eye), axis=1)
        z = design/sigma[:, None]
        fold = position*5//len(PREFIX_CENTRES)
        self._early_grams[fold] += z.T@z
        self._early_rhs[fold] += z.T@y
        self._early_rows += 1

    def _process_centre(self, slot, centre):
        FullStream._process_centre(self, slot, centre)
        if hasattr(self, "_prefix_slots"):
            self._accumulate_early(slot)

    @staticmethod
    def _early_expected(mass, dynamics, baseline):
        return np.asarray(baseline)[0]+(float(mass)-float(dynamics["center"]))*np.asarray(baseline)[1]

    def finish_early_decision(self):
        if self._early_result is not None:
            return dict(self._early_result)
        features = self.finish_prefix()
        if self._early_rows != len(PREFIX_CENTRES):
            raise ValueError("incremental early rows incomplete")
        model = self._early_calibration; dynamics = model["dynamics"]
        sigma = np.asarray(dynamics["sigma"], float)
        scales = np.r_[np.full(3, 15.), .065, np.maximum(sigma*3, .06)]
        ridge = np.diag(1/scales**2)
        total_g = self._early_grams.sum(axis=0); total_b = self._early_rhs.sum(axis=0)
        solve = lambda g, b: np.linalg.solve(g+ridge, b)
        full = np.asarray(features["prior_com_mm"])+solve(total_g, total_b)[:3]
        deleted = np.array([np.asarray(features["prior_com_mm"])
                            +solve(total_g-g, total_b-b)[:3]
                            for g, b in zip(self._early_grams, self._early_rhs)])
        normal = self._early_expected(self.mass, dynamics, model["baseline_xy"])
        xy = full[:2]-normal; length = float(np.linalg.norm(xy))
        radius = (.75+float(model["normal_guard_mm"])
                  +float(np.max(np.linalg.norm(deleted[:, :2]-full[:2], axis=1))))
        domain = (dynamics["range"][0] <= self.mass <= dynamics["range"][1]
                  and np.linalg.norm(full-np.asarray(features["prior_com_mm"])) <= 20)
        accept = bool(domain and radius <= 2.5 and length > radius)
        action = xy*min(1, 15/max(length, 1e-12)) if accept else np.zeros(2)
        self._early_result = {"relative_xy_mm": xy.tolist(), "absolute_com_offset_mm": full.tolist(),
            "expected_normal_xy_mm": normal.tolist(),
            "deleted_relative_xy_mm": (deleted[:, :2]-normal).tolist(),
            "halfwidth_radius_mm": radius, "accept": accept,
            "model_action_xy_mm": action.tolist(),
            "completed_at_s": float(features["completed_at_s"])}
        return dict(self._early_result)

    def advance_height_stage(self, t):
        if self._height_result is not None or self._next_centre != len(CENTRES):
            return
        calibration = self._height_calibration
        stage = self._height_stage; work = self._height_work
        if stage == 0:
            sigma = np.asarray(calibration["sigma"]); mass = self.mass
            y = (self._y-np.asarray(calibration["bias"])[0]
                 -(mass-float(calibration["center"]))*np.asarray(calibration["bias"])[1])/sigma
            clock = DT*CENTRES-2.55; eye6 = np.broadcast_to(np.eye(6), (len(CENTRES), 6, 6))
            scales = np.r_[np.full(3, 15.), .065, np.maximum(sigma*3, .06),
                           np.maximum(sigma*.8, .02)]
            work.update(y=y, sigma=sigma, clock=clock, eye6=eye6, scales=scales,
                        vertical=self._vertical.copy(), fold=(CENTRES//80)%5,
                        grams=[], rhs=[])
        elif 1 <= stage <= 5:
            fold = stage-1; mask = work["fold"] == fold
            design = np.concatenate((self._jac[mask], self._stiffness[mask, :, None],
                                     work["eye6"][mask],
                                     work["eye6"][mask]*work["clock"][mask, None, None]), axis=-1)
            block = (design/work["sigma"][None, :, None]
                     *work["scales"]).reshape(-1, len(work["scales"]))
            target = work["y"][mask].ravel()
            work["grams"].append(block.T@block); work["rhs"].append(block.T@target)
        elif stage == 6:
            work["grams"] = np.stack(work["grams"]); work["rhs"] = np.stack(work["rhs"])
            work["gt"] = work["grams"].sum(axis=0); work["bt"] = work["rhs"].sum(axis=0)
            work["eye"] = np.eye(len(work["scales"])); work["deleted"] = []
            work["coef"] = np.linalg.solve(work["gt"]+work["eye"], work["bt"])*work["scales"]
        elif 7 <= stage <= 11:
            fold = stage-7
            coef = np.linalg.solve(work["gt"]-work["grams"][fold]+work["eye"],
                                   work["bt"]-work["rhs"][fold])*work["scales"]
            work["deleted"].append(float(work["vertical"]@coef[:3]))
        elif stage == 12:
            scales = work["scales"]; unscaled = work["gt"]/(scales[:, None]*scales[None, :])
            vertical = work["vertical"]
            orth = np.linalg.svd(vertical[None, :], full_matrices=True)[2][1:].T
            ns = np.eye(len(scales))[:, 1:].copy(); ns[:3, :2] = orth
            unit = np.r_[vertical, np.zeros(len(scales)-3)]
            work.update(unscaled=unscaled, ns=ns, unit=unit)
        elif stage == 13:
            unscaled, ns, unit = work["unscaled"], work["ns"], work["unit"]
            cov = ns.T@unscaled@ns; cross = ns.T@unscaled@unit
            remaining = unit@unscaled@unit-cross@np.linalg.lstsq(cov, cross, rcond=1e-14)[0]
            work["info"] = float(np.sqrt(max(0., remaining)/(len(CENTRES)*6)))
        elif stage == 14:
            coef = work["coef"]; residual = float(work["vertical"]@coef[:3])
            prior = float(self._prior_bottom_mm)
            a, b = calibration["normal_bottom_intercept_slope_mm"]
            centre = float(a+b*(self.mass-float(calibration["center"])))
            point = prior+residual-centre
            width = .75+float(calibration["normal_guard_mm"])+max(abs(x-residual) for x in work["deleted"])
            domain = (calibration["range"][0] <= self.mass <= calibration["range"][1]
                      and np.linalg.norm(coef[:3]) <= 15. and work["info"] > 1e-4)
            accepted = bool(domain and width <= 2.5 and abs(point) > width)
            self._height_result = dict(extra_height_mm=point, absolute_bottom_mm=prior+residual,
                expected_normal_bottom_mm=centre, prior_bottom_mm=prior,
                com_residual_height_mm=residual, half_width_mm=width,
                interval_mm=[point-width, point+width], accept=accepted,
                applied_mm=point if accepted else 0.,
                soft_interval_action_mm=float(np.sign(point)*max(0., abs(point)-width)) if domain else 0.,
                independent_height_information_per_mm=work["info"],
                fitted_displacement_within_domain=bool(domain),
                block_delete_mm=[prior+x-centre for x in work["deleted"]])
        self.staging_ticks.append({"t": float(t), "stage": int(stage)})
        self._height_stage += 1

    def finish_height_decision(self, features):
        if self._height_result is None or self._height_stage < 15:
            raise ValueError("staged height decision incomplete")
        checks = ((features["y"], self._y), (features["jac"], self._jac),
                  (features["stiffness"], self._stiffness))
        if any(not np.array_equal(left, right) for left, right in checks):
            raise ValueError("stale staged height features")
        return dict(self._height_result)
