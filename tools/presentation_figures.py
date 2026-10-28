"""Independently authored presentation figures using the external kit runner.

True table force and runner error are evaluation-only outputs, never production
inputs. Baseline wrench reconstruction uses the documented elastic angle, not
the simulator's true angle. No simulator source is included in this tool.
"""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools.sweep import available_memory_bytes, check_integrity, evaluate_control_metrics
from sim.runner import run_episode
from sim import config as C


def record(controller_name, detector_name, collision=None):
    controller = importlib.import_module(controller_name).Controller()
    base = importlib.import_module(detector_name).Detector
    class Recorder(base):
        def reset(self):
            super().reset()
            self.wrenches = []
            self.tables = []
        def update(self, t, th, thd, tau_m, tau_s, grip):
            residual, alarm = super().update(t, th, thd, tau_m, tau_s, grip)
            if hasattr(self, 'last_wrench'):
                wrench = self.last_wrench.copy()
                table = self.last_table_wrench.copy()
            else:
                q = th.copy()
                elastic = np.asarray(self.nom.elastic, bool)
                q[elastic] -= tau_s[elastic] / self.nom.K[elastic]
                self.nom.d.qpos[:] = q
                mujoco.mj_kinematics(self.nom.m, self.nom.d)
                mujoco.mj_comPos(self.nom.m, self.nom.d)
                body = self.nom.m.body('link_6').id
                rotation = self.nom.d.xmat[body].reshape(3, 3)
                point = self.nom.d.xpos[body] + rotation @ np.array([0., 0., C.BOX_POS[2]+.5*C.BOX_SIZE[2]])
                jacp, jacr = np.zeros((3, 6)), np.zeros((3, 6))
                mujoco.mj_jac(self.nom.m, self.nom.d, jacp, jacr, point, body)
                wrench = np.linalg.lstsq(np.vstack((jacp, jacr)).T, residual, rcond=1e-5)[0]
                table = np.zeros(6)
            self.wrenches.append(wrench)
            self.tables.append(table)
            return residual, alarm
    detector = Recorder()
    result, log = run_episode(controller, box_mass=8., detector=detector, collision=collision)
    arrays = {key: value for key, value in log.items() if isinstance(value, np.ndarray)}
    arrays.update(wrench=np.asarray(detector.wrenches), table_wrench=np.asarray(detector.tables))
    return evaluate_control_metrics(result, log), arrays


def events(axes):
    for ax in np.atleast_1d(axes):
        ax.axvline(1.4, color='gray', ls=':', lw=.8)
        ax.axvline(3.9, color='gray', ls=':', lw=.8)
        ax.grid(alpha=.25)


def build(raw_dir):
    if available_memory_bytes() < 6*1024**3:
        raise RuntimeError('RAM below6GiB')
    check_integrity()
    output = ROOT/'results/figures'
    output.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for mode in ('servo', 'torque'):
        saved = {}
        for label, controller, detector in (('baseline', mode+'_controller', 'collision_detector'),
                                             ('ours', 'my_'+mode, 'my_detector')):
            metrics, arrays = record(controller, detector)
            saved[label] = arrays
            summary.append(dict(mode=mode, variant=label, **metrics))
            np.savez_compressed(raw_dir/f'{mode}_{label}_free.npz', **arrays)
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for label, arrays in saved.items():
            axes[0].plot(arrays['t'], arrays['tau'][:, 1], label=label)
            axes[1].plot(arrays['t'], arrays['e_mm'], label=label)
            axes[2].plot(arrays['t'], arrays['f_table'], label=label)
        axes[0].set_ylabel('J2 motor torque [N m]')
        axes[1].set_ylabel('TCP error [mm]')
        axes[2].set_ylabel('Table normal force [N]')
        axes[2].axhline(1.1*8*9.81, color='red', ls='--', label='1.1 mg')
        axes[2].set_xlabel('Time [s]')
        for ax in axes:
            ax.legend(loc='upper right')
        events(axes)
        fig.suptitle(f'{mode.title()} architecture: same 8 kg, original reference, fixed 6 s cycle')
        fig.tight_layout()
        fig.savefig(output/f'control_{mode}.png', dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(3, 2, figsize=(12, 8))
        for col, (low, high, title) in enumerate(((1.3, 1.65, 'Pick'), (3.7, 4.15, 'Place'))):
            for label, arrays in saved.items():
                mask = (arrays['t'] >= low) & (arrays['t'] <= high)
                for ax, key, units in zip(axes[:, col], ('tau', 'e_mm', 'f_table'),
                                          ('J2 torque [N m]', 'TCP error [mm]', 'Table force [N]')):
                    values = arrays[key][:, 1] if key == 'tau' else arrays[key]
                    ax.plot(arrays['t'][mask], values[mask], label=label)
                    ax.set_ylabel(units)
                    ax.grid(alpha=.25)
            axes[0, col].set_title(title)
            axes[0, col].legend()
            axes[-1, col].set_xlabel('Time [s]')
        fig.suptitle(f'{mode.title()}: no claim of sustained pre-release handover')
        fig.tight_layout()
        fig.savefig(output/f'transfer_{mode}.png', dpi=160)
        plt.close(fig)
    # These are illustrative fixed conditions, not tuning data or official cells.
    for at, label in ((1.44, 'pick'), (2.7, 'transport'), (3.88, 'place')):
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for variant, detector in (('baseline', 'collision_detector'), ('ours', 'my_detector')):
            _, arrays = record('my_servo', detector, (at, (0., 40., 0.)))
            np.savez_compressed(raw_dir/f'detector_{label}_{variant}.npz', **arrays)
            axes[0].plot(arrays['t'], arrays['wrench'][:, 1], label=variant+' estimated Fy')
            axes[1].plot(arrays['t'], arrays['wrench'][:, 2], label=variant+' estimated Fz')
            axes[2].step(arrays['t'], arrays['detect'].astype(int), where='post', label=variant+' alarm')
        axes[0].plot(arrays['t'], arrays['f_ext'], 'k--', label='applied +y force')
        axes[1].plot(arrays['t'], arrays['f_table'], 'k--', label='true table normal (analysis only)')
        axes[0].set_ylabel('Bottom-centre Fy [N]')
        axes[1].set_ylabel('Bottom-centre Fz [N]')
        axes[2].set_ylabel('Alarm')
        axes[-1].set_xlabel('Time [s]')
        for ax in axes:
            ax.set_xlim(at-.18, at+.35)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(alpha=.25)
        fig.suptitle(f'Detector comparison with same new servo controller: {label}, 40 N +y')
        fig.tight_layout()
        fig.savefig(output/f'detection_{label}.png', dpi=160)
        plt.close(fig)
    (ROOT/'results/D5_TRACE_METRICS.json').write_text(json.dumps(summary, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    args = parser.parse_args()
    build(args.raw_dir)


if __name__ == '__main__':
    main()
