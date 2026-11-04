"""Read-only host capacity guard; only the caller ever waits."""
from pathlib import Path
import os
import time

SIMULATION_PATTERNS = ('evaluate.py', 'run.py', 'train.py', 'spawn_main', 'mujoco',
    'isaac', 'robomimic', 'generate_dataset', 'sweep.py', 'r2_study', 'r3_study',
    'r3_extended', 'r3_official', 'r3_replay_timing', 'v2_study', 'v2_collect',
    'render_comparison', 'render_r2', 'render_r3', 'perturb', 'simulation',
    'rollout', 'validate', 'arena', 'mimic')


def is_potential_simulator(command, rss_kib=0):
    return any(pattern in command.lower() for pattern in SIMULATION_PATTERNS) \
        or rss_kib >= 128 * 1024


def snapshot():
    available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith('MemAvailable:'))) / 1024**2
    jobs = []
    for item in Path('/proc').iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        try:
            if 'python' not in (item/'comm').read_text().lower():
                continue
            command = (item/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').lower()
            status = (item/'status').read_text()
            rss = next((int(line.split()[1]) for line in status.splitlines()
                        if line.startswith('VmRSS:')), 0)
            # Large unknown Python jobs count conservatively as potential
            # simulators too.  False positives only make this owner wait.
            if is_potential_simulator(command, rss):
                jobs.append(int(item.name))
        except (OSError, PermissionError):
            pass
    return dict(mem_available_gib=available, load_average=list(os.getloadavg()),
                foreign_heavy_python_pids=jobs, owned_simulation_workers=1,
                gpu_jobs=0, foreign_processes_signalled=0)


def wait_for_capacity():
    while True:
        state = snapshot()
        if state['mem_available_gib'] >= 6 and len(state['foreign_heavy_python_pids']) <= 3:
            return state
        print('R3 waits for capacity:', state, flush=True)
        time.sleep(5)
