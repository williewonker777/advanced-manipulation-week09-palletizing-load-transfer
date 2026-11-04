"""Conservative per-episode host guard; never signals another process."""
from pathlib import Path
import os
import time


def snapshot():
    memory = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                      if x.startswith('MemAvailable:'))) / 1024**2
    foreign = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            command = (p/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            executable = (p/'comm').read_text().strip()
            if 'python' not in executable.lower():
                continue
            if any(x in command for x in ('spawn_main', 'evaluate.py', 'run.py', 'train.py',
                                          'scripts/robomimic', 'generate_dataset', 'v2_study.py',
                                          'v2_collect.py', 'render_comparison.py')):
                foreign.append(int(p.name))
        except (OSError, PermissionError):
            continue
    return dict(mem_available_gib=memory, load_average=list(os.getloadavg()),
                foreign_heavy_python_pids=foreign, owned_simulation_workers=1)


def wait_for_capacity():
    while True:
        state = snapshot()
        if state['mem_available_gib'] >= 6 and len(state['foreign_heavy_python_pids']) <= 3:
            return state
        # No killing/pausing another session; this owner waits before starting.
        print('Waiting for host capacity:', state, flush=True)
        time.sleep(5)
