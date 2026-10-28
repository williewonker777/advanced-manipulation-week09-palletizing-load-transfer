# D1 environment (2026-10-28)

- Ubuntu 24.04.4 LTS, x86_64; AMD Ryzen 7 7800X3D; ~30 GiB RAM.
- Python 3.12.3, isolated `.venv/` (no system/Conda/other environment installs).
- Created with `env -u PYTHONPATH python3.12 -m venv --without-pip .venv`.
- Installed with `env -u PYTHONPATH PIP_NO_CACHE_DIR=1 pip3 --python .venv/bin/python install -r requirements.txt`.
- System pip 24.0 drives installation; ensurepip and pip inside the venv are absent.
- Run all Python with `env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python`.
- Bytecode suppression protects the immutable `sim/` tree as well as clearing ROS/PRISM import paths.
- No GUI, GPU, or Isaac Sim interaction. Wall-clock timings are host-load dependent.

## Installed versions

```text
absl-py==2.5.1
contourpy==1.4.0
cycler==0.12.1
etils==1.14.0
fonttools==4.68.0
fsspec==2026.9.0
glfw==2.11.0
kiwisolver==1.5.1
matplotlib==3.11.2
mujoco==3.17.0
numpy==2.5.4
packaging==26.3
pillow==13.0.0
PyOpenGL==3.1.10
pyparsing==3.3.3
python-dateutil==3.0.0
six==1.17.0
typing_extensions==4.16.0
tzdata==2026.5
websockets==17.3
zipp==4.1.2
```

## Dependency validation

No broken requirements found.
