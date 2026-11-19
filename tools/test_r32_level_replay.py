import tempfile
from pathlib import Path
from tools import r32_level_replay as replay
def test_closure_loads_and_zip_is_deterministic():
 raw,files,overlay=replay.load();assert raw and files and overlay
 with tempfile.TemporaryDirectory() as td:
  a,b=Path(td)/'a.zip',Path(td)/'b.zip';x=replay.bundle(a);y=replay.bundle(b)
  assert x==y and a.read_bytes()==b.read_bytes()
