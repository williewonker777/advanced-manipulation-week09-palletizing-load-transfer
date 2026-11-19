# R34 placement replay contract

This directory and `tools/r34_placement_replay.py` provide a kit-free source contract.  They do not contain or modify course-kit files and packaging is intentionally disabled.

By default the replay runs the frozen R31 and R33 detectors independently on publicly disclosed placement collisions plus their physically matched no-force references.  `smoke` contains two vertical placement inputs, `placement9` contains the nine disclosed placement misses, and `full-place` contains all 60 D4 placement collisions.

An R34 arm is opt-in only.  `--r34-registry` must name a public manifest with an `experiments.r34_*:Detector` entry and SHA-256 pins for every R34 `.py`/`.json` source.  Private, sealed, validation, result, path-traversal, symlink, hash-mismatched, and frozen-closure-overlapping sources are rejected.  Omitting the manifest never falls back to an unregistered entrant.

Each arm executes in a separate stock-simulator episode on a temporary copy of the authenticated, unmodified kit.  Output records exact signed force schedules, physical hashes, matched-reference prefixes, force-window first rises, strict detections, duplicates, false rises, masks, stuck tails, tool notices, gains/losses, and protection comparisons.  A receipt re-authenticates both the original kit and temporary work copy after execution.
