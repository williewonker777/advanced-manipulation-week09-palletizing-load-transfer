# R27 C3 history protection gate

This package compares a future explicitly named R27 detector against frozen R26
C3 (`212f4e79…`), never against the weaker R25 baseline. It does not read
`r27_loaded_holdout` and refuses that path in both the entrypoint and closure.

Baseline (run now):

```bash
python -B -m experiments.r27_history.guard baseline \
  --history "$E/r26/acquisition/history" \
  --authentic-official "$E/r26/cohorts/authentic-official.json" \
  --kit "$K" --workers 2 --out-dir "$E/r27/history/baseline"
```

After a future candidate is named and its early gates pass, freeze its explicit
authored dependency closure:

```bash
python -B -m experiments.r27_history.make_closure \
  --candidate experiments.r27_NAME.candidate:Detector \
  --candidate-sha256 "$SHA" \
  --file experiments/r27_NAME/candidate.py \
  --file experiments/r27_NAME/__init__.py \
  --out "$E/r27/history/NAME-closure.json"
```

Add every new transitive authored dependency with another `--file`. Then run:

```bash
python -B -m experiments.r27_history.guard candidate \
  --candidate experiments.r27_NAME.candidate:Detector \
  --candidate-sha256 "$SHA" \
  --closure "$E/r27/history/NAME-closure.json" \
  --baseline-receipt "$E/r27/history/baseline/receipt.json" \
  --history "$E/r26/acquisition/history" \
  --kit "$K" --workers 2 --out-dir "$E/r27/history/NAME-result"
```

The candidate command performs 385 D4 episodes, 29 direct official-grid
episodes, and 29 episodes through unchanged stock `evaluate.py`. It fails on any
C3 strict/false/delay/duration/tool/duplicate/prefix/physics regression or any
of the 66 displayed evaluator cells changing. A successful process is finite
simulation evidence only, not a default promotion or safety qualification.
