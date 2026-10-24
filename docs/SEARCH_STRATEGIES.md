# Search Strategies

## Random baseline

Random search uses an independent NumPy generator seeded by the search configuration. Each proposal samples the declared bounded parameters once. Equivalent space and seed configurations produce the same candidate sequence.

## Optuna TPE

The TPE strategy uses Optuna's public `TPESampler` API with the configured seed and objective direction. The default Optuna startup behavior means small runs include startup trials before density-guided proposals; this is why a 20-trial run is a useful smoke comparison but not a statistical claim.

Both strategies use the same controller, environment, bounds, objective, scenario seed set, and trial budget. The report separately counts proposals, valid trials, actual simulation evaluations, invalid trials, and execution failures. No parallel or distributed optimization is used.

Run a fair local comparison with:

```powershell
python scripts/run_search_benchmark.py --trials 20 --seed 11
```
