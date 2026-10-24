# Search Benchmarks

The benchmark command compares random search and Optuna TPE with matched settings:

```powershell
python scripts/run_search_benchmark.py --trials 20 --seed 11 --output artifacts/search-benchmark
```

Measured local run on Windows 11, Python 3.12, the pump stress space, pump rule controller, constraint-violation objective, one scenario seed, and 20 proposals per strategy:

| Strategy | Simulation evaluations | Best objective | Unique discoveries | Wall time |
|---|---:|---:|---:|---:|
| Random | 20 | 0.03444 | 13 | 3.1021 s |
| TPE | 20 | 0.06371 | 16 | 4.1488 s |

Both strategies had 20 valid trials and zero execution failures. TPE performed better in this one run, but this is not evidence of general superiority. Repeated independent runs and uncertainty reporting are required before making a scientific comparison.

The search benchmark also writes each search's trial/discovery artifacts and a machine-readable comparison JSON. Search persistence and replay overhead are included in the per-search wall time but are not treated as simulation throughput.
