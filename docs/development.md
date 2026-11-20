# Development and research workflows

Install from the repository root in a virtual environment:

```bash
python -m pip install -e ".[dev]"
python run.py
python -m pytest tests/test_core_objects.py tests/test_engine.py tests/test_runner.py tests/test_example.py -q
```

`run.py` runs a deterministic synthetic example; it needs no market data or
saved model. Its output demonstrates execution mechanics, not a trading edge.
Install `.[research,dev]` for the historical research workflows, or select the
`data`, `simulation`, `ml`, and `visualization` extras individually. Dependency
ranges define supported minimums; record resolved versions with experiment results.

## Source layout

| Path | Purpose |
| --- | --- |
| `backtest/` | Reusable execution, research, ML, and account-simulation code |
| `strategies/` | Flat strategy library; historical labels in `STATUS.md` |
| `examples/` | Self-contained synthetic example |
| `scripts/backtest/` | Historical backtest and tearsheet driver |
| `scripts/data/` | Market-data cache and news-calendar builders |
| `scripts/ml/` | Dataset collection, training, threshold selection, and evaluation |
| `scripts/sweeps/` | Parameter sweeps and account-management experiments |
| `scripts/validation/` | Baseline and strategy verification drivers |
| `scripts/propfirm/` | Account experiments, regime updates, and shared research helpers |
| `scripts/portfolio/` | Multi-instrument portfolio research |
| `baseline/` | Committed regression records, retained intentionally |
| `configs/` | Selected reproducible configuration inputs |
| `docs/research/` | Historical strategy specifications |

Run historical scripts from the repository root, after reviewing their date
boundaries and inputs. They retain experiment-specific settings and can perform
long runs. Examples of relocated commands:

```bash
python scripts/backtest/run.py
python scripts/data/build_md_cache.py
python scripts/ml/run_ml_collect.py
python scripts/ml/run_ml_train.py
python scripts/validation/make_baseline.py --check
python scripts/sweeps/amr_v2_sweep.py
python scripts/sweeps/run_both_sweeps.py
```

The original `tmp_reinvest.py` helpers now live in
`backtest.propfirm.reinvestment`. Old strategy namespaces such as
`strategies.profitable.amr_strategy` remain importable through lazy aliases for
existing notebooks and serialized classes. New code should use
`strategies.amr_strategy`. Strategy discovery uses flat modules; local files
left under old directories must be moved deliberately before joining a portfolio.

## Local artifacts

Caches, trained models, raw results, logs, checkpoints, and archived experiments
are no longer tracked. Existing local copies are preserved. New clones need
their own market data and must regenerate required artifacts. Old committed
versions remain available in Git history; this cleanup does not rewrite history.

| Artifact | Producer or prerequisite |
| --- | --- |
| `data/NQ_1m.parquet`, `data/NQ_5m.parquet`, `data/NQ_bar_map.npy` | User-supplied market data prepared with `backtest.data.loader`; review each driver's input schema |
| `md_cache.pkl` | `python scripts/data/build_md_cache.py` |
| `data/ml_dataset.parquet` | `python scripts/ml/run_ml_collect.py` |
| `models/ict_smc*.pkl` | `python scripts/ml/run_ml_train.py` (review direction and ensemble settings) |
| `models/threshold_opt.json` | `python scripts/ml/run_ml_threshold_opt.py` after training |
| `cache/amr_hmm.pkl` | `python scripts/propfirm/regime_update.py` using `data/NQ_1day_full_data.parquet` |
| `cache/amr_realistic_pools.pkl` | `python scripts/propfirm/rebuild_realistic_pools.py`, using required data and HMM cache |
| Sweep checkpoints and raw result tables | The corresponding script in `scripts/sweeps/` or `scripts/propfirm/` |

Regenerating an HMM cache with newer data or dependencies can change regime
labels; it does not reproduce an old experiment automatically. The phase-2/3
rerun script specifically needs a historical pre-fix result pickle, which a
fresh clone does not contain. Start a new sweep or restore that local artifact
when intentionally reproducing that experiment.

Keep published findings selective: include the configuration, data period,
costs, selection procedure, and environment with any result added to the repo.
CI exercises synthetic core behavior; market-data baselines and research
evaluations require the corresponding private inputs and optional dependencies.
