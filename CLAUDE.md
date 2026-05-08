## Project

Intraday NQ futures backtesting framework with an ICT/SMC strategy and ML pipeline.
Instrument is hard-coded throughout: `POINT_VALUE = 20.0`, `TICK_SIZE = 0.25`.

## Commands

```bash
python -m pytest tests/ -v          # run tests
python run.py                        # full backtest → tearsheet.html
python run_ml_collect.py             # collect ML dataset → data/ml_dataset.parquet
python run_ml_train.py               # train model → models/ict_smc.pkl
python run_ml_validate.py            # evaluate on validation split
python run_ml_tearsheets.py          # generate tearsheets for all splits
python make_baseline.py --check      # verify trade hashes haven't changed
```

No build step. Dependencies: `pandas numpy pytest plotly hmmlearn pyarrow fastparquet scipy numba`

## Architecture

```
DataLoader → MarketData → run_backtest() → RunResult → PerformanceEngine → TearsheetRenderer
```

- `backtest/data/` — data loading, MarketData container
- `backtest/runner/` — per-bar execution loop, order fills, SL/TP logic
- `backtest/engine/` — position and order state machines
- `backtest/performance/` — metrics, tearsheet, trade log
- `backtest/ml/` — dataset collection, features, walk-forward training, model
- `backtest/regime/` — HMM regime detection
- `backtest/propfirm/` — prop firm Monte Carlo simulation
- `strategies/` — strategy implementations (active: `ict_smc.py`)

### Strategy contract

Subclass `BaseStrategy`, implement:
- `generate_signals(data, i)` → `Optional[Order]` — called when flat
- `on_fill(position, data, i)` → set SL/TP via `position.set_initial_sl_tp()`
- `manage_position(data, i, position)` → `Optional[PositionUpdate]`

### ML data splits

- **train**: 2019–2022 (walk-forward cross-validation)
- **validation**: 2023 (threshold tuning only)
- **test1 / test2**: never touch — final OOS evaluation only

## Code Style

- Python 3.12. Type hints on all function signatures.
- NumPy for hot loops; avoid pandas inside per-bar code.
- Numba (`@_njit` / `@_njit_nogil`) for tight inner loops that are measurably slow.
- `__slots__` on any class instantiated thousands of times per backtest.
- No speculative abstractions — write for the current use case, not hypothetical future ones.
- No docstrings on private helpers unless the logic is genuinely non-obvious.
- Keep module-level state (caches, globals) clearly documented with their key schema.

## Rules

- **Baseline check required** after any change that could affect trade results: `python make_baseline.py --check`
- **Never touch test1/test2 data** — reserved for final OOS evaluation.
- **Propose before changing** — outline the plan and wait for approval before modifying code.
- **Commit after every meaningful change** and push to remote.
- **No Claude references** in commit messages or anywhere in the codebase.
- Results must be bit-for-bit identical before and after any performance optimisation.
- **No Co-authored** section anywhere



## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don’t keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: “Would a staff engineer approve this?”
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask “is there a more elegant way?”
- If a fix feels hacky: “Knowing everything I know now, implement the elegant solution”
- Skip this for simple, obvious fixes — don’t over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don’t ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items  
2. **Verify Plan**: Check in before starting implementation  
3. **Track Progress**: Mark items complete as you go  
4. **Explain Changes**: High-level summary at each step  
5. **Document Results**: Add review section to `tasks/todo.md`  
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections  

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.  
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.  
- **Minimal Impact**: Changes should only touch what’s necessary. Avoid introducing bugs.