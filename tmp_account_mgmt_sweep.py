"""
Account management strategy sweep — correlated reinvestment MC.

Loads top N configs from the AnchoredMeanReversion sweep results and tests all
account management strategy combinations against each config.

Ranked by min P($0).
"""
from __future__ import annotations

import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from itertools import product

import numpy as np
import pandas as pd

from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS
from backtest.propfirm.correlated_mc import (
    AccountManagementStrategy,
    AccountSlot,
    StrategyConfig,
    correlated_reinvestment_mc,
)
from backtest.regime.hmm import RegimeResult

# ── CONFIG ────────────────────────────────────────────────────────────────

BUDGET    = 1_000.0
HORIZON   = 84
GOAL      = 10_000.0
N_SIMS    = 5_000    # per combo
SEED      = 42
N_WORKERS = 2
TOP_N     = 100      # how many AnchoredMeanReversion configs to load

CHECKPOINT_FILE = "account_mgmt_checkpoint.pkl"
CHECKPOINT_EVERY = 5_000             # save every 5k combos
PROGRESS_EVERY   = 1_000             # print progress every N combos

# ── SESSION START TIMES (minutes from midnight) ───────────────────────────
_SESS_ENTRY_MIN = {'930': 9*60+30, '1400': 14*60, '2000': 20*60, '300': 3*60}

# ── LOAD TOP N CONFIGS FROM ANCHORED MEAN REVERSION SWEEP ────────────────────────────────

def _get_arrays(r):
    """Reconstruct pnl_pts/sl_dists for both A and lazy B records."""
    if '_parent' in r:
        idx = r['_indices']
        return r['_parent']['pnl_pts'][idx], r['_parent']['sl_dists'][idx]
    return r['pnl_pts'], r['sl_dists']


def _make_strategy_config(r, name: str) -> StrategyConfig:
    """Build a single-regime StrategyConfig from a AnchoredMeanReversion sweep result."""
    pnl_pts, sl_dists = _get_arrays(r)
    sessions = r['params']['sessions']
    # Use the first session's start time as entry_time_min
    entry_min = _SESS_ENTRY_MIN.get(sessions[0], 570)
    return StrategyConfig(
        name=name,
        pnl_pts_by_regime={0: pnl_pts.astype(np.float32)},
        sl_dists_by_regime={0: sl_dists.astype(np.float32)},
        tpd_by_regime={0: float(r['tpd'])},
        entry_time_min=entry_min,
        eval_risk=float(r['best_erp']),
        fund_risk=float(r['best_frp']),
    )


print("Loading AnchoredMeanReversion sweep results...", flush=True)
with open('sweeps/logs/amr_v2_results.pkl', 'rb') as f:
    jj_results = pickle.load(f)

top_results = jj_results[:TOP_N]
print(f"  Loaded top {len(top_results)} configs (ranked by min P($0))", flush=True)

# Build StrategyConfig objects
configs = [_make_strategy_config(r, f"cfg_{i:03d}") for i, r in enumerate(top_results)]

# ── REGIME RESULT (single absorbing regime — no regime split) ─────────────
# Using 1 state means all trades are drawn from the same distribution.
# Regime-aware splits can be added later once HMM labels are available.
_base_date = date(2019, 1, 1)
_labels = {_base_date + timedelta(days=i): 0 for i in range(2000)}
REGIME_RESULT = RegimeResult(
    labels=_labels,
    label_names={0: 'all'},
    train_end_date=None,
    in_sample_dates=[],
    out_of_sample_dates=[],
    transition_matrix=np.array([[1.0]]),
    state_means=np.zeros(1),
    state_stds=np.ones(1),
    avg_duration_days={'all': 2000.0},
    n_states=1,
)

# ── ACCOUNT ───────────────────────────────────────────────────────────────
# All top configs had best_account='25K'; use that.
ACCOUNT = LUCIDFLEX_ACCOUNTS['25K']

# ── SLOT TEMPLATES ────────────────────────────────────────────────────────
# Each config individually + all pairs of top N (entry_time_min priority).
slot_templates = {cfg.name: [AccountSlot([cfg])] for cfg in configs}

# All pairs of top N — both (A,B) and (B,A) priority orderings? No: priority
# is determined by entry_time_min, so (A,B) and (B,A) yield the same slot.
# Just generate i<j pairs.
n_pairs = 0
for i, ca in enumerate(configs):
    for j in range(i + 1, len(configs)):
        cb = configs[j]
        key = f"{ca.name}+{cb.name}"
        slot_templates[key] = [AccountSlot([ca, cb])]
        n_pairs += 1

print(f"  {len(slot_templates)} slot templates total "
      f"({TOP_N} single + {n_pairs} pairs)", flush=True)

# ── MANAGEMENT STRATEGY GRID ──────────────────────────────────────────────
TRIGGERS        = ["greedy", "on_fail", "on_close", "on_payout", "staggered"]
MAX_CONCURRENTS = [1, 3, 5]
RESERVE_EVALS   = [0, 1]
STAGGER_DAYS    = [7, 14]

combos = [
    (tmpl_name, trigger, max_c, reserve, stagger)
    for tmpl_name in slot_templates
    for trigger in TRIGGERS
    for max_c in MAX_CONCURRENTS
    for reserve in RESERVE_EVALS
    for stagger in (STAGGER_DAYS if trigger == "staggered" else [0])
]
total = len(combos)
print(f"\n{total} total combos  ({N_SIMS} sims each,  {N_WORKERS} workers)", flush=True)


# ── RUN ──────────────────────────────────────────────────────────────────

def _run_one(args):
    tmpl_name, trigger, max_c, reserve, stagger, seed = args
    strat = AccountManagementStrategy(trigger, max_c, reserve, stagger)
    cash  = correlated_reinvestment_mc(
        slot_template=slot_templates[tmpl_name],
        regime_result=REGIME_RESULT,
        account=ACCOUNT,
        eval_fee=ACCOUNT.eval_fee,
        strategy=strat,
        budget=BUDGET,
        horizon=HORIZON,
        n_sims=N_SIMS,
        seed=seed,
    )
    return dict(
        slot_template=tmpl_name,
        trigger=trigger,
        max_concurrent=max_c,
        reserve_n_evals=reserve,
        stagger_days=stagger,
        p_zero=float((cash <= 0).mean()),
        p_goal=float((cash >= GOAL).mean()),
        p_profit=float((cash > BUDGET).mean()),
        median_cash=float(np.median(cash)),
        mean_cash=float(cash.mean()),
    )


def _atomic_save(path: str, data) -> None:
    """Write to a tmp file and atomically rename. Survives mid-write crashes."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)   # atomic on Windows + POSIX


def _combo_key(combo) -> tuple:
    """Serialisable, hashable key uniquely identifying a combo."""
    tmpl_name, trigger, max_c, reserve, stagger = combo
    return (tmpl_name, trigger, int(max_c), int(reserve), int(stagger))


# ── Resume from checkpoint ──────────────────────────────────────────────
rows = []
done_keys: set = set()
if os.path.exists(CHECKPOINT_FILE):
    try:
        with open(CHECKPOINT_FILE, "rb") as f:
            ckpt = pickle.load(f)
        rows      = ckpt.get("rows", [])
        done_keys = {_combo_key((r["slot_template"], r["trigger"],
                                  r["max_concurrent"], r["reserve_n_evals"],
                                  r["stagger_days"])) for r in rows}
        print(f"  Resumed from checkpoint: {len(rows)} combos already done", flush=True)
    except Exception as e:
        print(f"  Checkpoint load failed ({e}), starting fresh", flush=True)
        rows = []
        done_keys = set()

# Filter remaining combos to skip already-done ones
remaining = [(i, c) for i, c in enumerate(combos) if _combo_key(c) not in done_keys]
print(f"  {len(remaining)}/{total} combos remaining", flush=True)

t0    = time.time()
done  = len(rows)            # already-done count
lock  = threading.Lock()     # protects rows + checkpoint writes
last_ckpt = done

with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futs = {ex.submit(_run_one, (*c, SEED + i)): c for i, c in remaining}
    for fut in as_completed(futs):
        try:
            result = fut.result()
        except Exception as exc:
            combo = futs[fut]
            print(f"  ERROR on combo {combo}: {exc}", flush=True)
            continue
        with lock:
            rows.append(result)
            done += 1

            if done - last_ckpt >= CHECKPOINT_EVERY or done == total:
                _atomic_save(CHECKPOINT_FILE, {"rows": rows})
                last_ckpt = done

            if done % PROGRESS_EVERY == 0 or done == total:
                el   = time.time() - t0
                done_this_run = done - len(done_keys)
                rate = done_this_run / el if el > 0 else 0
                eta  = (total - done) / rate if rate > 0 else 0
                print(f"  {done}/{total}  elapsed={el:.0f}s  ETA={eta:.0f}s",
                      flush=True)

df = (pd.DataFrame(rows)
        .sort_values("p_zero")
        .reset_index(drop=True))

print(f"\n-- Top 30 by min P($0) --")
print(df.head(30).to_string(index=False))

df.to_csv("account_mgmt_results.csv", index=False)
print(f"\nSaved to account_mgmt_results.csv  ({len(rows)} rows)")
print(f"Checkpoint: {CHECKPOINT_FILE}")
