"""
Rerun AnchoredMeanReversion Phase 2 + Phase 3 ONLY, using the top 500 records from the
pre-MLL-fix backup. Sweep A and B are skipped (Sortino ranking doesn't depend
on MLL, so the same top 500 would emerge).

Saves updated results to sweeps/logs/amr_v2_results.{pkl,txt}.
"""
from __future__ import annotations

import io
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

# ── Match settings from tmp_amr_v2_sweep.py ────────────────────────
SRC_PKL  = "sweeps/logs/amr_v2_results_pre_mll_fix.pkl"
DST_PKL  = "sweeps/logs/amr_v2_results.pkl"
DST_TXT  = "sweeps/logs/amr_v2_results.txt"

N_WORKERS = 2
ERP_GRID  = [round(x * 0.1, 1) for x in range(1, 11)]
FRP_GRID  = [round(x * 0.1, 1) for x in range(1, 11)]
P2_NPOOL  = 2_000
P3_NPOOL  = 5_000
P3_NMC    = 2_000
BUDGET    = 1_000.0
HORIZON   = 84
GOAL      = 10_000.0
MIN_TRADES = 20
MIN_TPD    = 0.5

# ── Load market data for Sortino recompute ─────────────────────────────
with open("md_cache.pkl", "rb") as f:
    _cache = pickle.load(f)
md     = _cache["md"]
n_days = _cache["n_days"]
print(f"Loaded {n_days} days of market data", flush=True)

_bar_day_ord = np.array([d.toordinal() for d in md.df_1m.index.date], dtype=np.int32)

from backtest.propfirm.lucidflex import (
    LUCIDFLEX_ACCOUNTS, MICRO_COMM_RT, MINI_COMM_RT,
    simulate_eval_batch, simulate_funded_batch,
)
from tmp_reinvest import simulate_lifecycles, reinvestment_mc

print("Imports done", flush=True)


# ── Atomic save ────────────────────────────────────────────────────────
def _atomic_dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        try: os.fsync(f.fileno())
        except: pass
    os.replace(tmp, path)


# ── Rehydrate _parent/_indices records into flat pnl/sl/exit arrays ────
def _get_arrays(r):
    if "_parent" in r:
        idx = r["_indices"]
        p   = r["_parent"]
        return p["pnl_pts"][idx], p["sl_dists"][idx], p["exit_bars"][idx]
    return r["pnl_pts"], r["sl_dists"], r["exit_bars"]


# ── Load top 500 from backup ───────────────────────────────────────────
with open(SRC_PKL, "rb") as f:
    src = pickle.load(f)
print(f"Loaded {len(src)} pre-fix records", flush=True)


def _rehydrate(r):
    pnl, sl, eb = _get_arrays(r)
    return dict(
        params    = r["params"],
        pnl_pts   = pnl,
        sl_dists  = sl,
        exit_bars = eb,
        n_t       = int(r["n_t"]),
        tpd       = float(r["tpd"]),
        wr        = float(r["wr"]),
        avgR      = float(r["avgR"]),
        sortino   = float(r["sortino"]),
    )


records = [_rehydrate(r) for r in src]
print(f"Rehydrated {len(records)} records", flush=True)


# ── Numba warm-up so Phase 2 kernels are ready ─────────────────────────
print("Warming up Numba (lucidflex kernels)...", flush=True)
_t0 = time.perf_counter()
_dummy_pnl = np.zeros(10, dtype=np.float32)
_dummy_sl  = np.ones(10,  dtype=np.float32) * 5.0
_dummy_rng = np.random.default_rng(0)
_dummy_acct = LUCIDFLEX_ACCOUNTS["25K"]
simulate_eval_batch(_dummy_pnl, _dummy_sl, _dummy_acct, 0.2,
                    "fixed_dollar", "nq_first", _dummy_rng, 1.0, 50,
                    max_minis=_dummy_acct.max_micros // 10,
                    mini_comm_rt=MINI_COMM_RT)
_sb = np.full(10, float(_dummy_acct.starting_balance), dtype=np.float32)
simulate_funded_batch(_dummy_pnl, _dummy_sl, _dummy_acct, 0.2,
                      "fixed_dollar", "nq_first", _dummy_rng, 1.0, _sb,
                      max_minis=_dummy_acct.max_micros // 10,
                      mini_comm_rt=MINI_COMM_RT)
print(f"  Numba ready in {time.perf_counter()-_t0:.1f}s", flush=True)


# ══════════════════════════════════════════════════════════════════════
# PHASE 2 — ERP × FRP × account optimization (corrected MLL)
# ══════════════════════════════════════════════════════════════════════
def _phase2(rec):
    pnl_pts  = rec["pnl_pts"]
    sl_dists = rec["sl_dists"]
    tpd      = rec["tpd"]

    best_ev_med  = -np.inf
    best_ev      = 0.0
    best_erp     = ERP_GRID[0]
    best_frp     = FRP_GRID[0]
    best_pass    = 0.0
    best_payout  = 0.0
    best_pay_med = 0.0
    best_acct    = "25K"

    for acct in LUCIDFLEX_ACCOUNTS.values():
        max_minis = int(acct.max_micros) // 10
        sb_f = np.full(P2_NPOOL, float(acct.starting_balance), dtype=np.float32)
        rng  = np.random.default_rng(42)

        eval_cache  = {}
        for erp in ERP_GRID:
            try:
                passed, _, eval_days = simulate_eval_batch(
                    pnl_pts, sl_dists, acct,
                    risk_pct=erp, scheme="fixed_dollar", sizing_mode="nq_first",
                    rng=rng, trades_per_day=tpd, n_sims=P2_NPOOL,
                    max_minis=max_minis, mini_comm_rt=MINI_COMM_RT,
                )
                eval_cache[erp] = (float(passed.mean()), float(eval_days.mean()))
            except Exception:
                eval_cache[erp] = (0.0, 1.0)

        funded_cache = {}
        for frp in FRP_GRID:
            try:
                total_w, _, _, _, _, funded_days = simulate_funded_batch(
                    pnl_pts, sl_dists, acct,
                    risk_pct=frp, scheme="fixed_dollar", sizing_mode="nq_first",
                    rng=rng, trades_per_day=tpd, starting_balances=sb_f,
                    max_minis=max_minis, mini_comm_rt=MINI_COMM_RT,
                )
                funded_cache[frp] = (float(total_w.mean()),
                                     float(np.median(total_w)),
                                     float(funded_days.mean()))
            except Exception:
                funded_cache[frp] = (0.0, 0.0, 1.0)

        # ERP: maximise pass rate
        best_erp_acct = max(ERP_GRID, key=lambda e: eval_cache[e][0])
        pass_rate, mean_eval_days = eval_cache[best_erp_acct]

        # FRP: maximise median EV/day given best ERP
        best_frp_acct = max(
            FRP_GRID,
            key=lambda f: (
                (pass_rate * funded_cache[f][1] - acct.eval_fee)
                / max(mean_eval_days + pass_rate * funded_cache[f][2], 1.0)
            ),
        )
        mean_payout, median_payout, _ = funded_cache[best_frp_acct]
        total_days = mean_eval_days + pass_rate * funded_cache[best_frp_acct][2]
        denom      = max(total_days, 1.0)
        ev_day     = (pass_rate * mean_payout   - acct.eval_fee) / denom
        ev_day_med = (pass_rate * median_payout - acct.eval_fee) / denom

        if ev_day_med > best_ev_med:
            best_ev_med  = ev_day_med
            best_ev      = ev_day
            best_erp     = best_erp_acct
            best_frp     = best_frp_acct
            best_pass    = pass_rate
            best_payout  = mean_payout
            best_pay_med = median_payout
            best_acct    = acct.name

    return dict(**rec,
                best_ev=best_ev, best_ev_med=best_ev_med,
                best_erp=best_erp, best_frp=best_frp,
                best_pass=best_pass, best_payout=best_payout,
                best_pay_med=best_pay_med, best_account=best_acct)


print(f"\n-- PHASE 2: ERP × FRP × account on {len(records)} configs --", flush=True)
t_p2 = time.perf_counter()
p2_results = []
done2 = 0
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futs = {ex.submit(_phase2, r): r for r in records}
    for fut in as_completed(futs):
        done2 += 1
        p2_results.append(fut.result())
        if done2 % 25 == 0 or done2 == len(records):
            el = time.perf_counter() - t_p2
            print(f"  {done2}/{len(records)}  elapsed={el:.0f}s", flush=True)
print(f"Phase 2 done in {time.perf_counter()-t_p2:.0f}s", flush=True)


# ══════════════════════════════════════════════════════════════════════
# PHASE 3 — Reinvestment MC (corrected MLL)
# ══════════════════════════════════════════════════════════════════════
print(f"\n-- PHASE 3: reinvestment MC (budget=${BUDGET:.0f}, horizon={HORIZON}d) --",
      flush=True)
t_p3 = time.perf_counter()


def _phase3(rec):
    acct = LUCIDFLEX_ACCOUNTS[rec.get("best_account", "25K")]
    pool = simulate_lifecycles(
        rec["pnl_pts"], rec["sl_dists"], rec["tpd"], acct,
        eval_risk=rec["best_erp"], fund_risk=rec["best_frp"],
        N=P3_NPOOL, seed=42,
    )
    cash = reinvestment_mc(
        pool, eval_fee=acct.eval_fee,
        budget=BUDGET, horizon=HORIZON, n_sims=P3_NMC, seed=99,
    )
    return dict(**rec,
                p_zero=float((cash <= 0).mean()),
                p_goal=float((cash >= GOAL).mean()),
                p_profit=float((cash > BUDGET).mean()),
                median_cash=float(np.median(cash)),
                mean_cash=float(cash.mean()))


done3 = 0
p3_results = []
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futs = {ex.submit(_phase3, r): r for r in p2_results}
    for fut in as_completed(futs):
        done3 += 1
        p3_results.append(fut.result())
        if done3 % 50 == 0 or done3 == len(p2_results):
            el = time.perf_counter() - t_p3
            print(f"  {done3}/{len(p2_results)}  elapsed={el:.0f}s", flush=True)
p3_results.sort(key=lambda r: r["p_zero"])
print(f"Phase 3 done in {time.perf_counter()-t_p3:.0f}s", flush=True)


# ── Save ───────────────────────────────────────────────────────────────
def _make_label(p):
    s   = "+".join(p["sessions"])
    w   = "+".join(f"{k}={v}" for k, v in p["window_mins_per_session"].items())
    ds  = p.get("displacement_style", "UW")[:4]
    thr = p.get("threshold_uw", p.get("threshold_mw", "?"))
    rb  = "BOS" if p.get("require_bos", True) else "noBOS"
    af  = f"af={p['against_fair_mins']}"
    sf  = f"sf={p.get('skip_first_mins', 0)}"
    tp  = f"tp={p['tp_multiple']}"
    am  = f"am={p['atr_mult']}"
    ft  = f"ft={p['tp_beyond_fair_pct']}" if p.get("filter_tp_beyond_fair") else ""
    return f"[{s}|{w}] {ds}{thr} {rb} {af} {sf} {tp} {am} {ft}".strip()


_atomic_dump(DST_PKL, p3_results)

_buf = io.StringIO()
_buf.write("ANCHORED MEAN REVERSION V2 SWEEP RESULTS (corrected MLL rules)\n")
_buf.write(f"Phase 2/3 rerun on top 500 from pre-fix backup\n")
_buf.write(f"Budget=${BUDGET:.0f}  Horizon={HORIZON}d  Goal=${GOAL:.0f}\n\n")
_buf.write(f"{'#':>3}  {'Config':<72}  {'Acct':>5}  {'N':>5}  {'tpd':>5}  {'WR':>6}  {'avgR':>7}  "
           f"{'Sortino':>8}  {'P($0)':>7}  {'P(>bgt)':>8}  {'P(goal)':>8}  "
           f"{'med$':>7}  {'EV/d(med)':>10}  {'ERP':>5}  {'FRP':>5}\n")
_buf.write("-"*190 + "\n")
for rank, r in enumerate(p3_results, 1):
    _buf.write(f"{rank:>3}  {_make_label(r['params']):<72}  {r.get('best_account','25K'):>5}  "
               f"{r['n_t']:>5}  {r['tpd']:>5.2f}  "
               f"{r['wr']:>6.1%}  {r['avgR']:>+7.4f}  {r['sortino']:>8.3f}  "
               f"{r['p_zero']:>7.1%}  {r['p_profit']:>8.1%}  {r['p_goal']:>8.1%}  "
               f"{r['median_cash']:>7.0f}  {r['best_ev_med']:>+10.2f}  "
               f"{r['best_erp']:>5.2f}  {r['best_frp']:>5.2f}\n")
with open(DST_TXT, "w", encoding="utf-8") as f:
    f.write(_buf.getvalue())

print(f"\nResults saved to:")
print(f"  {DST_PKL}")
print(f"  {DST_TXT}")
print("DONE")
