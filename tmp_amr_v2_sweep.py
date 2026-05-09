"""
AnchoredMeanReversion v2 — decomposed parameter sweep.

Sweep A  (structural params, ~16K configs)
  skip_first=0, filter_tp=False, move_sl_to_entry=False, link1400to930=False fixed.
  Ranked by daily Sortino ratio. Min 0.5 tpd required.
  → top 500 survive into Sweep B

Sweep B  (filter params: top-500 × skip_first × filter_tp combos)
  → top 250 overall survive into Phase 2

Phase 2  (ERP × FRP 0.1–1.0 × 0.1–1.0, 100 combos, large pool, no new backtests)
  → final ranked table with per-config optimal risk levels

Parameters swept:
  Sessions × window  (15 combos)
  displacement_style × threshold (6 combos)
  atr_mult           [0.0, 0.5, 1.0, 1.5, 2.0]
  swing_periods      [1, 2]
  require_bos        [True, False]
  against_fair_mins  [0, 15, 30]
  tp_multiple        [1.0, 1.5, 2.0]
  --- Sweep B only ---
  skip_first_mins    [0, 1, 2, 3, 4, 5, 10]
  filter_tp × pct    (False) | (True, 0%..100% in 5% steps)

Metric: Sortino (A/B ranking) → P($0) asc / P(>$10K) desc (Phase 2 final).
Checkpoints every 100 configs.
"""
import os, pickle
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

# ── Checkpoint directory ────────────────────────────────────────────────────────
CKPT_DIR     = "sweep_results_amr_v2"
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_A       = os.path.join(CKPT_DIR, "checkpoint_A.pkl")
CKPT_B       = os.path.join(CKPT_DIR, "checkpoint_B.pkl")
RESULTS_FILE = os.path.join(CKPT_DIR, "final_results.pkl")

with open("md_cache.pkl", "rb") as f:
    _cache = pickle.load(f)
md     = _cache["md"]
n_days = _cache["n_days"]
print(f"Loaded {n_days} days of market data", flush=True)

# Precompute bar → trading-day ordinal for Sortino grouping
_bar_day_ord = np.array([d.toordinal() for d in md.df_1m.index.date], dtype=np.int32)

from backtest.runner.runner import run_backtest
from backtest.runner.config import RunConfig
from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS, extract_normalised_trades
from strategies.profitable.amr_strategy import AnchoredMeanReversionStrategy
from tmp_reinvest import simulate_lifecycles, reinvestment_mc, BUDGET, HORIZON, GOAL

print("Imports done", flush=True)

ACCOUNT   = LUCIDFLEX_ACCOUNTS["25K"]
N_WORKERS = 2

# ERP × FRP grid: 0.1 to 1.0 in 0.1 steps (100 combos) — Phase 2 only
ERP_GRID    = [round(x * 0.1, 1) for x in range(1, 11)]
FRP_GRID    = [round(x * 0.1, 1) for x in range(1, 11)]
RISK_COMBOS = list(product(ERP_GRID, FRP_GRID))   # 100 combos

# Phase 2 large pool
P2_NPOOL = 5_000
P2_NMC   = 1_000

TOP_A     = 500   # survivors from Sweep A into Sweep B
TOP_FINAL = 250   # survivors total for Phase 2

MIN_TRADES = 20   # minimum absolute trade count
MIN_TPD    = 0.5  # minimum trades/day (~2.5/week)

# ── Session/window combos (15) ─────────────────────────────────────────────────
SESSION_WINDOW_COMBOS = [
    # NY Morning (930): 60, 90
    (["930"],        {"930":  60}),
    (["930"],        {"930":  90}),
    # NY Afternoon (1400): 30, 60, 90
    (["1400"],       {"1400": 30}),
    (["1400"],       {"1400": 60}),
    (["1400"],       {"1400": 90}),
    # Asia (2000): 30, 60
    (["2000"],       {"2000": 30}),
    (["2000"],       {"2000": 60}),
    # London (300): 30, 60
    (["300"],        {"300":  30}),
    (["300"],        {"300":  60}),
    # Combined 930+1400 (60/90 × 30/60/90)
    (["930","1400"], {"930":  60, "1400": 30}),
    (["930","1400"], {"930":  60, "1400": 60}),
    (["930","1400"], {"930":  60, "1400": 90}),
    (["930","1400"], {"930":  90, "1400": 30}),
    (["930","1400"], {"930":  90, "1400": 60}),
    (["930","1400"], {"930":  90, "1400": 90}),
]

DISP_THRESH_COMBOS = [
    {"displacement_style": "Upper Wick", "threshold_uw": 0.7},
    {"displacement_style": "Upper Wick", "threshold_uw": 0.8},
    {"displacement_style": "Upper Wick", "threshold_uw": 0.9},
    {"displacement_style": "Marubozu",   "threshold_mw": 0.10},
    {"displacement_style": "Marubozu",   "threshold_mw": 0.15},
    {"displacement_style": "Marubozu",   "threshold_mw": 0.20},
]

ATR_MULT_VALS    = [0.0, 0.5, 1.0, 1.5, 2.0]
SWING_VALS       = [1, 2]
BOS_VALS         = [True, False]
AGAINST_FAIR_VALS= [0, 15, 30]
TP_MULT_VALS     = [1.0, 1.5, 2.0]

# Sweep B filter combos
SKIP_FIRST_VALS_B = [0, 1, 2, 3, 4, 5, 10]
FILTER_TP_VALS_B  = [(False, 50.0)] + [(True, float(p)) for p in range(0, 105, 5)]


def _build_params(sess_win, disp_kw, atr_mult, swing, req_bos, against_fair, tp_mult,
                  skip_first=0, filter_tp=False, filter_pct=50.0):
    sessions, windows = sess_win
    return {
        "sessions":                sessions,
        "window_mins_per_session": windows,
        "atr_mult":                atr_mult,
        "swing_periods":           swing,
        "require_bos":             req_bos,
        "against_fair_mins":       against_fair,
        "tp_multiple":             tp_mult,
        "move_sl_to_entry":        False,
        "link1400to930":           False,
        "skip_first_mins":         skip_first,
        "filter_tp_beyond_fair":   filter_tp,
        "tp_beyond_fair_pct":      filter_pct,
        **disp_kw,
    }


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
    extras = ft
    return f"[{s}|{w}] {ds}{thr} {rb} {af} {sf} {tp} {am} {extras}".strip()


def _build_sweep_a():
    configs = []
    for sw, dkw in product(SESSION_WINDOW_COMBOS, DISP_THRESH_COMBOS):
        for atr_m, swing, req_bos, af, tp_m in product(
            ATR_MULT_VALS, SWING_VALS, BOS_VALS, AGAINST_FAIR_VALS, TP_MULT_VALS,
        ):
            configs.append(_build_params(sw, dkw, atr_m, swing, req_bos, af, tp_m))
    return configs


def _sortino(trades, pnl_pts, sl_dists) -> float:
    """Daily Sortino on R-multiples. Groups by exit bar's trading day."""
    sl_safe     = np.where(sl_dists > 0, sl_dists, 1.0)
    r_per_trade = pnl_pts / sl_safe
    daily: dict[int, float] = {}
    for t, r in zip(trades, r_per_trade):
        day = int(_bar_day_ord[t.exit_bar])
        daily[day] = daily.get(day, 0.0) + r
    daily_r = np.array(list(daily.values()))
    if len(daily_r) < 5:
        return -np.inf
    mean_r  = daily_r.mean()
    down    = daily_r[daily_r < 0]
    if len(down) == 0:
        return np.inf
    down_std = down.std()
    if down_std == 0:
        return np.inf
    return float(mean_r / down_std)


def _run_one(params):
    """Backtest + compute Sortino for ranking. ERP/FRP evaluated in Phase 2 only."""
    try:
        cfg    = RunConfig(starting_capital=100_000, params=params)
        result = run_backtest(AnchoredMeanReversionStrategy, cfg, data=md, validate=False)
        trades = result.trades
        if len(trades) < MIN_TRADES:
            return None
        pnl_pts, sl_dists = extract_normalised_trades(trades)
        n_t    = len(trades)
        tpd    = n_t / n_days
        if tpd < MIN_TPD:
            return None
        sl_safe = np.where(sl_dists > 0, sl_dists, 1.0)
        wr      = sum(1 for t in trades if t.pnl_points > 0) / n_t
        avgR    = float((pnl_pts / sl_safe).mean())
        sortino = _sortino(trades, pnl_pts, sl_dists)
        return dict(params=params, pnl_pts=pnl_pts, sl_dists=sl_dists,
                    n_t=n_t, tpd=tpd, wr=wr, avgR=avgR, sortino=sortino)
    except Exception:
        return None


def _phase2_optimize(rec):
    """Re-evaluate one record across all 100 ERP×FRP combos with large accurate pool."""
    pnl_pts  = rec["pnl_pts"]
    sl_dists = rec["sl_dists"]
    tpd      = rec["tpd"]
    best_p0  = 1.0
    best_p10 = 0.0
    best_erp = 0.2
    best_frp = 1.0
    best_ev  = 0.0
    for erp, frp in RISK_COMBOS:
        try:
            pool = simulate_lifecycles(
                pnl_pts, sl_dists, tpd, ACCOUNT, erp, frp, N=P2_NPOOL, seed=42,
            )
            outcomes = reinvestment_mc(
                pool, ACCOUNT.eval_fee,
                budget=BUDGET, horizon=HORIZON, n_sims=P2_NMC, seed=99,
            )
            p0  = float((outcomes == 0).mean())
            p10 = float((outcomes >= GOAL).mean())
            ev  = pool.get("ev_per_day", 0.0)
            if p0 < best_p0 or (p0 == best_p0 and p10 > best_p10):
                best_p0, best_p10, best_erp, best_frp, best_ev = p0, p10, erp, frp, ev
        except Exception:
            pass
    return dict(**rec, best_p0=best_p0, best_p10=best_p10,
                best_erp=best_erp, best_frp=best_frp, best_ev=best_ev)


def _run_parallel(configs, label, n_workers, ckpt_path=None, resume_done=None):
    results  = []
    done_set = set(resume_done or [])
    pending  = [p for i, p in enumerate(configs) if i not in done_set]
    done     = len(done_set)
    total    = len(configs)
    t0       = _time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        idx_map = {ex.submit(_run_one, p): (i + done) for i, p in enumerate(pending)}
        for fut in as_completed(idx_map):
            done += 1
            r = fut.result()
            if r is not None:
                results.append(r)
            if done % 500 == 0 or done == total:
                el  = _time.perf_counter() - t0
                rem = (el / done) * (total - done) if done > 0 else 0
                print(f"  [{label}] {done}/{total}  valid={len(results)}"
                      f"  elapsed={el:.0f}s  est_remaining={rem:.0f}s", flush=True)
            if ckpt_path and done % 100 == 0:
                with open(ckpt_path, "wb") as f:
                    pickle.dump({"done": done, "results": results}, f)
    return results


# ── Numba warm-up ──────────────────────────────────────────────────────────────
print("Warming up Numba...", flush=True)
_t0 = _time.perf_counter()
run_backtest(AnchoredMeanReversionStrategy, RunConfig(starting_capital=100_000, params={}),
             data=md, validate=False)
print(f"  Numba ready in {_time.perf_counter()-_t0:.1f}s", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# SWEEP A
# ══════════════════════════════════════════════════════════════════════════════
sweep_a_configs = _build_sweep_a()
print(f"\nSweep A configs: {len(sweep_a_configs)}", flush=True)

a_results_extra, a_resume = [], None
if os.path.exists(CKPT_A):
    with open(CKPT_A, "rb") as f:
        ck = pickle.load(f)
    a_results_extra = ck.get("results", [])
    a_done_prev     = ck.get("done", 0)
    a_resume        = list(range(a_done_prev))
    print(f"  Resuming from checkpoint: {a_done_prev} done, "
          f"{len(a_results_extra)} valid", flush=True)

print(f"\n── SWEEP A: structural params ({N_WORKERS} workers) ──", flush=True)
t_a = _time.perf_counter()
a_results_new = _run_parallel(sweep_a_configs, "A", N_WORKERS,
                               ckpt_path=CKPT_A, resume_done=a_resume)
a_results = a_results_extra + a_results_new
print(f"\nSweep A done: {len(a_results)} valid in {_time.perf_counter()-t_a:.0f}s",
      flush=True)

with open(CKPT_A, "wb") as f:
    pickle.dump({"done": len(sweep_a_configs), "results": a_results}, f)

a_results.sort(key=lambda r: -r["sortino"])
top_a = a_results[:TOP_A]
print(f"Top {len(top_a)} from Sweep A (by Sortino) enter Sweep B", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# SWEEP B
# ══════════════════════════════════════════════════════════════════════════════
def _build_sweep_b(top_a_records):
    configs = []
    for rec in top_a_records:
        p0 = rec["params"]
        for sf, (ftp, ftp_pct) in product(SKIP_FIRST_VALS_B, FILTER_TP_VALS_B):
            if sf == 0 and not ftp:
                continue   # already covered in Sweep A
            p = dict(p0)
            p["skip_first_mins"]       = sf
            p["filter_tp_beyond_fair"] = ftp
            p["tp_beyond_fair_pct"]    = ftp_pct
            configs.append(p)
    return configs

sweep_b_configs = _build_sweep_b(top_a)
print(f"\nSweep B configs: {len(sweep_b_configs)}", flush=True)

b_results_extra, b_resume = [], None
if os.path.exists(CKPT_B):
    with open(CKPT_B, "rb") as f:
        ck = pickle.load(f)
    b_results_extra = ck.get("results", [])
    b_done_prev     = ck.get("done", 0)
    b_resume        = list(range(b_done_prev))
    print(f"  Resuming from checkpoint: {b_done_prev} done, "
          f"{len(b_results_extra)} valid", flush=True)

print(f"\n── SWEEP B: filter params ({N_WORKERS} workers) ──", flush=True)
t_b = _time.perf_counter()
b_results_new = _run_parallel(sweep_b_configs, "B", N_WORKERS,
                               ckpt_path=CKPT_B, resume_done=b_resume)
b_results = b_results_extra + b_results_new
print(f"\nSweep B done: {len(b_results)} valid in {_time.perf_counter()-t_b:.0f}s",
      flush=True)

with open(CKPT_B, "wb") as f:
    pickle.dump({"done": len(sweep_b_configs), "results": b_results}, f)

all_results = top_a + b_results
all_results.sort(key=lambda r: -r["sortino"])
top_final = all_results[:TOP_FINAL]
print(f"\nCombined {len(all_results)} valid → top {len(top_final)} enter Phase 2",
      flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — ERP × FRP optimization
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n── PHASE 2: ERP×FRP sweep for top {len(top_final)} ({N_WORKERS} workers) ──",
      flush=True)
p2_results = []
t_p2 = _time.perf_counter()
done2 = 0
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futs = {ex.submit(_phase2_optimize, r): r for r in top_final}
    for fut in as_completed(futs):
        done2 += 1
        p2_results.append(fut.result())
        if done2 % 25 == 0 or done2 == len(top_final):
            el = _time.perf_counter() - t_p2
            print(f"  {done2}/{len(top_final)}  elapsed={el:.0f}s", flush=True)

print(f"\nPhase 2 done in {_time.perf_counter()-t_p2:.0f}s", flush=True)
p2_results.sort(key=lambda r: (r["best_p0"], -r["best_p10"]))

with open(RESULTS_FILE, "wb") as f:
    pickle.dump(p2_results, f)
print(f"Results saved → {RESULTS_FILE}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════
W = 165
print("\n" + "="*W)
print("  ANCHORED MEAN REVERSION V2 SWEEP — ranked P($0) asc → P(>$10K) desc  [Phase-2 propfirm optimal]")
print(f"  Account: 25K LucidFlex  Budget=$300  Horizon=84d  Goal=$10K")
print(f"  Sweep A/B ranked by daily Sortino | Phase 2: ERP/FRP 0.1–1.0 × 0.1–1.0 (100 combos)")
print("="*W)
hdr = (f"{'#':>3}  {'Config':<78}  {'N':>5}  {'tpd':>5}  {'WR':>6}  {'avgR':>7}  "
       f"{'Sortino':>8}  {'P($0)':>7}  {'P>10K':>7}  {'EV/d':>7}  {'ERP':>5}  {'FRP':>5}")
print(hdr)
print("-"*W)
for rank, r in enumerate(p2_results[:50], 1):
    label = _make_label(r["params"])
    print(f"{rank:>3}  {label:<78}  {r['n_t']:>5}  {r['tpd']:>5.2f}  "
          f"{r['wr']:>6.1%}  {r['avgR']:>+7.4f}  {r['sortino']:>8.3f}  "
          f"{r['best_p0']:>7.1%}  {r['best_p10']:>7.1%}  {r['best_ev']:>7.2f}  "
          f"{r['best_erp']:>5.2f}  {r['best_frp']:>5.2f}")

if p2_results:
    best = p2_results[0]
    p    = best["params"]
    print("\n" + "="*W)
    print("  BEST CONFIG — FULL DETAIL")
    print("="*W)
    print(f"  {_make_label(p)}")
    print(f"\n  Trade stats:")
    print(f"    Trades={best['n_t']}  tpd={best['tpd']:.3f}  "
          f"WR={best['wr']:.1%}  avgR={best['avgR']:+.4f}  Sortino={best['sortino']:.3f}")
    print(f"\n  Propfirm outcome (optimal risk):")
    print(f"    P($0)={best['best_p0']:.1%}  P(>$10K)={best['best_p10']:.1%}  "
          f"EV/day=${best['best_ev']:.2f}")
    print(f"\n  Full parameters:")
    for k, v in sorted(p.items()):
        print(f"    {k}: {v}")
    erp = best["best_erp"]
    frp = best["best_frp"]
    n_accounts = int(BUDGET // ACCOUNT.eval_fee)
    print(f"\n  Account strategy:")
    print(f"    Account:          25K LucidFlex  (eval fee=${ACCOUNT.eval_fee:.0f})")
    print(f"    Starting capital: ${BUDGET:.0f} → {n_accounts} accounts at once")
    print(f"    Eval risk (ERP):  {erp:.0%} of MLL = ${erp * ACCOUNT.mll_amount:.0f}/trade target risk")
    print(f"    Funded risk (FRP):{frp:.0%} of funded drawdown limit per trade")
    print(f"    Reinvest rule:    deposit every payout into new evals immediately")
    print(f"    Payout:           min(profits×50%, $1500) × 90% net to trader")
