"""
AnchoredMeanReversion v2 — decomposed parameter sweep.

Sweep A  (structural params, ~16K configs)
  skip_first=0, filter_tp=False, move_sl_to_entry=False, link1400to930=False fixed.
  Ranked by daily Sortino ratio. Min 0.5 tpd required.
  -> top 500 survive into Sweep B

Sweep B  (filter params: top-500 × skip_first × filter_tp combos)
  -> top 250 overall survive into Phase 2

Phase 2  (ERP × FRP 0.1–1.0 × 0.1–1.0, 100 combos, large pool, no new backtests)
  -> final ranked table with per-config optimal risk levels

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

Metric: Sortino (A/B ranking) -> P($0) asc / P(>$10K) desc (Phase 2 final).
Checkpoints every 100 configs.
"""
import os, pickle
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

# -- Checkpoint directory --──────────────────────────────────────────────────────
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

# Precompute bar -> trading-day ordinal for Sortino grouping
_bar_day_ord = np.array([d.toordinal() for d in md.df_1m.index.date], dtype=np.int32)

from backtest.runner.runner import run_backtest
from backtest.runner.config import RunConfig
from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS, extract_normalised_trades, simulate_eval_batch, simulate_funded_batch, MICRO_COMM_RT, MINI_COMM_RT
from strategies.profitable.amr_strategy import AnchoredMeanReversionStrategy
# tmp_reinvest no longer used in Phase 2 — metrics come from lucidflex directly

print("Imports done", flush=True)

ACCOUNT   = LUCIDFLEX_ACCOUNTS["25K"]
N_WORKERS = 2

# ERP × FRP grid: 0.1 to 1.0 in 0.1 steps (100 combos) — Phase 2 only
ERP_GRID    = [round(x * 0.1, 1) for x in range(1, 11)]
FRP_GRID    = [round(x * 0.1, 1) for x in range(1, 11)]
RISK_COMBOS = list(product(ERP_GRID, FRP_GRID))   # 100 combos

# Phase 2 pool (smaller = faster, still accurate enough for ranking)
P2_NPOOL = 2_000
P2_NMC   = 500

BUDGET    = 300.0  # reinvestment budget for account strategy section
TOP_A     = 9999  # all valid A results enter Sweep B (capped by actual count)
TOP_FINAL = 500   # survivors total for Phase 2

MIN_TRADES = 20   # minimum absolute trade count
MIN_TPD    = 0.5  # minimum trades/day (~2.5/week)

# -- Session/window combos (15) --───────────────────────────────────────────────
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
SWING_VALS       = [1]        # swing_periods=2 produces identical signals — removed
BOS_VALS         = [True, False]
AGAINST_FAIR_VALS= [0, 15, 30]
TP_MULT_VALS     = [1.0, 1.5, 2.0]

# Sweep B filter combos
SKIP_FIRST_VALS_B = [0, 1, 2, 3, 4, 5, 10]
FILTER_TP_VALS_B  = [(False, 50.0)] + [(True, float(p)) for p in range(0, 105, 5)]

# NQ mini commission matching lucidflex (MINI_COMM_RT = $3.50 RT -> $1.75/side)
COMM_PER_CONTRACT = MINI_COMM_RT / 2.0   # $1.75 per contract per side

# -- Fair-price cache: {(date, sess_label) -> 1m open at session start} --────────
_SESS_START_MIN = {'930': 9*60+30, '1400': 14*60, '2000': 20*60, '300': 3*60}
_FAIR_CACHE: dict = {}
_idx = md.df_1m.index
_open_arr = md.df_1m['open'].values
for _sl, (_h, _m) in [('930',(9,30)),('1400',(14,0)),('2000',(20,0)),('300',(3,0))]:
    _mask = (_idx.hour == _h) & (_idx.minute == _m)
    for _pos in np.where(_mask)[0]:
        _FAIR_CACHE[(_idx[_pos].date(), _sl)] = float(_open_arr[_pos])


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


def _sortino_from_r(r_per_trade: np.ndarray, exit_bars: np.ndarray) -> float:
    """Daily Sortino on R-multiples given pre-computed per-trade R and exit bars."""
    days_ord = _bar_day_ord[exit_bars]
    u, inv   = np.unique(days_ord, return_inverse=True)
    daily_r  = np.zeros(len(u), dtype=np.float64)
    np.add.at(daily_r, inv, r_per_trade)
    if len(daily_r) < 5:
        return -np.inf
    down = daily_r[daily_r < 0]
    if len(down) == 0:
        return np.inf
    down_std = down.std()
    return float(daily_r.mean() / down_std) if down_std > 0 else np.inf


def _sortino(trades, pnl_pts, sl_dists) -> float:
    sl_safe   = np.where(sl_dists > 0, sl_dists, 1.0)
    exit_bars = np.array([t.exit_bar for t in trades], dtype=np.int32)
    return _sortino_from_r(pnl_pts / sl_safe, exit_bars)


def _extract_filter_meta(trades, params) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-trade filter metadata without re-running the backtest.

    Returns:
      elapsed_mins  (n_trades,) float32 — minutes into session when signal fired
      beyond_ratio  (n_trades,) float32 — TP-beyond-fair ratio (inf = always passes)
    """
    sessions = params['sessions']
    windows  = params['window_mins_per_session']
    n = len(trades)
    elapsed_mins = np.zeros(n, dtype=np.float32)
    beyond_ratio = np.full(n, np.inf, dtype=np.float32)

    idx = md.df_1m.index
    for j, trade in enumerate(trades):
        bar_ts  = idx[trade.entry_bar]
        bar_min = bar_ts.hour * 60 + bar_ts.minute

        # Identify session this bar belongs to
        matched = None
        for sl in sessions:
            start = _SESS_START_MIN.get(sl)
            if start is None:
                continue
            win = windows.get(sl, 60)
            if start <= bar_min < start + win:
                matched = sl
                break
        if matched is None:
            continue

        elapsed_mins[j] = bar_min - _SESS_START_MIN[matched]

        fair = _FAIR_CACHE.get((bar_ts.date(), matched))
        if fair is None:
            continue
        entry = trade.entry_price
        itp   = trade.initial_tp_price
        if itp is None:
            continue
        tp_dist = abs(itp - entry)
        if tp_dist <= 0:
            continue

        if trade.direction == 1 and entry < fair:    # long toward fair
            if itp > fair:
                beyond_ratio[j] = (fair - entry) / tp_dist
        elif trade.direction == -1 and entry > fair: # short toward fair
            if itp < fair:
                beyond_ratio[j] = (entry - fair) / tp_dist
        # away-from-fair or TP doesn't cross fair -> stays inf

    return elapsed_mins, beyond_ratio


def _run_one(params):
    """Backtest + extract all data needed for Sweep A ranking and fast Sweep B."""
    try:
        cfg    = RunConfig(starting_capital=100_000,
                           commission_per_contract=COMM_PER_CONTRACT, params=params)
        result = run_backtest(AnchoredMeanReversionStrategy, cfg, data=md, validate=False)
        trades = result.trades
        if len(trades) < MIN_TRADES:
            return None
        pnl_pts, sl_dists = extract_normalised_trades(trades)
        n_t  = len(trades)
        tpd  = n_t / n_days
        if tpd < MIN_TPD:
            return None
        sl_safe   = np.where(sl_dists > 0, sl_dists, 1.0)
        exit_bars = np.array([t.exit_bar for t in trades], dtype=np.int32)
        wr        = float((pnl_pts > 0).mean())
        avgR      = float((pnl_pts / sl_safe).mean())
        sortino   = _sortino_from_r(pnl_pts / sl_safe, exit_bars)
        elapsed_mins, beyond_ratio = _extract_filter_meta(trades, params)
        return dict(params=params, pnl_pts=pnl_pts, sl_dists=sl_dists,
                    exit_bars=exit_bars, elapsed_mins=elapsed_mins,
                    beyond_ratio=beyond_ratio,
                    n_t=n_t, tpd=tpd, wr=wr, avgR=avgR, sortino=sortino)
    except Exception:
        return None


def _phase2_optimize(rec):
    """
    Re-evaluate one record across all 100 ERP x FRP combos using lucidflex
    eval + funded kernels directly. All metrics from single-account analysis.

    EV/day = (pass_rate * mean_funded_payout - eval_fee) / mean_total_days
    Ranked by EV/day(mean) descending.
    """
    pnl_pts  = rec["pnl_pts"]
    sl_dists = rec["sl_dists"]
    tpd      = rec["tpd"]
    max_minis = int(ACCOUNT.max_micros) // 10

    best_ev     = -np.inf
    best_ev_med = 0.0
    best_erp    = ERP_GRID[0]
    best_frp    = FRP_GRID[0]
    best_pass   = 0.0
    best_payout = 0.0
    best_pay_med= 0.0

    rng = np.random.default_rng(42)

    for erp, frp in RISK_COMBOS:
        try:
            passed, _, eval_days = simulate_eval_batch(
                pnl_pts, sl_dists, ACCOUNT,
                risk_pct=erp, scheme="fixed_dollar", sizing_mode="nq_first",
                rng=rng, trades_per_day=tpd, n_sims=P2_NPOOL,
                max_minis=max_minis, mini_comm_rt=MINI_COMM_RT,
            )
            pass_rate      = float(passed.mean())
            mean_eval_days = float(eval_days.mean())
            n_passed       = int(passed.sum())

            if n_passed > 0:
                sb_f = np.full(n_passed, float(ACCOUNT.starting_balance), dtype=np.float32)
                total_w, _, _, _, _, funded_days = simulate_funded_batch(
                    pnl_pts, sl_dists, ACCOUNT,
                    risk_pct=frp, scheme="fixed_dollar", sizing_mode="nq_first",
                    rng=rng, trades_per_day=tpd, starting_balances=sb_f,
                    max_minis=max_minis, mini_comm_rt=MINI_COMM_RT,
                )
                mean_payout   = float(total_w.mean())
                median_payout = float(np.median(total_w))
                mean_fund_days= float(funded_days.mean())
            else:
                mean_payout = median_payout = mean_fund_days = 0.0

            total_days  = mean_eval_days + pass_rate * mean_fund_days
            ev_day      = (pass_rate * mean_payout    - ACCOUNT.eval_fee) / max(total_days, 1.0)
            ev_day_med  = (pass_rate * median_payout  - ACCOUNT.eval_fee) / max(total_days, 1.0)

            if ev_day > best_ev:
                best_ev     = ev_day
                best_ev_med = ev_day_med
                best_erp    = erp
                best_frp    = frp
                best_pass   = pass_rate
                best_payout = mean_payout
                best_pay_med= median_payout
        except Exception:
            pass

    return dict(**rec,
                best_ev=best_ev, best_ev_med=best_ev_med,
                best_erp=best_erp, best_frp=best_frp,
                best_pass=best_pass,
                best_payout=best_payout, best_pay_med=best_pay_med)


def _run_parallel(configs, label, n_workers, ckpt_path=None, resume_done=None,
                  extra_results=None):
    """extra_results: results from previous runs to include in every checkpoint."""
    extra    = extra_results or []
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
                print(f"  [{label}] {done}/{total}  valid={len(extra)+len(results)}"
                      f"  elapsed={el:.0f}s  est_remaining={rem:.0f}s", flush=True)
            if ckpt_path and done % 100 == 0:
                with open(ckpt_path, "wb") as f:
                    pickle.dump({"done": done, "results": extra + results}, f)
    return results


# -- Numba warm-up --────────────────────────────────────────────────────────────
print("Warming up Numba...", flush=True)
_t0 = _time.perf_counter()
run_backtest(AnchoredMeanReversionStrategy,
             RunConfig(starting_capital=100_000,
                       commission_per_contract=COMM_PER_CONTRACT, params={}),
             data=md, validate=False)
# Also compile lucidflex eval/funded kernels so Phase 2 doesn't stall
_dummy_pnl = np.zeros(10, dtype=np.float32)
_dummy_sl  = np.ones(10,  dtype=np.float32) * 5.0
_dummy_rng = np.random.default_rng(0)
simulate_eval_batch(_dummy_pnl, _dummy_sl, ACCOUNT, 0.2, "fixed_dollar", "nq_first",
                    _dummy_rng, 1.0, 50)
_dummy_sb = np.full(10, float(ACCOUNT.starting_balance), dtype=np.float32)
simulate_funded_batch(_dummy_pnl, _dummy_sl, ACCOUNT, 0.2, "fixed_dollar", "nq_first",
                      _dummy_rng, 1.0, _dummy_sb)
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

print(f"\n-- SWEEP A: structural params ({N_WORKERS} workers) --", flush=True)
t_a = _time.perf_counter()
a_results_new = _run_parallel(sweep_a_configs, "A", N_WORKERS,
                               ckpt_path=CKPT_A, resume_done=a_resume,
                               extra_results=a_results_extra)
a_results = a_results_extra + a_results_new
print(f"\nSweep A done: {len(a_results)} valid in {_time.perf_counter()-t_a:.0f}s",
      flush=True)

with open(CKPT_A, "wb") as f:
    pickle.dump({"done": len(sweep_a_configs), "results": a_results}, f)

a_results.sort(key=lambda r: -r["sortino"])
top_a = a_results[:TOP_A]
print(f"Top {len(top_a)} from Sweep A (by Sortino) enter Sweep B", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# SWEEP B  (fast post-processing — no new backtests)
# ══════════════════════════════════════════════════════════════════════════════
def _score_filtered(rec, skip_first: int, filter_tp: bool, ftp_pct: float):
    """Apply skip_first / filter_tp as NumPy masks on a cached A record."""
    mask = np.ones(rec['n_t'], dtype=bool)
    if skip_first > 0:
        mask &= rec['elapsed_mins'] >= skip_first
    if filter_tp:
        mask &= rec['beyond_ratio'] >= (ftp_pct / 100.0)
    n_kept = int(mask.sum())
    if n_kept < MIN_TRADES:
        return None
    pnl_f = rec['pnl_pts'][mask]
    sl_f  = rec['sl_dists'][mask]
    eb_f  = rec['exit_bars'][mask]
    tpd   = n_kept / n_days
    if tpd < MIN_TPD:
        return None
    sl_safe = np.where(sl_f > 0, sl_f, 1.0)
    wr      = float((pnl_f > 0).mean())
    avgR    = float((pnl_f / sl_safe).mean())
    sortino = _sortino_from_r(pnl_f / sl_safe, eb_f)
    new_params = dict(rec['params'])
    new_params['skip_first_mins']     = skip_first
    new_params['filter_tp_beyond_fair'] = filter_tp
    new_params['tp_beyond_fair_pct']  = ftp_pct
    return dict(params=new_params, pnl_pts=pnl_f, sl_dists=sl_f, exit_bars=eb_f,
                elapsed_mins=rec['elapsed_mins'][mask],
                beyond_ratio=rec['beyond_ratio'][mask],
                n_t=n_kept, tpd=tpd, wr=wr, avgR=avgR, sortino=sortino)


n_b_combos = len(SKIP_FIRST_VALS_B) * len(FILTER_TP_VALS_B) - 1  # minus skip_first=0,no_filter
print(f"\nSweep B: {len(top_a)} base configs x {n_b_combos} filter combos "
      f"= {len(top_a)*n_b_combos} scored (no new backtests)", flush=True)
print(f"\n-- SWEEP B: fast filter post-processing --", flush=True)
t_b = _time.perf_counter()
b_results = []
total_b = len(top_a) * (len(SKIP_FIRST_VALS_B) * len(FILTER_TP_VALS_B))
done_b   = 0
for rec in top_a:
    for sf, (ftp, ftp_pct) in product(SKIP_FIRST_VALS_B, FILTER_TP_VALS_B):
        if sf == 0 and not ftp:
            done_b += 1
            continue  # already in A results
        r = _score_filtered(rec, sf, ftp, ftp_pct)
        if r is not None:
            b_results.append(r)
        done_b += 1
        if done_b % 5000 == 0:
            print(f"  [B] {done_b}/{total_b}  valid={len(b_results)}", flush=True)
print(f"\nSweep B done: {len(b_results)} valid in {_time.perf_counter()-t_b:.1f}s",
      flush=True)

all_results = top_a + b_results
all_results.sort(key=lambda r: -r["sortino"])
top_final = all_results[:TOP_FINAL]
print(f"Combined {len(all_results)} valid -> top {len(top_final)} enter Phase 2",
      flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — ERP × FRP optimization
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n-- PHASE 2: ERP x FRP sweep for top {len(top_final)} ({N_WORKERS} workers) --",
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
p2_results.sort(key=lambda r: -r["best_ev"])

with open(RESULTS_FILE, "wb") as f:
    pickle.dump(p2_results, f)
print(f"Results saved -> {RESULTS_FILE}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════
W = 165
print("\n" + "="*W)
print("  ANCHORED MEAN REVERSION V2 SWEEP — ranked P($0) asc -> P(>$10K) desc  [Phase-2 propfirm optimal]")
print(f"  Account: 25K LucidFlex  Budget=$300  Horizon=84d  Goal=$10K")
print(f"  Sweep A/B ranked by daily Sortino | Phase 2: ERP/FRP 0.1–1.0 × 0.1–1.0 (100 combos)")
print("="*W)
hdr = (f"{'#':>3}  {'Config':<78}  {'N':>5}  {'tpd':>5}  {'WR':>6}  {'avgR':>7}  "
       f"{'Sortino':>8}  {'pass%':>6}  {'payout':>8}  {'EV/d(mean)':>11}  {'EV/d(med)':>10}  {'ERP':>5}  {'FRP':>5}")
print(hdr)
print("-"*W)
for rank, r in enumerate(p2_results[:50], 1):
    label = _make_label(r["params"])
    print(f"{rank:>3}  {label:<78}  {r['n_t']:>5}  {r['tpd']:>5.2f}  "
          f"{r['wr']:>6.1%}  {r['avgR']:>+7.4f}  {r['sortino']:>8.3f}  "
          f"{r['best_pass']:>6.1%}  {r['best_payout']:>8.0f}  "
          f"{r['best_ev']:>11.2f}  {r['best_ev_med']:>10.2f}  "
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
    print(f"    Pass rate={best['best_pass']:.1%}  Mean payout=${best['best_payout']:.0f}  Median payout=${best['best_pay_med']:.0f}")
    print(f"    EV/day(mean)=${best['best_ev']:.2f}  EV/day(median)=${best['best_ev_med']:.2f}")
    print(f"\n  Full parameters:")
    for k, v in sorted(p.items()):
        print(f"    {k}: {v}")
    erp = best["best_erp"]
    frp = best["best_frp"]
    n_accounts = int(BUDGET // ACCOUNT.eval_fee)
    print(f"\n  Account strategy:")
    print(f"    Account:          25K LucidFlex  (eval fee=${ACCOUNT.eval_fee:.0f})")
    print(f"    Starting capital: ${BUDGET:.0f} -> {n_accounts} accounts at once")
    print(f"    Eval risk (ERP):  {erp:.0%} of MLL = ${erp * ACCOUNT.mll_amount:.0f}/trade target risk")
    print(f"    Funded risk (FRP):{frp:.0%} of funded drawdown limit per trade")
    print(f"    Reinvest rule:    deposit every payout into new evals immediately")
    print(f"    Payout:           min(profits×50%, $1500) × 90% net to trader")

# ── Save to git-tracked location so results survive push/pull ─────────────────
import io as _io, shutil as _sh

_LOG_DIR = "sweeps/logs"
os.makedirs(_LOG_DIR, exist_ok=True)

# pkl — full results for programmatic reload
_pkl_dst = os.path.join(_LOG_DIR, "amr_v2_results.pkl")
with open(_pkl_dst, "wb") as _f:
    pickle.dump(p2_results, _f)

# txt — human-readable table (same as stdout above)
_txt_dst = os.path.join(_LOG_DIR, "amr_v2_results.txt")
_buf = _io.StringIO()
_buf.write("ANCHORED MEAN REVERSION V2 SWEEP RESULTS\n")
_buf.write(f"Dataset: {n_days} days (2019+)  Account: 25K LucidFlex  Budget=$300  Horizon=84d  Goal=$10K\n")
_buf.write(f"ERP/FRP: 0.1-1.0 x 0.1-1.0 (100 combos, optimized per config)\n\n")
_buf.write(f"{'#':>3}  {'Config':<78}  {'N':>5}  {'tpd':>5}  {'WR':>6}  {'avgR':>7}  "
           f"{'Sortino':>8}  {'pass%':>6}  {'payout':>8}  {'EV/d(mean)':>11}  {'EV/d(med)':>10}  {'ERP':>5}  {'FRP':>5}\n")
_buf.write("-"*175 + "\n")
for rank, r in enumerate(p2_results, 1):
    label = _make_label(r["params"])
    _buf.write(f"{rank:>3}  {label:<78}  {r['n_t']:>5}  {r['tpd']:>5.2f}  "
               f"{r['wr']:>6.1%}  {r['avgR']:>+7.4f}  {r['sortino']:>8.3f}  "
               f"{r['best_pass']:>6.1%}  {r['best_payout']:>8.0f}  "
               f"{r['best_ev']:>11.2f}  {r['best_ev_med']:>10.2f}  "
               f"{r['best_erp']:>5.2f}  {r['best_frp']:>5.2f}\n")
with open(_txt_dst, "w", encoding="utf-8") as _f:
    _f.write(_buf.getvalue())

# Copy sweep log too
_sh.copy2("sweep_results_amr_v2/sweep.log", os.path.join(_LOG_DIR, "amr_v2_sweep.log"))

print(f"\nResults saved to git-tracked paths:")
print(f"  {_pkl_dst}")
print(f"  {_txt_dst}")
print(f"  {_LOG_DIR}/amr_v2_sweep.log")
print(f"\nYour friend can now: git add sweeps/logs/ && git commit -m 'results: AnchoredMeanReversion v2 sweep' && git push")
