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

# Phase 3 reinvestment MC settings
P3_NPOOL = 5_000   # lifecycle pool size
P3_NMC   = 2_000   # reinvestment MC sims

BUDGET    = 1_000.0  # reinvestment starting budget
HORIZON   = 84       # trading days
GOAL      = 10_000.0 # target cash
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
    Find optimal (account, ERP, FRP) across all LucidFlex accounts and the
    10x10 ERP x FRP grid, ranked by median EV/day.

    Key optimization: eval depends only on ERP; funded depends only on FRP.
    So we run 10 eval + 10 funded calls per account instead of 100x2 = 200.
    With 4 accounts: 80 kernel calls vs 800. Results combined as cross-product.
    """
    pnl_pts, sl_dists, _ = _get_arrays(rec)
    tpd = rec["tpd"]

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
        sb_f      = np.full(P2_NPOOL, float(acct.starting_balance), dtype=np.float32)
        rng       = np.random.default_rng(42)

        # --- Eval: one kernel call per ERP value ---
        eval_cache: dict = {}
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

        # --- Funded: one kernel call per FRP value (P2_NPOOL fixed sims) ---
        funded_cache: dict = {}
        for frp in FRP_GRID:
            try:
                total_w, _, _, _, _, funded_days = simulate_funded_batch(
                    pnl_pts, sl_dists, acct,
                    risk_pct=frp, scheme="fixed_dollar", sizing_mode="nq_first",
                    rng=rng, trades_per_day=tpd, starting_balances=sb_f,
                    max_minis=max_minis, mini_comm_rt=MINI_COMM_RT,
                )
                funded_cache[frp] = (float(total_w.mean()), float(np.median(total_w)),
                                     float(funded_days.mean()))
            except Exception:
                funded_cache[frp] = (0.0, 0.0, 1.0)

        # --- Step 1: pick ERP that maximises pass rate ---
        best_erp_acct = max(ERP_GRID, key=lambda e: eval_cache[e][0])
        pass_rate, mean_eval_days = eval_cache[best_erp_acct]

        # --- Step 2: given best ERP, pick FRP that maximises median EV/day ---
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
                best_pass=best_pass,
                best_payout=best_payout, best_pay_med=best_pay_med,
                best_account=best_acct)


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
def _get_arrays(rec):
    """Return (pnl_pts, sl_dists, exit_bars) — works for both A and lazy B records."""
    if '_parent' in rec:
        idx = rec['_indices']
        p   = rec['_parent']
        return p['pnl_pts'][idx], p['sl_dists'][idx], p['exit_bars'][idx]
    return rec['pnl_pts'], rec['sl_dists'], rec['exit_bars']


def _score_filtered(rec, skip_first: int, filter_tp: bool, ftp_pct: float):
    """Apply skip_first / filter_tp masks on a cached A record.
    Returns a lightweight record: stats + parent ref + indices (no array copies)."""
    mask = np.ones(rec['n_t'], dtype=bool)
    if skip_first > 0:
        mask &= rec['elapsed_mins'] >= skip_first
    if filter_tp:
        mask &= rec['beyond_ratio'] >= (ftp_pct / 100.0)
    n_kept = int(mask.sum())
    if n_kept < MIN_TRADES:
        return None
    tpd = n_kept / n_days
    if tpd < MIN_TPD:
        return None
    indices = np.where(mask)[0].astype(np.int32)   # ~4KB vs ~20KB for full arrays
    pnl_f   = rec['pnl_pts'][indices]
    sl_f    = rec['sl_dists'][indices]
    eb_f    = rec['exit_bars'][indices]
    sl_safe = np.where(sl_f > 0, sl_f, 1.0)
    wr      = float((pnl_f > 0).mean())
    avgR    = float((pnl_f / sl_safe).mean())
    sortino = _sortino_from_r(pnl_f / sl_safe, eb_f)
    new_params = dict(rec['params'])
    new_params['skip_first_mins']       = skip_first
    new_params['filter_tp_beyond_fair'] = filter_tp
    new_params['tp_beyond_fair_pct']    = ftp_pct
    # Store parent ref + indices — arrays reconstructed on demand in Phase 2
    return dict(params=new_params, _parent=rec, _indices=indices,
                n_t=n_kept, tpd=tpd, wr=wr, avgR=avgR, sortino=sortino)


import heapq as _hq

_B_CAP = TOP_FINAL * 20   # keep at most this many B results in memory at once

n_b_combos = len(SKIP_FIRST_VALS_B) * len(FILTER_TP_VALS_B) - 1
print(f"\nSweep B: {len(top_a)} base configs x {n_b_combos} filter combos "
      f"= {len(top_a)*n_b_combos} scored (no new backtests)", flush=True)
print(f"  Memory-capped heap: keeping best {_B_CAP} B results")
print(f"\n-- SWEEP B: fast filter post-processing --", flush=True)
t_b     = _time.perf_counter()
b_heap  = []   # min-heap of (-sortino, counter, rec) — counter breaks sortino ties
_b_ctr  = 0
total_b = len(top_a) * (len(SKIP_FIRST_VALS_B) * len(FILTER_TP_VALS_B))
done_b  = 0
for rec in top_a:
    for sf, (ftp, ftp_pct) in product(SKIP_FIRST_VALS_B, FILTER_TP_VALS_B):
        if sf == 0 and not ftp:
            done_b += 1
            continue
        r = _score_filtered(rec, sf, ftp, ftp_pct)
        if r is not None:
            entry = (-r['sortino'], _b_ctr, r)
            _b_ctr += 1
            if len(b_heap) < _B_CAP:
                _hq.heappush(b_heap, entry)
            elif entry[0] < b_heap[0][0]:   # better than worst in heap
                _hq.heapreplace(b_heap, entry)
        done_b += 1
        if done_b % 50000 == 0:
            print(f"  [B] {done_b}/{total_b}  heap={len(b_heap)}", flush=True)
b_results = [r for _, _, r in b_heap]
print(f"\nSweep B done: {len(b_results)} in heap in {_time.perf_counter()-t_b:.1f}s",
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
p2_results.sort(key=lambda r: -r["best_ev"])   # interim sort for checkpoint
with open(RESULTS_FILE, "wb") as f:
    pickle.dump(p2_results, f)
print(f"Phase 2 interim results saved -> {RESULTS_FILE}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Reinvestment MC: compute P($0) and P(>BUDGET), re-rank
# ══════════════════════════════════════════════════════════════════════════════
from tmp_reinvest import simulate_lifecycles, reinvestment_mc

print(f"\n-- PHASE 3: reinvestment MC (budget=${BUDGET:.0f}, horizon={HORIZON}d) --",
      flush=True)
t_p3 = _time.perf_counter()


def _phase3_mc(rec):
    pnl_pts, sl_dists, _ = _get_arrays(rec)
    acct = LUCIDFLEX_ACCOUNTS[rec.get("best_account", "25K")]
    pool = simulate_lifecycles(
        pnl_pts, sl_dists, rec["tpd"], acct,
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
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futs3 = {ex.submit(_phase3_mc, r): r for r in p2_results}
    p3_results = []
    for fut in as_completed(futs3):
        done3 += 1
        p3_results.append(fut.result())
        if done3 % 50 == 0 or done3 == len(p2_results):
            el = _time.perf_counter() - t_p3
            print(f"  {done3}/{len(p2_results)}  elapsed={el:.0f}s", flush=True)

p3_results.sort(key=lambda r: r["p_zero"])   # final rank: min P($0) first
print(f"\nPhase 3 done in {_time.perf_counter()-t_p3:.0f}s", flush=True)

with open(RESULTS_FILE, "wb") as f:
    pickle.dump(p3_results, f)
print(f"Results saved -> {RESULTS_FILE}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════
W = 185
print("\n" + "="*W)
print(f"  ANCHORED MEAN REVERSION V2 SWEEP — ranked min P($0)  [budget=${BUDGET:.0f}  horizon={HORIZON}d  goal=${GOAL:.0f}]")
print(f"  Account: 25K LucidFlex")
print(f"  Sweep A/B: daily Sortino | Phase 2: ERP/FRP optimised by EV/day | Phase 3: MC re-rank by P($0)")
print("="*W)
hdr = (f"{'#':>3}  {'Config':<72}  {'Acct':>5}  {'N':>5}  {'tpd':>5}  {'WR':>6}  {'avgR':>7}  "
       f"{'Sortino':>8}  {'P($0)':>7}  {'P(>bgt)':>8}  {'P(goal)':>8}  "
       f"{'med$':>7}  {'EV/d(med)':>10}  {'ERP':>5}  {'FRP':>5}")
print(hdr)
print("-"*W)
for rank, r in enumerate(p3_results[:50], 1):
    label = _make_label(r["params"])
    print(f"{rank:>3}  {label:<72}  {r.get('best_account','25K'):>5}  "
          f"{r['n_t']:>5}  {r['tpd']:>5.2f}  "
          f"{r['wr']:>6.1%}  {r['avgR']:>+7.4f}  {r['sortino']:>8.3f}  "
          f"{r['p_zero']:>7.1%}  {r['p_profit']:>8.1%}  {r['p_goal']:>8.1%}  "
          f"{r['median_cash']:>7.0f}  {r['best_ev_med']:>+10.2f}  "
          f"{r['best_erp']:>5.2f}  {r['best_frp']:>5.2f}")

if p3_results:
    best  = p3_results[0]
    p     = best["params"]
    ba    = LUCIDFLEX_ACCOUNTS[best.get("best_account", "25K")]
    print("\n" + "="*W)
    print("  BEST CONFIG — FULL DETAIL")
    print("="*W)
    print(f"  {_make_label(p)}")
    print(f"\n  Trade stats:")
    print(f"    Trades={best['n_t']}  tpd={best['tpd']:.3f}  "
          f"WR={best['wr']:.1%}  avgR={best['avgR']:+.4f}  Sortino={best['sortino']:.3f}")
    print(f"\n  Propfirm outcome  [{ba.name} LucidFlex — optimal ERP/FRP by median EV/day]:")
    print(f"    Pass rate={best['best_pass']:.1%}  Mean payout=${best['best_payout']:.0f}  "
          f"Median payout=${best['best_pay_med']:.0f}")
    print(f"    EV/day(mean)=${best['best_ev']:.2f}  EV/day(median)=${best['best_ev_med']:.2f}")
    print(f"\n  Reinvestment MC (budget=${BUDGET:.0f}, horizon={HORIZON}d):")
    print(f"    P($0)={best['p_zero']:.1%}  P(>budget)={best['p_profit']:.1%}  "
          f"P(goal)={best['p_goal']:.1%}  median=${best['median_cash']:.0f}")
    print(f"\n  Full parameters:")
    for k, v in sorted(p.items()):
        print(f"    {k}: {v}")
    erp          = best["best_erp"]
    frp          = best["best_frp"]
    n_accounts   = int(BUDGET // ba.eval_fee)
    print(f"\n  Account strategy:")
    print(f"    Account:          {ba.name} LucidFlex  (eval fee=${ba.eval_fee:.0f})")
    print(f"    Starting capital: ${BUDGET:.0f} -> {n_accounts} accounts at once")
    print(f"    Eval risk (ERP):  {erp:.0%} of MLL = ${erp * ba.mll_amount:.0f}/trade target risk")
    print(f"    Funded risk (FRP):{frp:.0%} of funded drawdown limit per trade")
    print(f"    Reinvest rule:    deposit every payout into new evals immediately")

# ── Save to git-tracked location so results survive push/pull ─────────────────
import io as _io, shutil as _sh

_LOG_DIR = "sweeps/logs"
os.makedirs(_LOG_DIR, exist_ok=True)

# pkl — full results for programmatic reload
_pkl_dst = os.path.join(_LOG_DIR, "amr_v2_results.pkl")
with open(_pkl_dst, "wb") as _f:
    pickle.dump(p3_results, _f)

# txt — human-readable table (same as stdout above)
_txt_dst = os.path.join(_LOG_DIR, "amr_v2_results.txt")
_buf = _io.StringIO()
_buf.write("ANCHORED MEAN REVERSION V2 SWEEP RESULTS\n")
_buf.write(f"Dataset: {n_days} days (2019+)  All LucidFlex accounts  "
           f"Budget=${BUDGET:.0f}  Horizon={HORIZON}d  Goal=${GOAL:.0f}\n")
_buf.write(f"Phase 2: ERP/FRP 0.1-1.0 x 0.1-1.0 (EV/day) | Phase 3: MC ranked by min P($0)\n\n")
_buf.write(f"{'#':>3}  {'Config':<72}  {'Acct':>5}  {'N':>5}  {'tpd':>5}  {'WR':>6}  {'avgR':>7}  "
           f"{'Sortino':>8}  {'P($0)':>7}  {'P(>bgt)':>8}  {'P(goal)':>8}  "
           f"{'med$':>7}  {'EV/d(med)':>10}  {'ERP':>5}  {'FRP':>5}\n")
_buf.write("-"*190 + "\n")
for rank, r in enumerate(p3_results, 1):
    label = _make_label(r["params"])
    _buf.write(f"{rank:>3}  {label:<72}  {r.get('best_account','25K'):>5}  "
               f"{r['n_t']:>5}  {r['tpd']:>5.2f}  "
               f"{r['wr']:>6.1%}  {r['avgR']:>+7.4f}  {r['sortino']:>8.3f}  "
               f"{r['p_zero']:>7.1%}  {r['p_profit']:>8.1%}  {r['p_goal']:>8.1%}  "
               f"{r['median_cash']:>7.0f}  {r['best_ev_med']:>+10.2f}  "
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
