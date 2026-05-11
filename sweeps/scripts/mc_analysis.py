"""
Fast vectorized reinvestment MC.

The user's reinvest_mc does:
1. Start with budget=$300. Open as many eval accounts as budget allows at $70/ea.
2. Each account runs eval phase until pass or bust.
3. On pass → funded phase, collects payouts until horizon day 84.
4. Payouts received → reinvest in more eval accounts starting that day.
5. Final outcome = total withdrawn.

Key simplification: at $70/eval and $300 budget, we start with floor(300/70)=4 accounts.
With high-ev configs, accounts pass quickly, so reinvestment kicks in early.

Strategy: simulate each account independently (eval + funded), collect (start_day, payout_day, net)
tuples. Then run the reinvestment logic over those.

To speed up: vectorize across n_sims using numpy batch operations.
We simulate eval and funded phases for a large pool of accounts, then replay the reinvest logic.
"""
import pickle, time
import numpy as np

with open('tmp_orb_results.pkl', 'rb') as f:
    orb_results = pickle.load(f)
with open('tmp_smr_results.pkl', 'rb') as f:
    smr_results = pickle.load(f)
with open('md_cache.pkl', 'rb') as f:
    cache = pickle.load(f)
md = cache['md']
n_days = cache['n_days']

from backtest.runner.runner import run_backtest
from backtest.runner.config import RunConfig
from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS, extract_normalised_trades
from strategies.profitable.orb_atr_breakout_strategy import ORBATRBreakoutStrategy
from strategies.profitable.session_mean_rev import SessionMeanRevStrategy

account = LUCIDFLEX_ACCOUNTS['25K']
MICRO_PV = 2.0

# ---------------------------------------------------------------------------
# Fast vectorized account simulator
# ---------------------------------------------------------------------------

def sim_account_batch(pnl_pts, sl_dists, tpd, account, scheme, eval_risk, fund_risk,
                      n_accounts, start_day, horizon, rng):
    """
    Simulate n_accounts accounts in parallel starting at start_day.
    Returns list of payout events: [(abs_day, net_cash), ...]  for each account.
    Uses day-level vectorization: all n_accounts processed per day simultaneously.
    """
    sb = account.starting_balance
    mll_amt = account.mll_amount
    profit_target = account.profit_target
    max_micros = account.max_micros
    payout_cap = account.payout_cap
    split = account.split

    base_eval = eval_risk * mll_amt
    base_fund = fund_risk * mll_amt
    base_frac_eval = base_eval / sb
    base_frac_fund = base_fund / sb
    n_pool = len(pnl_pts)

    # ── EVAL phase ────────────────────────────────────────────────────────────
    balance    = np.full(n_accounts, sb)
    mll_level  = np.full(n_accounts, sb - mll_amt)
    peak_eod   = np.full(n_accounts, sb)
    total_pnl  = np.zeros(n_accounts)
    max_day_pnl= np.zeros(n_accounts)
    n_prof_days= np.zeros(n_accounts, dtype=np.int32)
    alive      = np.ones(n_accounts, dtype=bool)   # not busted
    passed     = np.zeros(n_accounts, dtype=bool)
    pass_day   = np.zeros(n_accounts, dtype=np.int32)

    max_eval_days = min(300, horizon + 50 - start_day)

    for d in range(max_eval_days):
        abs_day = start_day + d
        if not alive.any():
            break

        mask = alive & ~passed
        if not mask.any():
            break

        # Draw trade counts for active accounts
        n_t_arr = rng.poisson(tpd, size=n_accounts)
        n_t_arr = np.where(mask, n_t_arr, 0)
        max_trades = int(n_t_arr.max()) if mask.any() else 0

        day_pnl = np.zeros(n_accounts)
        for _ in range(max_trades):
            do_trade = mask & (n_t_arr > 0)
            n_t_arr = np.where(do_trade, n_t_arr - 1, n_t_arr)
            if not do_trade.any():
                break
            idx = rng.integers(0, n_pool, size=n_accounts)
            raw = pnl_pts[idx]
            sl_d = sl_dists[idx]

            if scheme == 'pct_balance':
                tr = balance * base_frac_eval
            else:
                tr = np.full(n_accounts, base_eval)

            nc = np.clip(np.floor(tr / (np.maximum(sl_d, 0.01) * MICRO_PV)), 1, max_micros).astype(np.int32)
            dp = raw * nc * MICRO_PV
            day_pnl = np.where(do_trade, day_pnl + dp, day_pnl)

        balance = np.where(mask, balance + day_pnl, balance)

        # Check busts
        busted = mask & (balance <= mll_level)
        alive = alive & ~busted

        # Update profit tracking for alive accounts
        upd = mask & alive
        profitable_day = upd & (day_pnl > 0)
        n_prof_days = np.where(profitable_day, n_prof_days + 1, n_prof_days)
        total_pnl = np.where(upd & (day_pnl > 0), total_pnl + day_pnl, total_pnl)
        max_day_pnl = np.where(upd & (day_pnl > max_day_pnl), day_pnl, max_day_pnl)

        # Update peak / trailing MLL
        new_peak = upd & (balance > peak_eod)
        peak_eod = np.where(new_peak, balance, peak_eod)
        mll_level = np.where(new_peak, np.minimum(mll_level, peak_eod - mll_amt), mll_level)

        # Check pass conditions
        consistency_ok = (n_prof_days >= 2) & ((total_pnl == 0) | (max_day_pnl / np.maximum(total_pnl, 1e-9) <= 0.5))
        just_passed = upd & (total_pnl >= profit_target) & consistency_ok & ~passed
        passed = passed | just_passed
        pass_day = np.where(just_passed, abs_day + 1, pass_day)

    # ── FUNDED phase ─────────────────────────────────────────────────────────
    # Only run for passed accounts
    funded_mask = passed
    all_payouts = [[] for _ in range(n_accounts)]

    if not funded_mask.any():
        return all_payouts

    f_balance   = np.where(funded_mask, sb, 0.0)
    f_mll_level = np.where(funded_mask, sb - mll_amt, 0.0)
    f_peak_eod  = np.where(funded_mask, sb, 0.0)
    f_mll_locked= np.zeros(n_accounts, dtype=bool)
    f_payout_days_cycle = np.zeros(n_accounts, dtype=np.int32)
    f_n_payout  = np.zeros(n_accounts, dtype=np.int32)
    f_alive     = funded_mask.copy()

    # Use max pass day across funded accounts to determine start
    max_pass = int(pass_day[funded_mask].max()) if funded_mask.any() else horizon + 1

    for abs_day in range(max_pass, horizon + 1):
        # Account active if: funded & alive & past its pass_day & n_payout < max
        mask_f = f_alive & funded_mask & (pass_day <= abs_day) & (f_n_payout < account.max_payouts)
        if not mask_f.any():
            break

        n_t_arr = rng.poisson(tpd, size=n_accounts)
        n_t_arr = np.where(mask_f, n_t_arr, 0)
        max_trades = int(n_t_arr.max()) if mask_f.any() else 0

        day_pnl = np.zeros(n_accounts)
        for _ in range(max_trades):
            do_trade = mask_f & (n_t_arr > 0)
            n_t_arr = np.where(do_trade, n_t_arr - 1, n_t_arr)
            if not do_trade.any():
                break
            idx = rng.integers(0, n_pool, size=n_accounts)
            raw = pnl_pts[idx]
            sl_d = sl_dists[idx]

            if scheme == 'pct_balance':
                tr = f_balance * base_frac_fund
            else:
                tr = np.full(n_accounts, base_fund)

            nc = np.clip(np.floor(tr / (np.maximum(sl_d, 0.01) * MICRO_PV)), 1, max_micros).astype(np.int32)
            dp = raw * nc * MICRO_PV
            day_pnl = np.where(do_trade, day_pnl + dp, day_pnl)

        f_balance = np.where(mask_f, f_balance + day_pnl, f_balance)

        # Bust check
        busted_f = mask_f & (f_balance <= f_mll_level)
        f_alive = f_alive & ~busted_f

        # MLL lock
        lock_now = mask_f & ~f_mll_locked & (f_balance >= sb)
        f_mll_locked = f_mll_locked | lock_now
        f_mll_level = np.where(lock_now, sb - 100.0, f_mll_level)

        # Peak tracking
        new_peak_f = mask_f & (f_balance > f_peak_eod)
        f_peak_eod = np.where(new_peak_f, f_balance, f_peak_eod)
        f_mll_level = np.where(new_peak_f & ~f_mll_locked,
                                np.minimum(f_mll_level, f_peak_eod - mll_amt),
                                f_mll_level)

        # Payout cycle
        profitable_day_f = mask_f & f_alive & (day_pnl > 0)
        f_payout_days_cycle = np.where(profitable_day_f, f_payout_days_cycle + 1, f_payout_days_cycle)

        # Trigger payout where cycle >= 5
        payout_trigger = mask_f & f_alive & (f_payout_days_cycle >= 5)
        if payout_trigger.any():
            profits = np.maximum(0.0, f_balance - sb)
            gross = np.minimum(0.5 * profits, payout_cap)
            net_pay = gross * split
            # Only pay if gross >= 500
            do_pay = payout_trigger & (gross >= 500.0)
            if do_pay.any():
                acct_indices = np.where(do_pay)[0]
                for ai in acct_indices:
                    all_payouts[ai].append((abs_day, float(net_pay[ai])))
                f_balance = np.where(do_pay, f_balance - gross, f_balance)
                f_n_payout = np.where(do_pay, f_n_payout + 1, f_n_payout)
                f_payout_days_cycle = np.where(do_pay, 0, f_payout_days_cycle)

    return all_payouts


def reinvest_mc_fast(pnl_pts, sl_dists, tpd, account, scheme, eval_risk, fund_risk,
                     budget=300.0, horizon=84, n_sims=5000, seed=42, batch_size=500):
    """
    Fast vectorized reinvestment MC.
    Simulates accounts in batches of batch_size across all sims.
    """
    eval_fee = account.eval_fee
    rng = np.random.default_rng(seed)
    outcomes = np.zeros(n_sims)
    n_pool = len(pnl_pts)

    # Pre-simulate a large pool of accounts starting at day 0
    # For reinvestment, we need accounts starting at various days
    # Key insight: with ~0.8 trades/day, and the budget allowing 4 accounts initially,
    # the reinvestment tree is small. We pre-sim a pool and sample from it.

    # Strategy: simulate a large pool of accounts starting at each day 0..horizon
    # Then for each sim, replay the reinvestment logic using the pool.

    # Pre-generate account outcomes: (pass_day, payouts_list)
    # For each starting day in [0..84], we need a pool of pre-simmed accounts.

    # Simplification: since tpd ~1, each account generates ~84 trades in the funded phase.
    # We pre-sim MAX_POOL accounts per starting day cluster.
    MAX_POOL = max(n_sims * 10, 20000)  # large enough pool

    print(f"  Pre-simulating {MAX_POOL} accounts...", flush=True)
    t0 = time.time()

    # Simulate MAX_POOL accounts starting at day 0
    batch = min(MAX_POOL, 5000)
    all_acct_results = []  # list of (payout_events_list,) where payout_events = [(day, net)]

    for b_start in range(0, MAX_POOL, batch):
        b_end = min(b_start + batch, MAX_POOL)
        n_b = b_end - b_start
        payouts_list = sim_account_batch(
            pnl_pts, sl_dists, tpd, account, scheme, eval_risk, fund_risk,
            n_accounts=n_b, start_day=0, horizon=horizon + 50, rng=rng
        )
        all_acct_results.extend(payouts_list)
        if b_start == 0:
            elapsed = time.time() - t0
            est_total = elapsed * (MAX_POOL / n_b)
            print(f"    First batch {n_b} done in {elapsed:.1f}s, est total {est_total:.0f}s", flush=True)

    print(f"  Pre-sim done in {time.time()-t0:.1f}s. Pool size: {len(all_acct_results)}", flush=True)

    # For each sim, replay reinvest logic using pool accounts
    # We draw random accounts from pool for each eval attempt
    pool = all_acct_results  # list of payout event lists
    n_pool_accts = len(pool)

    t1 = time.time()
    rng2 = np.random.default_rng(seed + 1)

    for sim_i in range(n_sims):
        cash = budget
        total_withdrawn = 0.0
        pending = []  # (abs_day, net) sorted

        def open_acct(start_d):
            nonlocal cash
            if cash < eval_fee:
                return
            cash -= eval_fee
            # Draw a random pool account and shift its payout days
            ai = int(rng2.integers(0, n_pool_accts))
            acct_pays = pool[ai]
            for (d, net) in acct_pays:
                shifted_d = start_d + d
                if shifted_d <= horizon:
                    pending.append((shifted_d, net))
            pending.sort()

        # Open initial accounts
        while cash >= eval_fee:
            open_acct(0)

        # Process payout events
        ei = 0
        while ei < len(pending):
            d, net = pending[ei]
            if d > horizon:
                break
            cash += net
            total_withdrawn += net
            while cash >= eval_fee:
                open_acct(d)
                pending[ei+1:] = sorted(pending[ei+1:])
            ei += 1

        outcomes[sim_i] = total_withdrawn

    print(f"  Reinvest replay done in {time.time()-t1:.1f}s", flush=True)
    return outcomes


# ---------------------------------------------------------------------------
# Identify top configs and run MC
# ---------------------------------------------------------------------------

all_results = orb_results + smr_results
all_results_sorted = sorted(all_results, key=lambda x: x['ev_per_day'], reverse=True)

print("All configs sorted by ev/day:")
print(f"{'Strategy':15} {'rr':5} {'atr_m':7} {'mtpd':5} {'n':6} {'tpd':6} {'WR':7} {'avgR':7} {'ev/day':8}")
seen_keys = set()
unique_sorted = []
for r in all_results_sorted:
    key = (r['strategy'], r.get('rr'), r.get('atr_m'))
    if key not in seen_keys:
        seen_keys.add(key)
        unique_sorted.append(r)
        print(f"{r['strategy']:15} {str(r.get('rr','?')):5} {str(r.get('atr_m','?')):7} {str(r.get('mtpd','?')):5} "
              f"{r['n_trades']:6} {r['tpd']:6.3f} {r['wr']:7.1%} {r['avgR']:7.4f} ${r['ev_per_day']:7.2f}")

top3 = unique_sorted[:3]
print("\nTop 3 for reinvestment MC:")
for r in top3:
    print(f"  {r['strategy']} rr={r.get('rr')} atr_m={r.get('atr_m')} ev/day=${r['ev_per_day']:.2f}")

# ── Run MC ────────────────────────────────────────────────────────────────────
mc_results = []

def run_config_mc(r, label=None):
    strat_name = r['strategy']
    lbl = label or strat_name
    print(f"\n{'='*60}\nRunning MC: {lbl}", flush=True)

    if strat_name == 'ORB':
        params = {
            'rr_ratio': r['rr'],
            'atr_space_multiplier': r['atr_m'],
            'max_trades_per_day': r['mtpd'],
            'sl_pct': 0.005,
            'use_200sma_filter': False,
            'risk_per_trade': 0.01,
            'starting_equity': 100_000,
            'equity_mode': 'dynamic',
        }
        cfg = RunConfig(starting_capital=100_000, slippage_points=0.25,
                        commission_per_contract=4.50, track_equity_curve=False, params=params)
        result = run_backtest(ORBATRBreakoutStrategy, cfg, data=md, validate=False)
    else:
        # SMR
        disp = r.get('atr_m', 0.0)
        smr_params = {
            'allowed_sessions': ['NY'],
            'require_bos': True,
            'momentum_only': True,
            'rr_ratio': r.get('rr', 0.5),
            'sl_atr_multiplier': 1.0,
            'wick_threshold': 0.15,
            'atr_period': 10,
            'disp_min_atr_mult': disp if disp is not None else 0.0,
        }
        cfg = RunConfig(starting_capital=100_000, slippage_points=0.25,
                        commission_per_contract=4.50, track_equity_curve=False, params=smr_params)
        result = run_backtest(SessionMeanRevStrategy, cfg, data=md, validate=False)

    pnl_pts, sl_dists = extract_normalised_trades(result.trades)
    tpd = r['tpd']
    scheme = r['scheme']
    erp = float(r['erp'])
    frp = float(r['frp'])

    t0 = time.time()
    outcomes = reinvest_mc_fast(
        pnl_pts, sl_dists, tpd, account, scheme, erp, frp,
        budget=300.0, horizon=84, n_sims=5000, seed=42, batch_size=2000
    )
    elapsed = time.time() - t0
    print(f"  Total MC time: {elapsed:.1f}s", flush=True)

    p = np.percentile(outcomes, [10, 25, 50, 75, 90, 95])
    mean = np.mean(outcomes)
    p_10k = np.mean(outcomes > 10000)
    p_5k = np.mean(outcomes > 5000)
    p_0 = np.mean(outcomes == 0)

    print(f"  RESULTS: P10=${p[0]:.0f} P25=${p[1]:.0f} P50=${p[2]:.0f} P75=${p[3]:.0f} P90=${p[4]:.0f} P95=${p[5]:.0f}")
    print(f"  Mean=${mean:.0f}  P(>$10K)={p_10k:.1%}  P(>$5K)={p_5k:.1%}  P($0)={p_0:.1%}")

    return {
        'label': lbl,
        'strategy': strat_name,
        'params': f"rr={r.get('rr')} atr_m={r.get('atr_m')} mtpd={r.get('mtpd')}",
        'tpd': tpd, 'wr': r['wr'], 'avgR': r['avgR'], 'ev_per_day': r['ev_per_day'],
        'scheme': scheme, 'erp': erp, 'frp': frp,
        'pass_rate': r.get('pass_rate', 0),
        'P10': p[0], 'P25': p[1], 'P50': p[2], 'P75': p[3], 'P90': p[4], 'P95': p[5],
        'mean': mean, 'P_10k': p_10k, 'P_5k': p_5k, 'P_0': p_0,
        'outcomes': outcomes,
    }

# Top 3 unique ORB configs
for r in top3:
    label = f"{r['strategy']}_rr{r.get('rr')}_atrm{r.get('atr_m')}"
    mc_r = run_config_mc(r, label=label)
    mc_results.append(mc_r)

# SMR disp=0
smr_disp0 = smr_results[0]  # disp=0 config
mc_smr0 = run_config_mc(smr_disp0, label='SMR_disp0')
mc_results.append(mc_smr0)

# Save results (without outcomes array to keep pkl small)
mc_save = []
for r in mc_results:
    s = {k: v for k, v in r.items() if k != 'outcomes'}
    mc_save.append(s)

with open('tmp_mc_results.pkl', 'wb') as f:
    pickle.dump(mc_save, f)
print("\nSaved to tmp_mc_results.pkl")
