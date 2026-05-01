"""
Monte Carlo distribution: 4x correlated 25K accounts over 84 trading days.
Answers: what is P(reaching $10K) vs just showing the EV?
"""
import numpy as np
import pandas as pd
from datetime import time as dtime

from backtest.data.loader import DataLoader
from backtest.runner.runner import run_backtest
from backtest.runner.config import RunConfig
from backtest.propfirm.lucidflex import LUCIDFLEX_ACCOUNTS, extract_normalised_trades
from strategies.session_mean_rev import SessionMeanRevStrategy
from backtest.data.market_data import MarketData

# ── Load data ──────────────────────────────────────────────────────────────────
loader = DataLoader()
df_1m = pd.read_parquet("data/NQ_1m.parquet")
df_5m = pd.read_parquet("data/NQ_5m.parquet")

start_ts = pd.Timestamp("2022-01-01", tz="America/New_York")
end_ts   = pd.Timestamp("2025-06-01", tz="America/New_York") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
mask_1m  = (df_1m.index >= start_ts) & (df_1m.index <= end_ts)
mask_5m  = (df_5m.index >= start_ts) & (df_5m.index <= end_ts)
df1f = df_1m[mask_1m]; df5f = df_5m[mask_5m]

rthf = (df1f.index.time >= dtime(9, 30)) & (df1f.index.time <= dtime(16, 0))
tdf  = sorted(set(df1f[rthf].index.date))
a1   = {c: df1f[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
a5   = {c: df5f[c].to_numpy(dtype="float64") for c in ["open","high","low","close","volume"]}
bm   = loader._build_bar_map(df1f, df5f)

data = MarketData(
    df_1m=df1f, df_5m=df5f,
    open_1m=a1["open"], high_1m=a1["high"], low_1m=a1["low"],
    close_1m=a1["close"], volume_1m=a1["volume"],
    open_5m=a5["open"], high_5m=a5["high"], low_5m=a5["low"],
    close_5m=a5["close"], volume_5m=a5["volume"],
    bar_map=bm, trading_dates=tdf,
)

# ── Run backtest ───────────────────────────────────────────────────────────────
config = RunConfig(
    starting_capital=100_000, slippage_points=0.25, commission_per_contract=4.50,
    eod_exit_time=dtime(23, 59),
    params={
        "atr_period": 10, "wick_threshold": 0.15, "rr_ratio": 1.5,
        "sl_atr_multiplier": 0.75, "risk_per_trade": 0.01,
        "equity_mode": "dynamic", "starting_equity": 100_000, "point_value": 20.0,
        "require_bos": True, "max_trades_per_day": 3,
        "disp_min_atr_mult": 2.0, "momentum_only": True, "allowed_sessions": ["NY"],
    },
)
result = run_backtest(SessionMeanRevStrategy, config, data)
pnl_pts, sl_dists = extract_normalised_trades(result.trades)

n_pool  = len(pnl_pts)
tpd_avg = n_pool / len(tdf)   # correct: 207/879 = 0.235, NOT estimate_trading_days which divides by 390 (equity) instead of ~1380 (futures)
print(f"{n_pool} trades | {tpd_avg:.4f} trades/trading-day (correct NQ rate)")
print(f"WR: {(pnl_pts > 0).mean():.1%}  "
      f"avg_win: {pnl_pts[pnl_pts>0].mean():.1f}pts  "
      f"avg_loss: {pnl_pts[pnl_pts<0].mean():.1f}pts")

# ── Account params ─────────────────────────────────────────────────────────────
ACCOUNT      = LUCIDFLEX_ACCOUNTS["25K"]
MLL          = ACCOUNT.mll_amount          # $1,000
PT           = ACCOUNT.profit_target       # $1,250
EVAL_RISK    = 0.20 * MLL                  # $200/trade in eval
FUNDED_RISK  = 0.70 * MLL                  # $700/trade in funded
PAYOUT_CAP   = ACCOUNT.payout_cap          # $1,500/payout gross
MIN_PAYOUT   = 500.0
MAX_PAYOUTS  = ACCOUNT.max_payouts         # 6
SPLIT        = ACCOUNT.split               # 0.90
EVAL_FEE     = ACCOUNT.eval_fee            # $70
RESET_FEE    = ACCOUNT.reset_fee           # $60
MICRO_PV     = 2.0                         # MNQ $2/point
TRADING_DAYS = 84                          # 4 calendar months

N_SIMS = 50_000
rng    = np.random.default_rng(42)

# ── Simulate ───────────────────────────────────────────────────────────────────
# trade_mask[sim, day] = True if a trade fires that day
trade_mask = rng.random((N_SIMS, TRADING_DAYS)) < tpd_avg
t_idx      = rng.integers(0, n_pool, size=(N_SIMS, TRADING_DAYS))
t_pnl_pts  = pnl_pts[t_idx]
t_sl_dists = sl_dists[t_idx]

SB = ACCOUNT.starting_balance  # $25,000

balance           = np.full(N_SIMS, SB, dtype=np.float64)
mll_level         = np.full(N_SIMS, SB - MLL, dtype=np.float64)
phase             = np.zeros(N_SIMS, dtype=np.int32)   # 0=eval, 1=funded
total_withdrawn   = np.zeros(N_SIMS, dtype=np.float64)
payouts_taken     = np.zeros(N_SIMS, dtype=np.int32)
funded_start_bal  = np.zeros(N_SIMS, dtype=np.float64)
mll_locked        = np.zeros(N_SIMS, dtype=np.bool_)
prof_days         = np.zeros(N_SIMS, dtype=np.int32)
total_cost        = np.full(N_SIMS, EVAL_FEE, dtype=np.float64)

for day in range(TRADING_DAYS):
    has_trade = trade_mask[:, day]
    ppts      = t_pnl_pts[:, day]
    sldist    = np.maximum(t_sl_dists[:, day], 0.01)

    contracts_eval   = np.clip(np.floor(EVAL_RISK   / (sldist * MICRO_PV)), 1, ACCOUNT.max_micros).astype(np.int32)
    contracts_funded = np.clip(np.floor(FUNDED_RISK / (sldist * MICRO_PV)), 1, ACCOUNT.max_micros).astype(np.int32)
    contracts        = np.where(phase == 0, contracts_eval, contracts_funded)
    dollar_pnl       = ppts * contracts * MICRO_PV

    apply_mask = has_trade  # alive is always true (we model resets, not death)
    balance    = np.where(apply_mask, balance + dollar_pnl, balance)

    # EOD trailing MLL rises when balance hits new high
    new_high  = balance > (mll_level + MLL)
    mll_level = np.where(new_high, balance - MLL, mll_level)

    # Lock MLL at SB-100 once funded balance crosses starting funded balance
    lock_cond  = (phase == 1) & (~mll_locked) & (balance >= funded_start_bal)
    mll_level  = np.where(lock_cond, funded_start_bal - 100.0, mll_level)
    mll_locked = mll_locked | lock_cond

    # MLL breach -> reset (pay reset fee, restart eval)
    blew_up    = balance <= mll_level
    balance    = np.where(blew_up, SB,           balance)
    mll_level  = np.where(blew_up, SB - MLL,     mll_level)
    phase      = np.where(blew_up, np.int32(0),  phase)
    payouts_taken     = np.where(blew_up, np.int32(0), payouts_taken)
    funded_start_bal  = np.where(blew_up, 0.0,   funded_start_bal)
    mll_locked        = np.where(blew_up, False,  mll_locked)
    prof_days         = np.where(blew_up, np.int32(0), prof_days)
    total_cost        = np.where(blew_up, total_cost + RESET_FEE, total_cost)

    # Eval -> Funded
    pass_eval = (phase == 0) & (balance >= SB + PT)
    phase     = np.where(pass_eval, np.int32(1), phase)
    funded_start_bal = np.where(pass_eval, balance, funded_start_bal)

    # Profitable day counter (funded, trade happened, positive pnl)
    is_funded_win = (phase == 1) & apply_mask & (dollar_pnl > 0)
    prof_days     = np.where(phase == 1, prof_days + is_funded_win.astype(np.int32), prof_days)

    # Payout check
    gross_profit  = np.maximum(0.0, balance - funded_start_bal)
    gross_pay     = np.minimum(gross_profit * 0.5, PAYOUT_CAP)
    can_pay       = (phase == 1) & (prof_days >= 5) & (payouts_taken < MAX_PAYOUTS) & (gross_pay >= MIN_PAYOUT)
    net_pay       = gross_pay * SPLIT
    total_withdrawn  = np.where(can_pay, total_withdrawn + net_pay, total_withdrawn)
    balance          = np.where(can_pay, balance - gross_pay, balance)
    funded_start_bal = np.where(can_pay, balance, funded_start_bal)
    payouts_taken    = np.where(can_pay, payouts_taken + 1, payouts_taken)
    prof_days        = np.where(can_pay, np.int32(0), prof_days)

# ── Results ────────────────────────────────────────────────────────────────────
net_1acct         = total_withdrawn - total_cost
net_4acct_corr    = net_1acct * 4   # perfectly correlated: same wins/losses on all 4

print()
print("=" * 60)
print("SINGLE 25K ACCOUNT — 84 trading day distribution")
print("=" * 60)
print(f"  Mean net:   ${net_1acct.mean():>8,.0f}")
print(f"  Median net: ${np.median(net_1acct):>8,.0f}")
print(f"  P(net > 0):      {(net_1acct > 0).mean():.1%}")
print(f"  P(net >= $2500): {(net_1acct >= 2500).mean():.1%}")

print()
print("=" * 60)
print("4x 25K ACCOUNTS (PERFECTLY CORRELATED copy trading)")
print("84 trading days = 4 calendar months")
print("=" * 60)
print(f"  Mean net total:   ${net_4acct_corr.mean():>8,.0f}")
print(f"  Median net total: ${np.median(net_4acct_corr):>8,.0f}")
print()
targets = [0, 2000, 5000, 8000, 10000, 12000, 15000]
print("  Probability of reaching target:")
for t in targets:
    p = (net_4acct_corr >= t).mean()
    print(f"    >= ${t:>6,}: {p:>6.1%}")
print()
print("  Percentile distribution:")
for pct in [5, 10, 25, 50, 75, 90, 95]:
    val = np.percentile(net_4acct_corr, pct)
    print(f"    p{pct:2d}: ${val:>8,.0f}")

print()
print("=" * 60)
print("EXTENDED HORIZON — 4x 25K accounts (correlated)")
print("=" * 60)
# What multiple of 84-day EV is needed for $10K?
mean_84 = net_4acct_corr.mean()
days_to_10k_at_mean = 10000 / (mean_84 / 84) if mean_84 > 0 else float("inf")
print(f"  Mean net in 84 td: ${mean_84:,.0f}")
print(f"  Days at mean rate to $10K: {days_to_10k_at_mean:.0f} trading days "
      f"({days_to_10k_at_mean/21:.1f} calendar months)")
