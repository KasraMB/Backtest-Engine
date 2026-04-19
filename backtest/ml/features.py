"""
Feature definitions for the ICT/SMC ML pipeline.

Signal features are captured inside the strategy at signal time and stored on
Trade.signal_features.  Context features are derived offline by build_dataset()
using the full trade history and market data.  Config features are the
normalised parameter values of the config that generated each trade — added
during multi-config collection so the joint model can learn param × signal
interactions.
"""
from __future__ import annotations

from backtest.ml.configs import CONFIG_FEATURE_NAMES

# ---------------------------------------------------------------------------
# Feature name lists — must stay in sync with extraction logic
# ---------------------------------------------------------------------------

# Captured inside ICTSMCStrategy.generate_signals() at signal time
SIGNAL_FEATURE_NAMES: list[str] = [
    # Setup type (one-hot)
    'is_ote',
    'is_stdv',
    'is_session_ote',
    # Trade direction
    'direction',          # 1 = long, -1 = short
    # Fib value
    'fib_value',          # 0.5-0.79 for OTE, 2.0-4.5 for STDV, 0.5-0.79 for SESSION_OTE
    # Confluence kind (one-hot)
    'conf_ob',
    'conf_bb',
    'conf_fvg',
    'conf_ifvg',
    'conf_rb',
    'conf_session',       # session level (PDH, PDL, Asia_H, …)
    # Confluence timeframe (one-hot)
    'conf_tf_1m',
    'conf_tf_5m',
    'conf_tf_15m',
    'conf_tf_30m',
    # Manipulation leg size relative to ATR (0 for SESSION_OTE which has no leg)
    'manip_leg_size_atr',
    # AFZ geometry (normalised by ATR)
    'zone_range_atr',     # (zone_top - zone_bot) / atr
    'sl_risk_atr',        # |entry - sl| / atr  (risk per trade in ATR units)
    # TP candidate info
    'n_tp_candidates',    # how many STDV extensions meet min_rr
    'tp_r',               # R:R of chosen (first confluent) TP
    'tp_next_r',          # R:R of next candidate, -1 if none
    # Market context
    'atr_pct',            # atr / close_price * 100
    'time_since_open_min',  # minutes since 09:30 ET
    'day_of_week',        # 0=Mon … 4=Fri
    'overnight_range_atr',  # overnight session H-L / atr
    # Session context
    'n_validated_levels', # total levels validated today (before per-type limit)
]

# Computed offline by build_dataset() from the surrounding trade history
CONTEXT_FEATURE_NAMES: list[str] = [
    'daily_trade_idx',        # index of this trade within the day (0=first, 1=second, …)
    'recent_win_rate_10',     # win rate of the 10 trades before this one (0.5 default)
    'recent_expectancy_r_10', # mean R-multiple of the 10 trades before this one
    'consecutive_losses',     # how many consecutive losing trades preceded this one
    'drawdown_pct',           # equity drawdown from peak at signal time (0.0 if unknown)
    'vol_regime_p_high',      # P(today = high-vol regime) via lag-1 Markov HMM on daily range
    'atr_pct_rank',           # percentile rank of today's ATR vs prior 60 days (0=low, 1=high)
    'session_r_so_far',       # cumulative R earned today for this config before this trade
    'days_since_last_win',    # calendar days since last winning trade for this config (capped at 30)
]

ALL_FEATURE_NAMES: list[str] = SIGNAL_FEATURE_NAMES + CONTEXT_FEATURE_NAMES + CONFIG_FEATURE_NAMES

# ---------------------------------------------------------------------------
# Confluence kind → one-hot column mapping
# ---------------------------------------------------------------------------

_CONF_KIND_MAP: dict[str, str] = {
    'OB':  'conf_ob',
    'BB':  'conf_bb',
    'FVG': 'conf_fvg',
    'IFVG': 'conf_ifvg',
    'RB':  'conf_rb',
}

_CONF_TF_MAP: dict[str, str] = {
    '1m':  'conf_tf_1m',
    '5m':  'conf_tf_5m',
    '15m': 'conf_tf_15m',
    '30m': 'conf_tf_30m',
}

# ---------------------------------------------------------------------------
# Signal feature extraction helper (called from the strategy)
# ---------------------------------------------------------------------------

def encode_signal_features(
    fib_type: str,               # 'OTE' | 'STDV' | 'SESSION_OTE'
    direction: int,              # 1 or -1
    fib_value: float,
    confluence_kind: str,        # 'OB', 'BB', 'FVG', 'IFVG', 'RB', or session level name
    confluence_tf: str,          # '1m', '5m', '15m', '30m', or ''
    manip_leg_size_atr: float,   # 0.0 for SESSION_OTE
    zone_top: float,
    zone_bot: float,
    entry: float,
    sl: float,
    atr: float,
    tp_candidates: list,         # list of (price, has_confluence) tuples
    chosen_tp: float,            # price of the TP the strategy will use
    time_since_open_min: int,
    day_of_week: int,
    overnight_range_atr: float,
    n_validated_levels: int,
    close_price: float,
) -> dict:
    """
    Build the signal_features dict from raw strategy values.
    All values are plain Python scalars (int or float) for easy JSON-serialisation.
    """
    feat: dict = {n: 0 for n in SIGNAL_FEATURE_NAMES}

    # Setup type
    feat['is_ote']         = int(fib_type == 'OTE')
    feat['is_stdv']        = int(fib_type == 'STDV')
    feat['is_session_ote'] = int(fib_type == 'SESSION_OTE')

    feat['direction']  = int(direction)
    feat['fib_value']  = float(fib_value)

    # Confluence kind
    col = _CONF_KIND_MAP.get(confluence_kind)
    if col:
        feat[col] = 1
    elif confluence_kind:  # any session level string
        feat['conf_session'] = 1

    # Confluence TF
    tf_col = _CONF_TF_MAP.get(confluence_tf)
    if tf_col:
        feat[tf_col] = 1

    feat['manip_leg_size_atr'] = float(manip_leg_size_atr)

    safe_atr = max(atr, 1e-9)
    feat['zone_range_atr'] = float((zone_top - zone_bot) / safe_atr)
    feat['sl_risk_atr']    = float(abs(entry - sl) / safe_atr)

    # TP candidates
    feat['n_tp_candidates'] = len(tp_candidates)
    risk = abs(entry - sl)
    if risk > 1e-9 and tp_candidates:
        feat['tp_r'] = float(abs(chosen_tp - entry) / risk)
        # Find next candidate after chosen_tp
        remaining = [p for p, _ in tp_candidates if p != chosen_tp]
        if remaining:
            # pick the one closest to chosen_tp (next in the chain)
            nxt = min(remaining, key=lambda p: abs(p - chosen_tp))
            feat['tp_next_r'] = float(abs(nxt - entry) / risk)
        else:
            feat['tp_next_r'] = -1.0
    else:
        feat['tp_r']      = -1.0
        feat['tp_next_r'] = -1.0

    safe_close = max(close_price, 1e-9)
    feat['atr_pct']             = float(atr / safe_close * 100.0)
    feat['time_since_open_min'] = int(time_since_open_min)
    feat['day_of_week']         = int(day_of_week)
    feat['overnight_range_atr'] = float(overnight_range_atr)
    feat['n_validated_levels']  = int(n_validated_levels)

    return feat
