import sys, time, traceback
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from datetime import time as dtime
from backtest.data.loader import DataLoader
from backtest.data.market_data import MarketData
from backtest.runner.config import RunConfig
from backtest.runner.runner import run_backtest

def load_data():
    loader = DataLoader()
    df_1m = pd.read_parquet('data/NQ_1m.parquet')
    df_5m = pd.read_parquet('data/NQ_5m.parquet')
    start = pd.Timestamp('2022-01-01', tz='America/New_York')
    end   = pd.Timestamp('2022-12-31 23:59:59', tz='America/New_York')
    m1 = df_1m[(df_1m.index >= start) & (df_1m.index <= end)]
    m5 = df_5m[(df_5m.index >= start) & (df_5m.index <= end)]
    rth = (m1.index.time >= dtime(9,30)) & (m1.index.time <= dtime(16,0))
    td = sorted(set(m1[rth].index.date))
    a1 = {c: m1[c].to_numpy('float64') for c in ['open','high','low','close','volume']}
    a5 = {c: m5[c].to_numpy('float64') for c in ['open','high','low','close','volume']}
    bm = loader._build_bar_map(m1, m5)
    return MarketData(df_1m=m1, df_5m=m5,
        open_1m=a1['open'], high_1m=a1['high'], low_1m=a1['low'],
        close_1m=a1['close'], volume_1m=a1['volume'],
        open_5m=a5['open'], high_5m=a5['high'], low_5m=a5['low'],
        close_5m=a5['close'], volume_5m=a5['volume'],
        bar_map=bm, trading_dates=td)

strategies = [
    ('MACD', 'strategies.macd_strategy', 'MACDStrategy', {}),
    ('ParabolicSAR', 'strategies.parabolic_sar_strategy', 'ParabolicSARStrategy', {}),
    ('RSI', 'strategies.rsi_strategy', 'RSIStrategy', {}),
    ('DualThrust', 'strategies.dual_thrust_strategy', 'DualThrustStrategy', {}),
    ('LondonBreakout', 'strategies.london_breakout_strategy', 'LondonBreakoutStrategy', {}),
    ('HeikinAshi', 'strategies.heikin_ashi_strategy', 'HeikinAshiStrategy', {}),
    ('AwesomeOscillator', 'strategies.awesome_oscillator_strategy', 'AwesomeOscillatorStrategy', {}),
    ('ShootingStar', 'strategies.shooting_star_strategy', 'ShootingStarStrategy', {}),
    ('BollingerBands', 'strategies.bollinger_bands_strategy', 'BollingerBandsStrategy', {}),
]

print('Loading data...')
data = load_data()
cfg = RunConfig(starting_capital=100_000, slippage_points=0.25,
                commission_per_contract=4.50, eod_exit_time=dtime(23,59), params={})

for name, module, cls_name, params in strategies:
    try:
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        cfg.params = params
        t0 = time.perf_counter()
        result = run_backtest(cls, cfg, data, validate=False)
        elapsed = time.perf_counter() - t0
        n = len(result.trades)
        if n > 0:
            wr = sum(1 for t in result.trades if t.net_pnl_dollars > 0) / n
            pnl = sum(t.net_pnl_dollars for t in result.trades)
            print(f'[OK] {name}: {n} trades, WR={wr:.1%}, PnL=${pnl:,.0f}  [{elapsed:.1f}s]')
        else:
            print(f'[OK] {name}: 0 trades  [{elapsed:.1f}s]')
    except Exception as e:
        print(f'[FAIL] {name}: {e}')
        traceback.print_exc()
