"""Per-instrument economics (the single source of truth).

The backtest engine historically hard-coded NQ economics (POINT_VALUE=20,
TICK_SIZE=0.25). For a multi-instrument portfolio each instrument must use its
own tick size, slippage, and point value. This table is the only place those
numbers live; the cell backtest injects them into RunConfig + strategy params,
and the propfirm sim reads point values from here for ES/YM cells.

Commission is identical across all three families (confirmed from the broker
schedule): $1.75/side mini, $0.50/side micro — encoded as MINI_COMM_RT/
MICRO_COMM_RT in lucidflex.py, so it is NOT duplicated here.

slippage_points is set to 2 ticks/side per instrument (generalises the
pre-registered 0.5pt = 2 NQ ticks). For market entries the runner bakes one
side into the fill price and the cell R subtracts the other; see backtest_cell.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpec:
    symbol:           str
    tick_size:        float
    slippage_points:  float   # 2 ticks/side
    mini_point_value:  float  # full-size contract $/pt (NQ 20, ES 50, YM 5)
    micro_point_value: float  # micro contract $/pt (MNQ 2, MES 5, MYM 0.5)


INSTRUMENTS: dict[str, InstrumentSpec] = {
    'NQ': InstrumentSpec('NQ', 0.25, 0.5, 20.0, 2.0),
    'ES': InstrumentSpec('ES', 0.25, 0.5, 50.0, 5.0),
    'YM': InstrumentSpec('YM', 1.00, 2.0,  5.0, 0.5),
}


def spec(symbol: str) -> InstrumentSpec:
    try:
        return INSTRUMENTS[symbol]
    except KeyError:
        raise KeyError(f"No InstrumentSpec for {symbol!r}; known: {list(INSTRUMENTS)}")
