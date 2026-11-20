# Historical research labels

Strategies now use flat imports such as `strategies.amr_strategy`. The labels below
record the previous folder organization only. They do not establish current
profitability, independent validation, or suitability for live trading. Preserve
experiment dates, costs, and selection history when interpreting past results.

Historical imports such as `strategies.profitable.amr_strategy` remain lazy aliases
of the same modules, including for loading previously serialized class references.
Portfolio discovery reads flat modules and retains the previous tracked module
order. Local files left in former subdirectories remain accessible by their old
imports but are not automatically registered as portfolio cells.

| Module | Former label |
| --- | --- |
| `asia_breakout_strategy` | `no_edge` |
| `ibs_strategy` | `no_edge` |
| `orb_tf_hybrid` | `no_edge` |
| `rsi_strategy` | `no_edge` |
| `turtle_soup_strategy` | `no_edge` |
| `vwap_mean_rev_strategy` | `no_edge` |
| `amr_strategy` | `profitable` |
| `amr_strategy_v1` | `profitable` |
| `ict_smc` | `profitable` |
| `macd_strategy` | `profitable` |
| `orb_atr_breakout_strategy` | `profitable` |
| `orb_delta_breakout` | `profitable` |
| `parabolic_sar_strategy` | `profitable` |
| `session_mean_rev` | `profitable` |
| `trend_following_strategy` | `profitable` |
| `awesome_oscillator_strategy` | `untested` |
| `bollinger_bands_strategy` | `untested` |
| `double_session_sweep` | `untested` |
| `dual_thrust_strategy` | `untested` |
| `dummy` | `untested` |
| `enhanced_orb` | `untested` |
| `gap_fill_strategy` | `untested` |
| `heikin_ashi_strategy` | `untested` |
| `ibs_mean_reversion` | `untested` |
| `intraday_interval_momentum_huang` | `untested` |
| `intraday_momentum_baltussen` | `untested` |
| `intraday_momentum_jin` | `untested` |
| `london_breakout_strategy` | `untested` |
| `noise_boundary_momentum` | `untested` |
| `orb_stop_strategy` | `untested` |
| `overnight_drift` | `untested` |
| `overnight_momentum_rosa` | `untested` |
| `overnight_reversal_cooc` | `untested` |
| `pdh_pdl_sweep_reversal` | `untested` |
| `shooting_star_strategy` | `untested` |
| `vwap_band_reversion` | `untested` |

`amr_spec.md` contains the AMR research specification. `dummy` is a runner
placeholder, and `amr_strategy_v1` is a historical implementation; both remain
excluded from automatic portfolio discovery.
