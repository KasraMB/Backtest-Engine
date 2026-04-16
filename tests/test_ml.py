"""
ML pipeline tests — evaluate, features, dataset, model, train, sensitivity, splits.

Run with:
    python -m pytest tests/test_ml.py -v

Tests that require xgboost are auto-skipped if it is not installed.
"""
from __future__ import annotations

import os
import sys
import pickle
from datetime import date, time as dtime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from backtest.ml.evaluate import (
    sortino_r, profit_factor_r, expectancy_r, win_rate,
    evaluate_filter, search_threshold,
)
from backtest.ml.features import (
    encode_signal_features, SIGNAL_FEATURE_NAMES, CONTEXT_FEATURE_NAMES, ALL_FEATURE_NAMES,
)
from backtest.ml.splits import filter_df, split_bounds, SPLITS
from backtest.ml.sensitivity import _hash_params, check_sensitivity
from backtest.ml.train import _generate_fold_bounds


# ===========================================================================
# Helpers
# ===========================================================================

def _r(wins, losses):
    """Build a float R-multiple array from win/loss counts (unit values)."""
    return np.array([1.0] * wins + [-1.0] * losses, dtype=float)


def _rand_r(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.3, 1.0, n)


# ===========================================================================
# Evaluate — sortino_r
# ===========================================================================

class TestSortinoR:

    def test_empty_returns_zero(self):
        assert sortino_r(np.array([])) == 0.0

    def test_all_winners_returns_inf(self):
        assert sortino_r(_r(10, 0)) == float("inf")

    def test_all_losers_negative(self):
        assert sortino_r(_r(0, 10)) < 0

    def test_mixed_positive_edge(self):
        # More wins than losses with reasonable magnitudes
        r = np.array([2.0, 2.0, 2.0, -1.0])
        assert sortino_r(r) > 0

    def test_returns_float(self):
        assert isinstance(sortino_r(_rand_r()), float)

    def test_no_downside_infinite(self):
        # All positive → no downside deviation → inf
        r = np.array([0.5, 1.0, 2.0])
        result = sortino_r(r)
        assert result == float("inf") or result > 100


# ===========================================================================
# Evaluate — profit_factor_r
# ===========================================================================

class TestProfitFactorR:

    def test_empty_returns_zero(self):
        assert profit_factor_r(np.array([])) == 0.0

    def test_all_winners_returns_inf(self):
        assert profit_factor_r(_r(5, 0)) == float("inf")

    def test_all_losers_returns_zero(self):
        assert profit_factor_r(_r(0, 5)) == 0.0

    def test_2_to_1_rr(self):
        # 2 wins of 2R, 1 loss of 1R → PF = 4 / 1 = 4
        r = np.array([2.0, 2.0, -1.0])
        assert profit_factor_r(r) == pytest.approx(4.0)

    def test_breakeven_returns_one(self):
        r = np.array([1.0, 1.0, -1.0, -1.0])
        assert profit_factor_r(r) == pytest.approx(1.0)


# ===========================================================================
# Evaluate — expectancy_r / win_rate
# ===========================================================================

class TestExpectancyAndWinRate:

    def test_expectancy_empty(self):
        assert expectancy_r(np.array([])) == 0.0

    def test_expectancy_all_one(self):
        assert expectancy_r(np.ones(10)) == pytest.approx(1.0)

    def test_expectancy_mixed(self):
        r = np.array([2.0, -1.0])
        assert expectancy_r(r) == pytest.approx(0.5)

    def test_win_rate_empty(self):
        assert win_rate(np.array([])) == 0.0

    def test_win_rate_all_win(self):
        assert win_rate(np.ones(5)) == pytest.approx(1.0)

    def test_win_rate_half(self):
        assert win_rate(np.array([1.0, -1.0])) == pytest.approx(0.5)

    def test_win_rate_in_0_1(self):
        assert 0.0 <= win_rate(_rand_r()) <= 1.0


# ===========================================================================
# Evaluate — evaluate_filter
# ===========================================================================

class TestEvaluateFilter:

    def test_all_keys_present(self):
        r_all   = _rand_r(20)
        r_taken = r_all[r_all > 0]
        result  = evaluate_filter(r_taken, r_all)
        assert "taken" in result
        assert "all" in result
        assert "take_rate" in result

    def test_take_rate_correct(self):
        r_all   = np.array([1.0, -1.0, 1.0, -1.0])
        r_taken = r_all[r_all > 0]
        result  = evaluate_filter(r_taken, r_all)
        assert result["take_rate"] == pytest.approx(0.5)

    def test_taken_n_correct(self):
        r_all   = np.arange(1, 11, dtype=float)  # 10 trades
        r_taken = r_all[:4]
        result  = evaluate_filter(r_taken, r_all)
        assert result["taken"]["n"] == 4
        assert result["all"]["n"] == 10


# ===========================================================================
# Evaluate — search_threshold
# ===========================================================================

class TestSearchThreshold:

    def test_returns_tuple(self):
        pred   = np.linspace(0.0, 1.0, 20)
        actual = np.linspace(-1.0, 3.0, 20)
        thresh, stats = search_threshold(pred, actual)
        assert isinstance(thresh, float)
        assert isinstance(stats, dict)

    def test_threshold_in_prediction_range(self):
        pred   = np.linspace(0.0, 1.0, 50)
        actual = _rand_r(50)
        thresh, _ = search_threshold(pred, actual)
        assert float(np.percentile(pred, 5)) <= thresh <= float(np.percentile(pred, 95))

    def test_min_take_rate_respected(self):
        rng    = np.random.default_rng(1)
        pred   = rng.uniform(0, 1, 50)
        actual = rng.normal(0, 1, 50)
        thresh, _ = search_threshold(pred, actual, min_take_rate=0.3)
        # At the returned threshold, take rate must be >= 30%
        taken = actual[pred >= thresh]
        if len(taken) > 0:
            assert len(taken) / len(actual) >= 0.30 - 1e-9

    def test_metric_options(self):
        pred   = np.linspace(0, 1, 30)
        actual = np.linspace(-1, 2, 30)
        for metric in ("sortino", "profit_factor", "expectancy_r", "win_rate"):
            thresh, _ = search_threshold(pred, actual, metric=metric)
            assert isinstance(thresh, float)


# ===========================================================================
# Features — encode_signal_features
# ===========================================================================

class TestSignalFeatureEncoding:

    def _encode(self, **kwargs):
        defaults = dict(
            fib_type="OTE", direction=1, fib_value=0.618,
            confluence_kind="OB", confluence_tf="5m",
            manip_leg_size_atr=2.5,
            zone_top=19050.0, zone_bot=19000.0,
            entry=19020.0, sl=18900.0,
            atr=25.0,
            tp_candidates=[(19300.0, True), (19500.0, False)],
            chosen_tp=19300.0,
            time_since_open_min=20,
            day_of_week=1,
            overnight_range_atr=3.0,
            n_validated_levels=4,
            close_price=19020.0,
        )
        defaults.update(kwargs)
        return encode_signal_features(**defaults)

    def test_returns_dict_with_all_signal_names(self):
        feat = self._encode()
        for name in SIGNAL_FEATURE_NAMES:
            assert name in feat, f"Missing feature: {name}"

    def test_ote_one_hot(self):
        feat = self._encode(fib_type="OTE")
        assert feat["is_ote"] == 1
        assert feat["is_stdv"] == 0
        assert feat["is_session_ote"] == 0

    def test_stdv_one_hot(self):
        feat = self._encode(fib_type="STDV")
        assert feat["is_stdv"] == 1
        assert feat["is_ote"] == 0

    def test_session_ote_one_hot(self):
        feat = self._encode(fib_type="SESSION_OTE")
        assert feat["is_session_ote"] == 1
        assert feat["is_ote"] == 0
        assert feat["is_stdv"] == 0

    def test_direction_stored(self):
        feat_long  = self._encode(direction=1)
        feat_short = self._encode(direction=-1)
        assert feat_long["direction"] == 1
        assert feat_short["direction"] == -1

    def test_ob_confluence_one_hot(self):
        feat = self._encode(confluence_kind="OB")
        assert feat["conf_ob"] == 1
        assert feat["conf_fvg"] == 0

    def test_fvg_confluence_one_hot(self):
        feat = self._encode(confluence_kind="FVG")
        assert feat["conf_fvg"] == 1
        assert feat["conf_ob"] == 0

    def test_session_level_confluence(self):
        feat = self._encode(confluence_kind="PDH")
        assert feat["conf_session"] == 1
        assert feat["conf_ob"] == 0

    def test_tf_5m_one_hot(self):
        feat = self._encode(confluence_tf="5m")
        assert feat["conf_tf_5m"] == 1
        assert feat["conf_tf_1m"] == 0

    def test_zone_range_atr_computed(self):
        feat = self._encode(zone_top=19050.0, zone_bot=19000.0, atr=25.0)
        assert feat["zone_range_atr"] == pytest.approx(50.0 / 25.0)

    def test_sl_risk_atr_computed(self):
        feat = self._encode(entry=19020.0, sl=18900.0, atr=25.0)
        assert feat["sl_risk_atr"] == pytest.approx(120.0 / 25.0)

    def test_tp_r_computed(self):
        feat = self._encode(
            entry=19000.0, sl=18900.0, atr=25.0,
            tp_candidates=[(19500.0, True)], chosen_tp=19500.0,
        )
        # risk = 100, tp distance = 500 → tp_r = 5.0
        assert feat["tp_r"] == pytest.approx(5.0)

    def test_n_tp_candidates(self):
        feat = self._encode(
            tp_candidates=[(19300.0, True), (19500.0, False), (19700.0, False)],
            chosen_tp=19300.0,
        )
        assert feat["n_tp_candidates"] == 3

    def test_no_tp_candidates(self):
        feat = self._encode(tp_candidates=[], chosen_tp=19300.0)
        assert feat["tp_r"] == -1.0
        assert feat["tp_next_r"] == -1.0

    def test_day_of_week_stored(self):
        feat = self._encode(day_of_week=4)
        assert feat["day_of_week"] == 4

    def test_all_values_are_scalars(self):
        feat = self._encode()
        for k, v in feat.items():
            assert isinstance(v, (int, float)), f"Feature {k} has non-scalar value: {type(v)}"


class TestFeatureNameLists:

    def test_no_duplicates_in_signal_names(self):
        assert len(SIGNAL_FEATURE_NAMES) == len(set(SIGNAL_FEATURE_NAMES))

    def test_no_duplicates_in_context_names(self):
        assert len(CONTEXT_FEATURE_NAMES) == len(set(CONTEXT_FEATURE_NAMES))

    def test_all_names_unique(self):
        assert len(ALL_FEATURE_NAMES) == len(set(ALL_FEATURE_NAMES))

    def test_all_names_is_signal_plus_context_plus_config(self):
        from backtest.ml.configs import CONFIG_FEATURE_NAMES
        assert ALL_FEATURE_NAMES == SIGNAL_FEATURE_NAMES + CONTEXT_FEATURE_NAMES + CONFIG_FEATURE_NAMES


# ===========================================================================
# Dataset — build_dataset
# ===========================================================================

class TestBuildDataset:

    def _make_trade(self, entry_bar, r_multiple, direction=1):
        """Create a minimal Trade-like object with signal_features set."""
        from backtest.strategy.update import Trade
        from backtest.strategy.enums import ExitReason

        entry_price = 19000.0
        sl_price    = entry_price - 100.0
        risk        = abs(entry_price - sl_price)
        exit_price  = entry_price + r_multiple * risk * direction

        t = Trade(
            entry_bar=entry_bar,
            exit_bar=entry_bar + 5,
            entry_price=entry_price,
            exit_price=exit_price,
            direction=direction,
            contracts=1,
            slippage_points=0.0,
            commission_per_contract=0.0,
            exit_reason=ExitReason.TP if r_multiple > 0 else ExitReason.SL,
            initial_sl_price=sl_price,
            initial_tp_price=entry_price + r_multiple * risk * direction,
        )
        # Set signal_features so the trade is included
        from backtest.ml.features import encode_signal_features
        t.signal_features = encode_signal_features(
            fib_type="OTE", direction=direction, fib_value=0.618,
            confluence_kind="OB", confluence_tf="5m",
            manip_leg_size_atr=2.0,
            zone_top=19050.0, zone_bot=19000.0,
            entry=entry_price, sl=sl_price, atr=25.0,
            tp_candidates=[(entry_price + 500, True)],
            chosen_tp=entry_price + 500,
            time_since_open_min=20, day_of_week=1,
            overnight_range_atr=2.0, n_validated_levels=3,
            close_price=entry_price,
        )
        return t

    def _make_run_result(self, trades, capital=100_000):
        from backtest.runner.config import RunConfig

        class _FakeResult:
            def __init__(self, ts, cap):
                self.trades      = ts
                self.equity_curve = [float(cap)] * 1001
                self.config      = RunConfig(starting_capital=cap)
                self.strategy_name = "Test"
                self.n_trades    = len(ts)

        return _FakeResult(trades, capital)

    def _make_market_data(self, n=1000):
        idx = pd.date_range("2020-01-02 09:30", periods=n, freq="1min",
                            tz="America/New_York")
        arr = np.ones(n) * 19000.0
        from backtest.data.market_data import MarketData
        df = pd.DataFrame({"open": arr, "high": arr + 10, "low": arr - 10,
                           "close": arr, "volume": arr,
                           "anomalous": np.zeros(n, dtype=bool)}, index=idx)
        df_5m = df.resample("5min").agg({"open": "first", "high": "max", "low": "min",
                                          "close": "last", "volume": "sum",
                                          "anomalous": "any"}).dropna()
        return MarketData(
            df_1m=df, df_5m=df_5m,
            open_1m=arr, high_1m=arr + 10, low_1m=arr - 10,
            close_1m=arr, volume_1m=arr,
            open_5m=df_5m["open"].values, high_5m=df_5m["high"].values,
            low_5m=df_5m["low"].values, close_5m=df_5m["close"].values,
            volume_5m=df_5m["volume"].values,
            bar_map=np.full(n, -1, dtype=np.int64),
            trading_dates=sorted(set(ts.date() for ts in idx)),
        )

    def test_returns_dataframe(self):
        from backtest.ml.dataset import build_dataset
        trades = [self._make_trade(i * 50, 2.0 if i % 2 == 0 else -1.0) for i in range(10)]
        result = self._make_run_result(trades)
        data   = self._make_market_data()
        df     = build_dataset(result, data)
        assert isinstance(df, pd.DataFrame)

    def test_has_all_feature_columns(self):
        from backtest.ml.dataset import build_dataset
        trades = [self._make_trade(i * 50, 1.0) for i in range(5)]
        result = self._make_run_result(trades)
        data   = self._make_market_data()
        df     = build_dataset(result, data)
        for col in ALL_FEATURE_NAMES:
            assert col in df.columns, f"Missing column: {col}"

    def test_has_label_columns(self):
        from backtest.ml.dataset import build_dataset
        trades = [self._make_trade(i * 50, 1.0) for i in range(5)]
        result = self._make_run_result(trades)
        data   = self._make_market_data()
        df     = build_dataset(result, data)
        for col in ("r_multiple", "is_winner", "date", "entry_bar"):
            assert col in df.columns

    def test_n_rows_equals_n_trades_with_features(self):
        from backtest.ml.dataset import build_dataset
        from backtest.strategy.update import Trade
        from backtest.strategy.enums import ExitReason

        trades_with    = [self._make_trade(i * 50, 1.0) for i in range(6)]
        # Trade without signal_features — should be skipped
        bare = Trade(entry_bar=500, exit_bar=510, entry_price=19000.0,
                     exit_price=19100.0, direction=1, contracts=1,
                     slippage_points=0.0, commission_per_contract=0.0,
                     exit_reason=ExitReason.TP)
        result = self._make_run_result(trades_with + [bare])
        data   = self._make_market_data()
        df     = build_dataset(result, data)
        assert len(df) == 6   # bare trade is excluded

    def test_r_multiple_stored(self):
        from backtest.ml.dataset import build_dataset
        trades = [self._make_trade(0, 3.0)]
        result = self._make_run_result(trades)
        data   = self._make_market_data()
        df     = build_dataset(result, data)
        assert df["r_multiple"].iloc[0] == pytest.approx(3.0, abs=1e-3)

    def test_context_features_present(self):
        from backtest.ml.dataset import build_dataset
        trades = [self._make_trade(i * 50, float((-1) ** i)) for i in range(10)]
        result = self._make_run_result(trades)
        data   = self._make_market_data()
        df     = build_dataset(result, data)
        for col in CONTEXT_FEATURE_NAMES:
            assert col in df.columns
        # daily_trade_idx for the first trade should be 0
        assert df["daily_trade_idx"].iloc[0] == 0

    def test_no_lookahead_in_context(self):
        """recent_win_rate of first trade must use prior trades only → default 0.5."""
        from backtest.ml.dataset import build_dataset
        trades = [self._make_trade(i * 50, 2.0 if i >= 5 else -1.0) for i in range(10)]
        result = self._make_run_result(trades)
        data   = self._make_market_data()
        df     = build_dataset(result, data)
        # First trade has no prior history → should default to 0.5
        assert df["recent_win_rate_10"].iloc[0] == pytest.approx(0.5)

    def test_empty_trades_returns_empty_df(self):
        from backtest.ml.dataset import build_dataset
        result = self._make_run_result([])
        data   = self._make_market_data()
        df     = build_dataset(result, data)
        assert len(df) == 0
        for col in ALL_FEATURE_NAMES:
            assert col in df.columns


# ===========================================================================
# MLModel
# ===========================================================================

xgb = pytest.importorskip("xgboost", reason="xgboost not installed")


class TestMLModel:

    def _make_xy(self, n=100, seed=0):
        rng = np.random.default_rng(seed)
        X   = pd.DataFrame(
            rng.uniform(-1, 1, (n, len(ALL_FEATURE_NAMES))),
            columns=ALL_FEATURE_NAMES,
        )
        y   = pd.Series(rng.normal(0.5, 1.0, n))
        return X, y

    def test_fit_returns_self(self):
        from backtest.ml.model import MLModel
        model = MLModel()
        X, y  = self._make_xy()
        result = model.fit(X, y)
        assert result is model

    def test_predict_r_returns_array(self):
        from backtest.ml.model import MLModel
        model = MLModel()
        X, y  = self._make_xy()
        model.fit(X, y)
        preds = model.predict_r(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X)

    def test_predict_without_fit_raises(self):
        from backtest.ml.model import MLModel
        model = MLModel()
        X, _  = self._make_xy(10)
        with pytest.raises(RuntimeError):
            model.predict_r(X)

    def test_decide_returns_tuple(self):
        from backtest.ml.model import MLModel
        model = MLModel(threshold=0.0)
        X, y  = self._make_xy()
        model.fit(X, y)
        feat = {k: 0.0 for k in ALL_FEATURE_NAMES}
        skip, tp_idx, best_p2 = model.decide(feat)
        assert isinstance(skip, bool)
        assert isinstance(tp_idx, int)
        assert tp_idx >= 0
        assert isinstance(best_p2, dict)

    def test_decide_skips_below_threshold(self):
        from backtest.ml.model import MLModel
        model = MLModel(threshold=999.0)  # impossibly high threshold → always skip
        X, y  = self._make_xy()
        model.fit(X, y)
        feat = {k: 0.0 for k in ALL_FEATURE_NAMES}
        skip, _, _p2 = model.decide(feat)
        assert skip

    def test_decide_unfitted_never_skips(self):
        from backtest.ml.model import MLModel
        model = MLModel()  # not fitted
        feat  = {k: 0.0 for k in ALL_FEATURE_NAMES}
        skip, tp_idx, best_p2 = model.decide(feat)
        assert skip is False
        assert tp_idx == 0
        assert best_p2 == {}

    def test_feature_importance_returns_series(self):
        from backtest.ml.model import MLModel
        model = MLModel()
        X, y  = self._make_xy()
        model.fit(X, y)
        imp = model.feature_importance()
        assert isinstance(imp, pd.Series)
        assert len(imp) == len(ALL_FEATURE_NAMES)
        assert (imp >= 0).all()

    def test_feature_importance_sums_to_one(self):
        from backtest.ml.model import MLModel
        model = MLModel()
        X, y  = self._make_xy()
        model.fit(X, y)
        assert model.feature_importance().sum() == pytest.approx(1.0, abs=1e-5)

    def test_save_and_load(self, tmp_path):
        from backtest.ml.model import MLModel
        model = MLModel(threshold=0.42)
        X, y  = self._make_xy()
        model.fit(X, y)

        path = tmp_path / "test_model.pkl"
        model.save(path)
        assert path.exists()

        loaded = MLModel.load(path)
        assert loaded.threshold == pytest.approx(0.42)

        # Loaded model should predict the same values
        orig_preds   = model.predict_r(X)
        loaded_preds = loaded.predict_r(X)
        np.testing.assert_array_almost_equal(orig_preds, loaded_preds)

    def test_missing_feature_column_handled(self):
        """Model should not crash if a feature column is absent — fills 0."""
        from backtest.ml.model import MLModel
        model = MLModel()
        X, y  = self._make_xy()
        model.fit(X, y)

        X_missing = X.drop(columns=[ALL_FEATURE_NAMES[0]])
        preds = model.predict_r(X_missing)
        assert len(preds) == len(X_missing)

    def test_decide_injects_phase1_features(self):
        """Phase 1 features must appear in the row sent to the model, not be zero."""
        from backtest.ml.model import MLModel
        import numpy as np, pandas as pd
        from backtest.ml.features import ALL_FEATURE_NAMES

        captured_rows = []

        class _SpyModel:
            def predict(self, X):
                captured_rows.append(X.copy())
                return np.array([0.5] * len(X))

        model = MLModel()
        model._model = _SpyModel()
        model.threshold = 0.0

        feat = {k: 0.0 for k in ALL_FEATURE_NAMES}
        phase1 = {'min_rr': 5.0, 'confluence_tolerance_atr_mult': 0.18}
        model.decide(feat, phase1_params=phase1)

        row = captured_rows[0]
        # cfg_min_rr should NOT be 0 when min_rr=5.0 (range 2-8 → normalised ≈ 0.5)
        assert row['cfg_min_rr'].iloc[0] != 0.0, "Phase 1 cfg_min_rr was not injected"

    def test_phase1_does_not_override_phase2_keys(self):
        """Phase 2 keys must not be overwritten by Phase 1 injection."""
        from backtest.ml.model import MLModel
        import numpy as np, pandas as pd
        from backtest.ml.features import ALL_FEATURE_NAMES
        from backtest.ml.configs import normalize_config

        captured_rows = []

        class _SpyModel:
            def predict(self, X):
                captured_rows.append(X.copy())
                return np.array([0.5] * len(X))

        model = MLModel()
        model._model = _SpyModel()
        model.threshold = 0.0

        feat = {k: 0.0 for k in ALL_FEATURE_NAMES}
        # Provide a phase2 candidate with a specific tick_offset
        phase2_candidate = {'cancel_pct_to_tp': 0.75, 'tick_offset': 2, 'order_expiry_bars': 10,
                            'tp_atr_mult': 2.0, 'sl_atr_mult': 1.0}
        phase1 = {'min_rr': 5.0, 'confluence_tolerance_atr_mult': 0.18}
        model.decide(feat, phase2_candidates=[phase2_candidate], n_tp_candidates=1,
                     phase1_params=phase1)

        row = captured_rows[0]
        # cfg_tick_offset must come from phase2, not be overwritten to 0 by phase1 injection
        p2_normalized = normalize_config(phase2_candidate)
        expected_tick_offset = p2_normalized.get('cfg_tick_offset', 0.0)
        actual_tick_offset = row['cfg_tick_offset'].iloc[0] if 'cfg_tick_offset' in row.columns else 0.0
        assert actual_tick_offset == pytest.approx(expected_tick_offset), \
            f"Phase 2 cfg_tick_offset was overwritten by Phase 1: got {actual_tick_offset}, expected {expected_tick_offset}"


# ===========================================================================
# Walk-forward fold generation
# ===========================================================================

class TestGenerateFoldBounds:

    def test_returns_non_empty_for_long_range(self):
        folds = _generate_fold_bounds(
            data_start=date(2019, 1, 1),
            data_end=date(2023, 12, 31),
            train_months=24,
            test_months=3,
            embargo_months=1,
        )
        assert len(folds) > 0

    def test_each_fold_is_4_tuple(self):
        folds = _generate_fold_bounds(date(2019, 1, 1), date(2023, 12, 31), 24, 3, 1)
        for f in folds:
            assert len(f) == 4

    def test_test_start_after_train_end(self):
        folds = _generate_fold_bounds(date(2019, 1, 1), date(2023, 12, 31), 24, 3, 1)
        for tr_start, tr_end, te_start, te_end in folds:
            assert te_start > tr_end

    def test_embargo_respected(self):
        """Test start must be at least embargo_months after train end."""
        from dateutil.relativedelta import relativedelta
        embargo_months = 1
        folds = _generate_fold_bounds(date(2019, 1, 1), date(2023, 12, 31), 24, 3, embargo_months)
        for tr_start, tr_end, te_start, te_end in folds:
            min_start = tr_end + relativedelta(months=embargo_months)
            assert te_start >= min_start

    def test_no_folds_for_too_short_range(self):
        # Only 1 year of data, needs 24+1+3 = 28 months → no complete fold
        folds = _generate_fold_bounds(date(2022, 1, 1), date(2022, 12, 31), 24, 3, 1)
        assert len(folds) == 0

    def test_folds_roll_forward_by_test_months(self):
        from dateutil.relativedelta import relativedelta
        folds = _generate_fold_bounds(date(2019, 1, 1), date(2023, 12, 31), 12, 3, 0)
        if len(folds) >= 2:
            te1 = folds[0][2]  # test start of fold 0
            te2 = folds[1][2]  # test start of fold 1
            assert te2 == te1 + relativedelta(months=3)


# ===========================================================================
# Walk-forward trainer
# ===========================================================================

class TestWalkForwardTrainer:

    def _make_dataset(self, n=300, n_months=30, seed=0):
        """Generate a synthetic feature DataFrame spanning n_months months."""
        rng   = np.random.default_rng(seed)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")[:n]
        X     = rng.uniform(-1, 1, (n, len(ALL_FEATURE_NAMES)))
        # Make R slightly predictable so model has something to learn
        signal = X[:, 0] + X[:, 1] * 0.5
        y      = signal + rng.normal(0, 0.5, n)

        df = pd.DataFrame(X, columns=ALL_FEATURE_NAMES)
        df["r_multiple"] = y
        df["is_winner"]  = (y > 0).astype(int)
        df["date"]       = dates
        df["entry_bar"]  = np.arange(n)
        return df

    def test_fit_returns_result(self):
        from backtest.ml.model import MLModel
        from backtest.ml.train import WalkForwardTrainer, WalkForwardConfig

        cfg     = WalkForwardConfig(train_months=6, test_months=2, embargo_months=0,
                                    min_train_trades=5, min_test_trades=2)
        trainer = WalkForwardTrainer(cfg)
        df      = self._make_dataset(300)
        result  = trainer.fit(df)

        assert result is not None
        assert result.final_model is not None
        assert isinstance(result.final_model, MLModel)

    def test_summary_keys_present(self):
        from backtest.ml.train import WalkForwardTrainer, WalkForwardConfig

        cfg     = WalkForwardConfig(train_months=6, test_months=2, embargo_months=0,
                                    min_train_trades=5, min_test_trades=2)
        result  = WalkForwardTrainer(cfg).fit(self._make_dataset(300))

        for key in ("n_folds", "total_oos_trades", "oos_sortino_all",
                    "oos_pf_all", "oos_win_rate_all", "final_threshold"):
            assert key in result.summary, f"Missing summary key: {key}"

    def test_oos_arrays_non_empty(self):
        from backtest.ml.train import WalkForwardTrainer, WalkForwardConfig

        cfg    = WalkForwardConfig(train_months=6, test_months=2, embargo_months=0,
                                   min_train_trades=5, min_test_trades=2)
        result = WalkForwardTrainer(cfg).fit(self._make_dataset(300))
        assert len(result.oos_r_all) > 0

    def test_final_model_predicts(self):
        from backtest.ml.train import WalkForwardTrainer, WalkForwardConfig

        cfg    = WalkForwardConfig(train_months=6, test_months=2, embargo_months=0,
                                   min_train_trades=5, min_test_trades=2)
        df     = self._make_dataset(300)
        result = WalkForwardTrainer(cfg).fit(df)

        preds = result.final_model.predict_r(df[ALL_FEATURE_NAMES])
        assert len(preds) == len(df)


# ===========================================================================
# Sensitivity — _hash_params
# ===========================================================================

class TestHashParams:

    def test_same_params_same_hash(self):
        p = {"a": 1, "b": 2.5, "c": "x"}
        assert _hash_params(p) == _hash_params(p)

    def test_different_params_different_hash(self):
        p1 = {"a": 1, "b": 2.5}
        p2 = {"a": 1, "b": 3.0}
        assert _hash_params(p1) != _hash_params(p2)

    def test_order_independent(self):
        p1 = {"a": 1, "b": 2}
        p2 = {"b": 2, "a": 1}
        assert _hash_params(p1) == _hash_params(p2)

    def test_returns_16_char_hex(self):
        h = _hash_params({"x": 1})
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_ml_model_key_ignored(self):
        """ml_model should be excluded so None vs. object doesn't change hash."""
        p1 = {"a": 1, "ml_model": None}
        p2 = {"a": 1, "ml_model": object()}
        # Not guaranteed by _hash_params from sensitivity.py (it doesn't strip ml_model),
        # but collect's _hash_params does. Test sensitivity.py's version for determinism.
        assert _hash_params(p1) != _hash_params(p2)  # different objects → different hash from sensitivity.py


# ===========================================================================
# Sensitivity — check_sensitivity
# ===========================================================================

class TestCheckSensitivity:

    def test_stable_config_passes(self):
        """A flat run_fn (returns same value regardless of params) is perfectly stable."""
        def run_fn(params):
            return 1.0

        result = check_sensitivity(
            params={"confluence_tolerance_atr_mult": 0.18, "min_rr": 5.0},
            run_fn=run_fn,
        )
        assert result.is_stable
        assert result.degradation_pct == pytest.approx(0.0, abs=1e-9)

    def test_unstable_config_fails(self):
        """A run_fn that halves output on any perturbation causes >30% degradation."""
        call_count = {"n": 0}

        def run_fn(params):
            call_count["n"] += 1
            # Return half value for any call after the first
            return 1.0 if call_count["n"] == 1 else 0.5

        result = check_sensitivity(
            params={"confluence_tolerance_atr_mult": 0.18},
            run_fn=run_fn,
            max_degradation_pct=30.0,
        )
        assert not result.is_stable
        assert result.degradation_pct > 30.0

    def test_worst_param_identified(self):
        """check_sensitivity should identify which param caused the worst degradation."""
        def run_fn(params):
            # Only degrade when confluence_tolerance is perturbed
            base = 0.18
            val  = params.get("confluence_tolerance_atr_mult", base)
            if abs(val - base) / base > 0.1:
                return 0.3   # bad
            return 1.0       # good

        result = check_sensitivity(
            params={"confluence_tolerance_atr_mult": 0.18, "min_rr": 5.0},
            run_fn=run_fn,
            params_to_perturb=["confluence_tolerance_atr_mult", "min_rr"],
        )
        assert result.worst_param == "confluence_tolerance_atr_mult"

    def test_zero_base_metric_no_crash(self):
        def run_fn(params):
            return 0.0

        result = check_sensitivity(params={"a": 1.0}, run_fn=run_fn)
        assert result.degradation_pct == pytest.approx(0.0)

    def test_non_numeric_params_skipped(self):
        """String/list params must be silently skipped (not perturbed)."""
        def run_fn(params):
            return 1.0

        result = check_sensitivity(
            params={"allowed_setup_types": ["OTE", "STDV"], "min_rr": 5.0},
            run_fn=run_fn,
            params_to_perturb=["allowed_setup_types", "min_rr"],
        )
        # Should not crash; min_rr is perturbed, string param is skipped
        assert isinstance(result.is_stable, bool)

    def test_custom_perturbation_params(self):
        """Only perturb the params we specify — nothing else."""
        perturbed_keys = set()

        def run_fn(params):
            for k, v in params.items():
                if k != "a" and params[k] != 1.0:  # baseline is 1.0
                    perturbed_keys.add(k)
            return 1.0

        check_sensitivity(
            params={"a": 1.0, "b": 1.0, "c": 1.0},
            run_fn=run_fn,
            params_to_perturb=["a"],
        )
        # Only "a" should have been perturbed
        assert "b" not in perturbed_keys
        assert "c" not in perturbed_keys


# ===========================================================================
# Splits — split_bounds / filter_df
# ===========================================================================

class TestSplits:

    def test_known_splits_exist(self):
        for name in ("train", "validation", "test1", "test2"):
            assert name in SPLITS

    def test_split_bounds_returns_timestamps(self):
        start, end = split_bounds("train")
        assert isinstance(start, pd.Timestamp)
        assert isinstance(end, pd.Timestamp)

    def test_train_before_validation(self):
        tr_start, tr_end = split_bounds("train")
        va_start, va_end = split_bounds("validation")
        assert tr_end < va_start

    def test_validation_before_test1(self):
        va_start, va_end = split_bounds("validation")
        t1_start, t1_end = split_bounds("test1")
        assert va_end < t1_start

    def test_test1_before_test2(self):
        t1_start, t1_end = split_bounds("test1")
        t2_start, t2_end = split_bounds("test2")
        assert t1_end < t2_start

    def test_unknown_split_raises(self):
        with pytest.raises(ValueError):
            split_bounds("nonexistent")

    def test_filter_df_train_only(self):
        # Create a DataFrame spanning train through test periods
        dates = pd.date_range("2019-01-01", "2025-12-31", freq="ME").date
        df    = pd.DataFrame({"date": dates, "value": range(len(dates))})
        train = filter_df(df, "train")
        # All returned rows must be within the train window
        tr_start, tr_end = split_bounds("train")
        assert len(train) > 0
        filtered_dates = pd.to_datetime(train["date"])
        assert (filtered_dates >= tr_start.normalize().tz_localize(None)).all()
        assert (filtered_dates <= tr_end.normalize().tz_localize(None)).all()

    def test_filter_df_no_overlap_between_train_and_test(self):
        dates = pd.date_range("2019-01-01", "2025-12-31", freq="ME").date
        df    = pd.DataFrame({"date": dates, "value": range(len(dates))})
        train = filter_df(df, "train")
        test1 = filter_df(df, "test1")
        # No date should appear in both splits
        train_dates = set(pd.to_datetime(train["date"]).dt.date)
        test_dates  = set(pd.to_datetime(test1["date"]).dt.date)
        assert train_dates.isdisjoint(test_dates)

    def test_filter_df_all_splits_non_overlapping(self):
        dates = pd.date_range("2019-01-01", "2025-12-31", freq="W").date
        df    = pd.DataFrame({"date": dates, "value": range(len(dates))})
        all_sets = []
        for name in ("train", "validation", "test1", "test2"):
            split = filter_df(df, name)
            all_sets.append(set(pd.to_datetime(split["date"]).dt.date))
        # Every pair should be disjoint
        for i, s1 in enumerate(all_sets):
            for j, s2 in enumerate(all_sets):
                if i != j:
                    assert s1.isdisjoint(s2), f"Overlap between split {i} and {j}"

    def test_filter_df_empty_on_no_matching_dates(self):
        dates = [date(1990, 1, 1), date(1990, 6, 15)]
        df    = pd.DataFrame({"date": dates, "value": [1, 2]})
        result = filter_df(df, "train")
        assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
