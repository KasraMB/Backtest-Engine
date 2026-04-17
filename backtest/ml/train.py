"""
Walk-forward training pipeline for the ICT/SMC ML trade filter.

Structure
---------
  train_months: 24   (2 years of training data per fold)
  test_months:  3    (3 months of out-of-sample evaluation)
  embargo_months: 1  (gap between train end and test start to avoid leakage)

For the 2019-2026 dataset this produces ~16 folds.
Each fold trains a fresh MLModel and generates OOS predictions.
The final model is trained on all available data.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from backtest.ml.evaluate import sortino_r, profit_factor_r, search_threshold
from backtest.ml.features import ALL_FEATURE_NAMES
from backtest.ml.model import MLModel


def _config_weights(df: pd.DataFrame) -> np.ndarray | None:
    """Per-config inverse frequency weights so every config contributes equally to the loss."""
    if 'config_hash' not in df.columns:
        return None
    counts = df['config_hash'].value_counts()
    w = df['config_hash'].map(counts).values.astype(np.float32)
    w = 1.0 / w
    return w / w.mean()  # normalize: mean weight = 1 preserves loss scale


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    train_months:   int   = 24    # training window length
    test_months:    int   = 3     # test (OOS) window length
    embargo_months: int   = 1     # gap between train end and test start
    metric:         str   = 'sortino'   # 'sortino' | 'profit_factor' | 'expectancy_r'
    min_train_trades: int = 50    # skip fold if fewer training trades
    min_test_trades:  int = 10    # skip fold if fewer test trades
    threshold_search: bool = True  # tune skip threshold per fold on train set
    min_take_rate:  float = 0.20  # min fraction of trades to keep after filtering
    xgb_params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-fold result
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_idx:         int
    train_start:      date
    train_end:        date
    test_start:       date
    test_end:         date
    n_train:          int
    n_test:           int
    threshold:        float
    oos_predicted_r:  np.ndarray
    oos_actual_r:     np.ndarray
    oos_sortino_taken:    float      # Sortino of OOS taken trades
    oos_sortino_all:      float      # Sortino of ALL OOS trades (baseline)
    oos_pf_taken:         float
    oos_pf_all:           float
    oos_win_rate_taken:   float
    oos_take_rate:        float


# ---------------------------------------------------------------------------
# Walk-forward result
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    folds:             list[FoldResult]
    final_model:       MLModel          # trained on all data
    final_threshold:   float
    oos_r_all:         np.ndarray       # all OOS actual R-multiples (chronological)
    oos_r_taken:       np.ndarray       # OOS R-multiples of taken trades
    summary:           dict             # aggregate metrics across all folds


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class WalkForwardTrainer:
    """
    Run walk-forward training on a feature DataFrame produced by build_dataset().

    Usage
    -----
    trainer = WalkForwardTrainer(WalkForwardConfig())
    result  = trainer.fit(df)
    result.final_model.save('models/ict_smc.pkl')
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()

    def fit(self, df: pd.DataFrame) -> WalkForwardResult:
        """
        Run all folds and return the consolidated result.

        df must contain columns: ALL_FEATURE_NAMES + ['date', 'r_multiple'].
        The 'date' column must be a Python date or pandas Timestamp.
        """
        cfg = self.config
        df  = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Generate fold boundaries
        folds_bounds = _generate_fold_bounds(
            df['date'].iloc[0].date(),
            df['date'].iloc[-1].date(),
            cfg.train_months,
            cfg.test_months,
            cfg.embargo_months,
        )

        feature_cols = ALL_FEATURE_NAMES

        fold_results = []
        all_oos_actual:    list = []
        all_oos_predicted: list = []
        all_oos_taken_mask: list = []

        for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(folds_bounds):
            train_mask = (df['date'].dt.date >= tr_start) & (df['date'].dt.date <= tr_end)
            test_mask  = (df['date'].dt.date >= te_start) & (df['date'].dt.date <= te_end)

            df_train = df[train_mask]
            df_test  = df[test_mask]

            if len(df_train) < cfg.min_train_trades or len(df_test) < cfg.min_test_trades:
                continue

            X_train = df_train[feature_cols]
            y_train = df_train['r_multiple']
            X_test  = df_test[feature_cols]
            y_test  = df_test['r_multiple'].values

            model = MLModel(xgb_params=cfg.xgb_params)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(X_train, y_train, sample_weight=_config_weights(df_train))

            pred_r = model.predict_r(X_test)

            # Threshold search on train set
            if cfg.threshold_search:
                train_pred = model.predict_r(X_train)
                thresh, _ = search_threshold(
                    train_pred, y_train.values,
                    metric=cfg.metric,
                    min_take_rate=cfg.min_take_rate,
                )
            else:
                thresh = 0.0

            model.threshold = thresh
            taken_mask = pred_r >= thresh

            r_taken = y_test[taken_mask]
            take_rate = float(taken_mask.mean())

            fold_res = FoldResult(
                fold_idx=fold_idx,
                train_start=tr_start,
                train_end=tr_end,
                test_start=te_start,
                test_end=te_end,
                n_train=len(df_train),
                n_test=len(df_test),
                threshold=thresh,
                oos_predicted_r=pred_r,
                oos_actual_r=y_test,
                oos_sortino_taken=sortino_r(r_taken) if len(r_taken) > 0 else 0.0,
                oos_sortino_all=sortino_r(y_test),
                oos_pf_taken=profit_factor_r(r_taken) if len(r_taken) > 0 else 0.0,
                oos_pf_all=profit_factor_r(y_test),
                oos_win_rate_taken=float(np.mean(r_taken > 0)) if len(r_taken) > 0 else 0.0,
                oos_take_rate=take_rate,
            )
            fold_results.append(fold_res)
            all_oos_actual.extend(y_test.tolist())
            all_oos_predicted.extend(pred_r.tolist())
            all_oos_taken_mask.extend(taken_mask.tolist())

        # Final model trained on all data
        X_all = df[feature_cols]
        y_all = df['r_multiple']
        final_model = MLModel(xgb_params=cfg.xgb_params)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            final_model.fit(X_all, y_all, sample_weight=_config_weights(df))

        pred_all = final_model.predict_r(X_all)
        final_thresh, _ = search_threshold(
            pred_all, y_all.values,
            metric=cfg.metric,
            min_take_rate=cfg.min_take_rate,
        ) if cfg.threshold_search else (0.0, {})
        final_model.threshold = final_thresh

        oos_actual  = np.array(all_oos_actual)
        oos_taken   = oos_actual[np.array(all_oos_taken_mask, dtype=bool)]

        summary = {
            'n_folds':           len(fold_results),
            'total_oos_trades':  len(oos_actual),
            'total_oos_taken':   len(oos_taken),
            'oos_take_rate':     len(oos_taken) / max(len(oos_actual), 1),
            'oos_sortino_all':   sortino_r(oos_actual),
            'oos_sortino_taken': sortino_r(oos_taken),
            'oos_pf_all':        profit_factor_r(oos_actual),
            'oos_pf_taken':      profit_factor_r(oos_taken),
            'oos_win_rate_all':      float(np.mean(oos_actual > 0)) if len(oos_actual) > 0 else 0.0,
            'oos_win_rate_taken':    float(np.mean(oos_taken  > 0)) if len(oos_taken)  > 0 else 0.0,
            'oos_expectancy_r_all':  float(np.mean(oos_actual)) if len(oos_actual) > 0 else 0.0,
            'oos_expectancy_r_taken': float(np.mean(oos_taken)) if len(oos_taken) > 0 else 0.0,
            'final_threshold':   final_thresh,
        }

        return WalkForwardResult(
            folds=fold_results,
            final_model=final_model,
            final_threshold=final_thresh,
            oos_r_all=oos_actual,
            oos_r_taken=oos_taken,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# Ensemble walk-forward trainer
# ---------------------------------------------------------------------------

LOSS_PENALTY: float = 1.5


@dataclass
class EnsembleWalkForwardResult:
    folds:          list[FoldResult]
    ensemble_model: 'EnsembleMLModel'
    summary:        dict


class EnsembleWalkForwardTrainer:
    """Trains EnsembleMLModel via walk-forward. Fold diagnostics use Model A."""

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()

    def fit(self, df: pd.DataFrame) -> EnsembleWalkForwardResult:
        from backtest.ml.model import EnsembleMLModel

        cfg = self.config
        df  = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        folds_bounds = _generate_fold_bounds(
            df['date'].iloc[0].date(),
            df['date'].iloc[-1].date(),
            cfg.train_months, cfg.test_months, cfg.embargo_months,
        )

        feature_cols = ALL_FEATURE_NAMES
        fold_results = []
        all_oos_actual:     list = []
        all_oos_taken_mask: list = []

        for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(folds_bounds):
            train_mask = (df['date'].dt.date >= tr_start) & (df['date'].dt.date <= tr_end)
            test_mask  = (df['date'].dt.date >= te_start) & (df['date'].dt.date <= te_end)
            df_train   = df[train_mask]
            df_test    = df[test_mask]

            if len(df_train) < cfg.min_train_trades or len(df_test) < cfg.min_test_trades:
                continue

            X_train = df_train[feature_cols]
            y_r     = df_train['r_multiple']
            y_asym  = y_r.apply(lambda r: r if r > 0 else r * LOSS_PENALTY)
            X_test  = df_test[feature_cols]
            y_test  = df_test['r_multiple'].values

            # Fold diagnostics: use Model A (asymmetric-R regressor)
            fold_model = MLModel(xgb_params=cfg.xgb_params)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fold_model.fit(X_train, y_asym, sample_weight=_config_weights(df_train))

            pred_r = fold_model.predict_r(X_test)

            if cfg.threshold_search:
                train_pred = fold_model.predict_r(X_train)
                thresh, _  = search_threshold(
                    train_pred, y_asym.values,
                    metric=cfg.metric, min_take_rate=cfg.min_take_rate,
                )
            else:
                thresh = 0.0

            fold_model.threshold = thresh
            taken_mask = pred_r >= thresh
            r_taken    = y_test[taken_mask]

            fold_results.append(FoldResult(
                fold_idx=fold_idx,
                train_start=tr_start, train_end=tr_end,
                test_start=te_start,  test_end=te_end,
                n_train=len(df_train), n_test=len(df_test),
                threshold=thresh,
                oos_predicted_r=pred_r,
                oos_actual_r=y_test,
                oos_sortino_taken=sortino_r(r_taken) if len(r_taken) > 0 else 0.0,
                oos_sortino_all=sortino_r(y_test),
                oos_pf_taken=profit_factor_r(r_taken) if len(r_taken) > 0 else 0.0,
                oos_pf_all=profit_factor_r(y_test),
                oos_win_rate_taken=float(np.mean(r_taken > 0)) if len(r_taken) > 0 else 0.0,
                oos_take_rate=float(taken_mask.mean()),
            ))
            all_oos_actual.extend(y_test.tolist())
            all_oos_taken_mask.extend(taken_mask.tolist())

        # Final EnsembleMLModel trained on all data (threshold=0.0 — set by threshold_opt)
        X_all     = df[feature_cols]
        y_r_all   = df['r_multiple']
        y_bin_all = (y_r_all > 0).astype(int)

        ensemble_model = EnsembleMLModel(threshold=0.0, loss_penalty=LOSS_PENALTY)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ensemble_model.fit(X_all, y_r_all, y_bin_all, sample_weight=_config_weights(df))

        oos_actual = np.array(all_oos_actual)
        oos_taken  = oos_actual[np.array(all_oos_taken_mask, dtype=bool)]

        summary = {
            'n_folds':            len(fold_results),
            'total_oos_trades':   len(oos_actual),
            'total_oos_taken':    len(oos_taken),
            'oos_take_rate':      len(oos_taken) / max(len(oos_actual), 1),
            'oos_sortino_all':    sortino_r(oos_actual),
            'oos_sortino_taken':  sortino_r(oos_taken),
            'oos_pf_all':         profit_factor_r(oos_actual),
            'oos_pf_taken':       profit_factor_r(oos_taken),
            'oos_win_rate_all':   float(np.mean(oos_actual > 0)) if len(oos_actual) > 0 else 0.0,
            'oos_win_rate_taken':      float(np.mean(oos_taken > 0)) if len(oos_taken) > 0 else 0.0,
            'oos_expectancy_r_all':   float(np.mean(oos_actual)) if len(oos_actual) > 0 else 0.0,
            'oos_expectancy_r_taken': float(np.mean(oos_taken))  if len(oos_taken)  > 0 else 0.0,
        }

        return EnsembleWalkForwardResult(
            folds=fold_results,
            ensemble_model=ensemble_model,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# Helper: generate fold date boundaries
# ---------------------------------------------------------------------------

def _generate_fold_bounds(
    data_start: date,
    data_end:   date,
    train_months: int,
    test_months:  int,
    embargo_months: int,
) -> list[tuple[date, date, date, date]]:
    """
    Returns list of (train_start, train_end, test_start, test_end).
    Rolls forward by test_months each step.
    """
    from dateutil.relativedelta import relativedelta

    folds = []
    test_start = data_start + relativedelta(months=train_months + embargo_months)

    while True:
        test_end   = test_start + relativedelta(months=test_months) - relativedelta(days=1)
        if test_end > data_end:
            break
        train_end   = test_start - relativedelta(months=embargo_months) - relativedelta(days=1)
        train_start = train_end - relativedelta(months=train_months) + relativedelta(days=1)
        if train_start < data_start:
            train_start = data_start
        folds.append((train_start, train_end, test_start, test_end))
        test_start += relativedelta(months=test_months)

    return folds
