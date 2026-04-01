"""
MLModel — XGBoost wrapper for the ICT/SMC trade filter.

Predicts the expected R-multiple for a given trade setup.
At inference time, skip the trade if predicted R < threshold.
Also outputs which TP candidate index to use (0 = nearest).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from backtest.ml.features import ALL_FEATURE_NAMES


class MLModel:
    """
    Wraps an XGBoost regressor that predicts expected R-multiple per trade.

    Usage
    -----
    # Training
    model = MLModel()
    model.fit(X_train, y_train)

    # Inference (called inside ICTSMCStrategy.generate_signals)
    skip, tp_idx = model.decide(signal_features_dict, threshold=0.3)
    """

    def __init__(self, threshold: float = 0.0, xgb_params: Optional[dict] = None):
        """
        Parameters
        ----------
        threshold : float
            Skip the trade if predicted R-multiple < threshold.
            0.0 means take all trades with positive expectation.
        xgb_params : dict, optional
            Overrides for XGBoost hyperparameters.
        """
        self.threshold = threshold
        self._model = None
        self._feature_names = ALL_FEATURE_NAMES

        default_xgb = dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,   # conservative — prevents fitting to tiny leaf groups
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        if xgb_params:
            default_xgb.update(xgb_params)
        self._xgb_params = default_xgb

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MLModel':
        """
        Train on a feature DataFrame X and R-multiple target y.
        X must contain all columns in ALL_FEATURE_NAMES.
        """
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError("xgboost is required: pip install xgboost") from e

        X_arr = self._prepare_X(X)
        self._model = XGBRegressor(**self._xgb_params)
        self._model.fit(X_arr, y.values)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_r(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted R-multiple for each row of X."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self._model.predict(self._prepare_X(X))

    def decide(
        self,
        signal_features: dict,
        n_tp_candidates: int = 1,
    ) -> Tuple[bool, int]:
        """
        Make a take/skip decision and choose the TP candidate index.

        Parameters
        ----------
        signal_features : dict
            Feature dict from encode_signal_features().
        n_tp_candidates : int
            How many TP candidates are available.

        Returns
        -------
        (skip, tp_idx)
            skip   : True → skip this trade; zone stays unconsumed.
            tp_idx : Index into the ordered TP candidate list (0 = nearest/default).
        """
        if self._model is None:
            return False, 0

        row = {k: signal_features.get(k, 0) for k in self._feature_names}
        X = pd.DataFrame([row])
        pred_r = float(self.predict_r(X)[0])

        skip = pred_r < self.threshold

        # TP selection: if n_tp_candidates > 1 use predicted R to decide
        # whether the higher target is reachable.  Simple heuristic:
        # choose the farthest TP that still has expected value (pred_r > 0).
        # Currently the model predicts a single R — pick TP index based on
        # how high pred_r is relative to tp_r values in the features.
        tp_idx = self._select_tp_idx(signal_features, n_tp_candidates, pred_r)

        return skip, tp_idx

    def _select_tp_idx(self, sf: dict, n_candidates: int, pred_r: float) -> int:
        """
        Choose which TP candidate index to use.
        If pred_r is well above the nearest TP's R, consider a higher target.
        Falls back to 0 (nearest) if information is unavailable.
        """
        if n_candidates <= 1 or pred_r <= 0:
            return 0
        tp_r      = sf.get('tp_r', -1.0)
        tp_next_r = sf.get('tp_next_r', -1.0)
        if tp_r <= 0 or tp_next_r <= 0:
            return 0
        # Go for the next TP if predicted R is at least 50% of the way there
        # (simple geometric scaling rule — can be replaced by a second model)
        midpoint = tp_r + 0.5 * (tp_next_r - tp_r)
        return 1 if pred_r >= midpoint and n_candidates >= 2 else 0

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.Series:
        """Return feature importances sorted descending."""
        if self._model is None:
            raise RuntimeError("Model not trained.")
        imp = self._model.feature_importances_
        return pd.Series(imp, index=self._feature_names).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'threshold': self.threshold,
                         'xgb_params': self._xgb_params,
                         'model': self._model,
                         'feature_names': self._feature_names}, f)

    @classmethod
    def load(cls, path: str | Path) -> 'MLModel':
        with open(path, 'rb') as f:
            state = pickle.load(f)
        obj = cls(threshold=state['threshold'], xgb_params=state['xgb_params'])
        obj._model = state['model']
        obj._feature_names = state['feature_names']
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prepare_X(self, X: pd.DataFrame) -> np.ndarray:
        """Ensure all expected features are present and in the correct order."""
        for col in self._feature_names:
            if col not in X.columns:
                X = X.copy()
                X[col] = 0
        return X[self._feature_names].values.astype(float)
