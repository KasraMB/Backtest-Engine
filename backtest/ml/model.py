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
from backtest.ml.configs import normalize_config, CONFIG_FEATURE_NAMES

# Phase 2 cfg_ keys — must NOT be overwritten by Phase 1 injection.
_PHASE2_CFG_KEYS: frozenset[str] = frozenset({
    'cfg_cancel_pct_to_tp',
    'cfg_tick_offset',
    'cfg_order_expiry_bars',
    'cfg_base_metric',
})


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

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight: np.ndarray | None = None) -> 'MLModel':
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
        self._model.fit(X_arr, y.values, sample_weight=sample_weight)
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
        phase2_candidates: Optional[list] = None,
        n_tp_candidates: int = 1,
        phase1_params: Optional[dict] = None,
    ) -> Tuple[bool, int, dict]:
        """
        Make a take/skip decision, choose the TP candidate index, and select
        the best Phase 2 execution config for this trade.

        Parameters
        ----------
        signal_features : dict
            Feature dict from encode_signal_features() — signal + context features.
            Should NOT include config features; those are added here per candidate.
        phase2_candidates : list of dicts, optional
            Each dict contains Phase 2 params (cancel_pct_to_tp, tick_offset_atr_mult,
            order_expiry_bars).  The model is queried for each candidate and the
            one with the highest predicted R is selected.
            If None or empty, falls back to the signal features as-is (single query).
        n_tp_candidates : int
            How many TP candidates are available.

        Returns
        -------
        (skip, tp_idx, best_phase2_config)
            skip             : True → skip this trade; zone stays unconsumed.
            tp_idx           : Index into the ordered TP candidate list (0 = nearest).
            best_phase2_config : dict of Phase 2 params to apply to this trade's order.
                                 Empty dict if no candidates were provided.
        """
        if self._model is None:
            return False, 0, {}

        # Build Phase 1 cfg_ overrides (fixed for this trade, same across all Phase 2 candidates).
        phase1_cfg: dict[str, float] = {}
        if phase1_params is not None:
            for k, v in normalize_config(phase1_params).items():
                if k not in _PHASE2_CFG_KEYS:
                    phase1_cfg[k] = v

        # Build candidate list — each entry is (phase2_params, feature_row)
        if phase2_candidates:
            rows = []
            for p2 in phase2_candidates:
                cfg_feat = normalize_config(p2)
                row      = {k: signal_features.get(k, 0) for k in self._feature_names}
                # Phase 1 first (lower priority), then Phase 2 overrides its own keys.
                for k, v in phase1_cfg.items():
                    if k in self._feature_names:
                        row[k] = v
                for k, v in cfg_feat.items():
                    if k in self._feature_names:
                        row[k] = v
                rows.append(row)
            X       = pd.DataFrame(rows)
            preds   = self.predict_r(X)
            best_i  = int(np.argmax(preds))
            pred_r  = float(preds[best_i])
            best_p2 = phase2_candidates[best_i]
            best_sf = {**signal_features, **phase1_cfg, **normalize_config(best_p2)}
        else:
            row    = {k: signal_features.get(k, 0) for k in self._feature_names}
            for k, v in phase1_cfg.items():
                if k in self._feature_names:
                    row[k] = v
            X      = pd.DataFrame([row])
            pred_r = float(self.predict_r(X)[0])
            best_p2 = {}
            best_sf = {**signal_features, **phase1_cfg}

        skip   = pred_r < self.threshold
        tp_idx = self._select_tp_idx(best_sf, n_tp_candidates, pred_r)

        return skip, tp_idx, best_p2

    def _select_tp_idx(self, sf: dict, n_candidates: int, pred_r: float) -> int:
        """
        Choose which TP candidate index to use based on predicted R vs. TP levels.
        Falls back to 0 (nearest) if information is unavailable.
        """
        if n_candidates <= 1 or pred_r <= 0:
            return 0
        tp_r      = sf.get('tp_r', -1.0)
        tp_next_r = sf.get('tp_next_r', -1.0)
        if tp_r <= 0 or tp_next_r <= 0:
            return 0
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

    def _prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure all expected features are present and in the correct order."""
        missing = [col for col in self._feature_names if col not in X.columns]
        if missing:
            X = X.copy()
            for col in missing:
                X[col] = 0
        return X[self._feature_names].astype(float)


class EnsembleMLModel:
    """
    Ensemble of Model A (asymmetric-R XGBoost regressor) and
    Model B (calibrated XGBoost classifier).

    Inference score: score = pred_asymmetric_r × P(win)
    Skips trade if score < threshold.
    Drop-in replace for MLModel — same decide() interface.
    """

    def __init__(self, threshold: float = 0.0, loss_penalty: float = 1.5):
        self.threshold    = threshold
        self.loss_penalty = loss_penalty
        self._model_a: Optional['MLModel'] = None
        self._model_b = None
        self._feature_names = ALL_FEATURE_NAMES

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y_r: pd.Series, y_bin: pd.Series,
            sample_weight: np.ndarray | None = None) -> 'EnsembleMLModel':
        """
        Train both sub-models.

        Parameters
        ----------
        X             : feature DataFrame with all ALL_FEATURE_NAMES columns
        y_r           : R-multiple targets
        y_bin         : binary win/loss labels (1=win, 0=loss)
        sample_weight : optional per-sample weights (e.g. inverse config frequency)
        """
        try:
            from xgboost import XGBClassifier
            from sklearn.calibration import CalibratedClassifierCV
        except ImportError as e:
            raise ImportError("xgboost and scikit-learn are required") from e

        import warnings

        # Model A: asymmetric-R regressor
        y_asym = y_r.apply(lambda r: r if r > 0 else r * self.loss_penalty)
        self._model_a = MLModel()
        self._model_a.fit(X, y_asym, sample_weight=sample_weight)

        # Model B: calibrated classifier
        spw = float((y_bin == 0).sum()) / max(float((y_bin == 1).sum()), 1.0)
        xgb_clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
        )
        self._model_b = CalibratedClassifierCV(xgb_clf, cv=3, method='isotonic')
        X_arr = X[self._feature_names].values.astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model_b.fit(X_arr, y_bin.values, sample_weight=sample_weight)

        return self

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _build_row(
        self,
        signal_features: dict,
        phase1_cfg: Optional[dict] = None,
        phase2_params: Optional[dict] = None,
    ) -> dict:
        row = {k: signal_features.get(k, 0) for k in self._feature_names}
        if phase1_cfg:
            for k, v in phase1_cfg.items():
                if k in self._feature_names:
                    row[k] = v
        if phase2_params:
            cfg_feat = normalize_config(phase2_params)
            for k, v in cfg_feat.items():
                if k in self._feature_names:
                    row[k] = v
        return row

    def predict_r(self, X: pd.DataFrame) -> np.ndarray:
        """Return ensemble score (pred_r × P(win)) for each row — drop-in for MLModel.predict_r."""
        pred_r = self._model_a.predict_r(X)
        X_arr  = X[self._feature_names].values.astype(float)
        p_win  = self._model_b.predict_proba(X_arr)[:, 1]
        return pred_r * p_win

    def score(
        self,
        signal_features: dict,
        phase2_params: Optional[dict] = None,
        phase1_params: Optional[dict] = None,
    ) -> float:
        """Return ensemble score = pred_asymmetric_r × P(win) for a single row."""
        phase1_cfg: dict = {}
        if phase1_params is not None:
            for k, v in normalize_config(phase1_params).items():
                if k not in _PHASE2_CFG_KEYS:
                    phase1_cfg[k] = v

        row = self._build_row(signal_features, phase1_cfg, phase2_params)
        X = pd.DataFrame([row])
        pred_r = float(self._model_a.predict_r(X)[0])
        X_arr  = X[self._feature_names].values.astype(float)
        p_win  = float(self._model_b.predict_proba(X_arr)[0, 1])
        return float(pred_r * p_win)

    # ------------------------------------------------------------------
    # Inference — decide()
    # ------------------------------------------------------------------

    def decide(
        self,
        signal_features: dict,
        phase2_candidates: Optional[list] = None,
        n_tp_candidates: int = 1,
        phase1_params: Optional[dict] = None,
    ) -> tuple[bool, int, dict]:
        if self._model_a is None or self._model_b is None:
            return False, 0, {}

        # Build Phase 1 cfg_ overrides
        phase1_cfg: dict = {}
        if phase1_params is not None:
            for k, v in normalize_config(phase1_params).items():
                if k not in _PHASE2_CFG_KEYS:
                    phase1_cfg[k] = v

        if phase2_candidates:
            rows = []
            for p2 in phase2_candidates:
                rows.append(self._build_row(signal_features, phase1_cfg, p2))
            X      = pd.DataFrame(rows)
            preds_r = self._model_a.predict_r(X)
            X_arr   = X[self._feature_names].values.astype(float)
            p_wins  = self._model_b.predict_proba(X_arr)[:, 1]
            scores  = preds_r * p_wins
            best_i  = int(np.argmax(scores))
            score   = float(scores[best_i])
            best_p2 = phase2_candidates[best_i]
            pred_r_single = float(preds_r[best_i])
            best_sf = {**signal_features, **phase1_cfg, **normalize_config(best_p2)}
        else:
            row    = self._build_row(signal_features, phase1_cfg)
            X      = pd.DataFrame([row])
            pred_r_single = float(self._model_a.predict_r(X)[0])
            X_arr  = X[self._feature_names].values.astype(float)
            p_win  = float(self._model_b.predict_proba(X_arr)[0, 1])
            score  = pred_r_single * p_win
            best_p2 = {}
            best_sf = {**signal_features, **phase1_cfg}

        skip   = score < self.threshold
        tp_idx = self._model_a._select_tp_idx(best_sf, n_tp_candidates, pred_r_single)

        return skip, tp_idx, best_p2

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'threshold':     self.threshold,
            'loss_penalty':  self.loss_penalty,
            'model_a':       self._model_a,
            'model_b':       self._model_b,
            'feature_names': self._feature_names,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> 'EnsembleMLModel':
        with open(path, 'rb') as f:
            state = pickle.load(f)
        obj = cls(threshold=state['threshold'], loss_penalty=state['loss_penalty'])
        obj._model_a       = state['model_a']
        obj._model_b       = state['model_b']
        obj._feature_names = state['feature_names']
        return obj
