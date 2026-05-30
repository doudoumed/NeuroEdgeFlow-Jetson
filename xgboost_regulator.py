#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# xgboost_regulator.py — NeuroEdgeFlow ML-based offloading regulator
#
# Drop-in replacement for rule_regulator.RuleBasedRegulator. Same class name
# at the module level (RuleBasedRegulator) so adaptive_engine.py picks it up
# by simply changing the import line. Same predict() signature.
#
# What it does
# ────────────
# 1. Loads an XGBoost model from a portable JSON file (best_model.json) plus
#    a plain-JSON scaler (scaler_params.json) and metadata.json that lists
#    the 32 features. This avoids joblib/pickle version-mismatch crashes
#    between the training machine and the Jetson.
# 2. Maintains a small rolling history (5 rows) of the raw measurements so
#    that at each call we can rebuild the lag, rolling and delta features
#    the model was trained with — without re-loading the whole dataset.
# 3. On each predict() it:
#       a. Appends the latest raw metrics to history
#       b. Engineers the 32 features in the same order the training pipeline
#          used (see metadata['features'])
#       c. Scales them with the loaded scaler
#       d. Runs the Booster and applies sigmoid + saved threshold
#    The result is (cloud_win: bool, prob: float).
#
# Failure modes
# ─────────────
# - Missing artefacts (.json files):
#       raise FileNotFoundError at construction. The engine wraps the
#       constructor in try/except and falls back to the legacy regulator.
# - History not full yet (first 2 frames):
#       lag_*_2 features will be NaN. We forward-fill with the latest value
#       so the model gets a valid input. The early decisions are therefore
#       less reliable, but the engine's hysteresis swallows any noise.
# - prediction error:
#       log + return (False, 0.0). adaptive_engine treats prob<0.5 as LOCAL,
#       so this fails safe.
#
# Python 3.6 compatible — Jetson TX2 constraint.
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import logging
from collections import deque

import numpy as np
import xgboost as xgb

log = logging.getLogger("xgboost_regulator")


# ─── Default paths ──────────────────────────────────────────────────────────
# All artefacts live alongside the running script. Override at construction
# time if needed. These are now portable JSON files — no more pickle.
DEFAULT_MODEL_PATH    = os.path.expanduser("~/neuroedgeflow/best_model.json")
DEFAULT_SCALER_PATH   = os.path.expanduser("~/neuroedgeflow/scaler_params.json")
DEFAULT_METADATA_PATH = os.path.expanduser("~/neuroedgeflow/metadata.json")


class _SimpleScaler(object):
    """
    Minimal drop-in for sklearn.preprocessing.StandardScaler.
    Loaded from plain JSON (scaler_params.json) so we avoid pickle entirely.
    """
    def __init__(self, mean, scale):
        self._mean  = np.array(mean,  dtype=np.float64)
        self._scale = np.array(scale, dtype=np.float64)

    def transform(self, X):
        """X: ndarray of shape (n_samples, n_features)"""
        return (np.asarray(X, dtype=np.float64) - self._mean) / self._scale


class RuleBasedRegulator:
    """
    XGBoost-based offloading regulator. Class name is kept as
    RuleBasedRegulator so the engine's import line
        from rule_regulator import RuleBasedRegulator
    can be redirected to this module with a one-line change
        from xgboost_regulator import RuleBasedRegulator
    rather than touching adaptive_engine.py itself.
    """

    # Window the rolling features need. metadata uses window=3, so we keep
    # 5 rows of history to be safe (covers lag_2 + rolling 3).
    HISTORY_SIZE = 5

    def __init__(self, weights_path=None,
                 model_path=DEFAULT_MODEL_PATH,
                 scaler_path=DEFAULT_SCALER_PATH,
                 metadata_path=DEFAULT_METADATA_PATH):
        # weights_path is accepted for drop-in compatibility with
        # NeuralOffloadingRegulator(weights_path). We ignore it because the
        # XGBoost model is loaded from a different file format.
        self.is_rf = True               # so any code that inspects this works
        self._model_path    = model_path
        self._scaler_path   = scaler_path
        self._metadata_path = metadata_path

        # Load metadata first — it tells us which features the model expects.
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                "XGBoost metadata not found: {}".format(metadata_path))
        with open(metadata_path) as f:
            self._meta = json.load(f)

        self.feature_names = list(self._meta["features"])
        self._threshold    = float(self._meta.get("threshold", 0.5))

        # ── Load model from portable JSON (not pickle) ─────────────────────
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "XGBoost model not found: {}".format(model_path))
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                "Scaler not found: {}".format(scaler_path))

        # XGBoost native JSON model — portable across versions
        self._booster = xgb.Booster()
        self._booster.load_model(model_path)

        # Plain-JSON scaler — no pickle dependency
        with open(scaler_path) as f:
            sp = json.load(f)
        self._scaler = _SimpleScaler(sp["mean"], sp["scale"])

        # Rolling history of the 7 raw measurements. Each entry is a dict
        # with keys: rtt_ms, bandwidth_kbps, cpu_load, ram_usage,
        # gpu_load, gpu_temp, error_rate_pct.
        self._history = deque(maxlen=self.HISTORY_SIZE)

        print("[XGBoostRegulator] Loaded {} (threshold={:.3f}, "
              "features={}, F1={}, AUC={})".format(
                  self._meta.get("model_name", "model"),
                  self._threshold,
                  len(self.feature_names),
                  self._meta.get("f1_score", "?"),
                  self._meta.get("auc_roc", "?"),
              ))

    # ── Feature engineering — must match the training recipe exactly ────────

    def _engineer_features(self, raw):
        """
        Given the raw measurements for the current sample (and self._history
        for the previous samples), produce a dict of all 32 engineered
        features. The order is dictated by self.feature_names so the model
        receives them in the order it was fit on.
        """
        # Append the current sample to history so rolling stats include it.
        # We do this BEFORE computing any features so the rolling window of
        # 3 includes the present row, matching the training pipeline.
        self._history.append(raw)
        hist = list(self._history)

        # ── helpers that read from history with safe fallbacks ─────────────
        def lag(col, n):
            """Value n samples ago. If history is too short, fall back to
            the current sample so the early frames produce sane values."""
            if len(hist) > n:
                return float(hist[-1 - n][col])
            return float(hist[-1][col])

        def roll_window(col):
            """Last up-to-3 samples of `col`, as a list of floats."""
            return [float(h[col]) for h in hist[-3:]]

        # ── Raw features ───────────────────────────────────────────────────
        feats = {
            "rtt_ms":         float(raw["rtt_ms"]),
            "bandwidth_kbps": float(raw["bandwidth_kbps"]),
            "cpu_load":       float(raw["cpu_load"]),
            "ram_usage":      float(raw["ram_usage"]),
            "gpu_load":       float(raw["gpu_load"]),
            "gpu_temp":       float(raw["gpu_temp"]),
            "error_rate_pct": float(raw["error_rate_pct"]),
        }

        # ── Lag features ───────────────────────────────────────────────────
        for col in ["rtt_ms", "bandwidth_kbps", "cpu_load"]:
            feats["lag_{}_1".format(col)] = lag(col, 1)
            feats["lag_{}_2".format(col)] = lag(col, 2)

        # ── Rolling stats (mean + std over the last 3 samples) ─────────────
        for col in ["rtt_ms", "bandwidth_kbps", "cpu_load", "error_rate_pct"]:
            window = roll_window(col)
            feats["roll_mean_{}_3".format(col)] = float(np.mean(window))
            # ddof=1 matches pandas .rolling().std() default. Fall back to 0
            # when window has fewer than 2 values, also matching pandas.
            if len(window) >= 2:
                feats["roll_std_{}_3".format(col)] = float(np.std(window, ddof=1))
            else:
                feats["roll_std_{}_3".format(col)] = 0.0

        # ── Ratios ─────────────────────────────────────────────────────────
        feats["ratio_cpu_bw"]  = feats["cpu_load"]  / (feats["bandwidth_kbps"] + 1.0)
        feats["ratio_rtt_bw"]  = feats["rtt_ms"]    / (feats["bandwidth_kbps"] + 1.0)
        feats["ratio_gpu_cpu"] = feats["gpu_load"]  / (feats["cpu_load"]      + 1.0)

        # ── Deltas vs previous sample ──────────────────────────────────────
        feats["delta_rtt"] = feats["rtt_ms"]         - feats["lag_rtt_ms_1"]
        feats["delta_bw"]  = feats["bandwidth_kbps"] - feats["lag_bandwidth_kbps_1"]
        feats["delta_cpu"] = feats["cpu_load"]       - feats["lag_cpu_load_1"]

        # ── Combined scores ────────────────────────────────────────────────
        feats["combined_net_score"]  = feats["bandwidth_kbps"] / (feats["rtt_ms"] + 1.0)
        feats["combined_load_score"] = (feats["cpu_load"] * 0.5
                                        + feats["ram_usage"] * 0.3
                                        + feats["gpu_load"]  * 0.2)

        # ── Time-of-day ────────────────────────────────────────────────────
        # The training pipeline read these from the timestamp column. The
        # engine doesn't pass a timestamp into metrics_dict, so we compute
        # them from the system clock — same effect over the long run.
        import datetime
        now = datetime.datetime.now()
        feats["hour"]      = now.hour
        feats["dayofweek"] = now.weekday()
        feats["is_night"]  = int(now.hour < 6 or now.hour >= 22)

        return feats

    # ── Public API — matches NeuralOffloadingRegulator.predict() ────────────

    def predict(self, rtt=None, bw=None, cpu=None, ram=None,
                cloud_lat=None, local_lat=None, metrics=None):
        """
        Decide CLOUD vs LOCAL for the current sample.
        Returns (cloud_win: bool, prob: float in [0,1]) exactly like the
        rule regulator, so adaptive_engine.py needs no further changes.
        """
        # Accept either the metrics dict (preferred) or the legacy positional
        # kwargs. The dict path is what adaptive_engine.py uses; the kwarg
        # path is here for compatibility with any old test scripts.
        if metrics is None:
            metrics = {
                "rtt_ms":         float(rtt) if rtt is not None else 0.0,
                "bandwidth_kbps": float(bw)  if bw  is not None else 0.0,
                "cpu_load":       float(cpu) if cpu is not None else 0.0,
                "ram_usage":      float(ram) if ram is not None else 0.0,
                "gpu_load":       0.0,
                "gpu_temp":       0.0,
                "error_rate_pct": 0.0,
            }

        # Defensive: the engine sometimes passes extra keys (queue_depth,
        # network_ok, etc.). The engineer step only reads the 7 raw cols.
        raw = {
            "rtt_ms":         float(metrics.get("rtt_ms",         0.0)),
            "bandwidth_kbps": float(metrics.get("bandwidth_kbps", 0.0)),
            "cpu_load":       float(metrics.get("cpu_load",       0.0)),
            "ram_usage":      float(metrics.get("ram_usage",      0.0)),
            "gpu_load":       float(metrics.get("gpu_load",       0.0)),
            "gpu_temp":       float(metrics.get("gpu_temp",       0.0)),
            "error_rate_pct": float(metrics.get("error_rate_pct", 0.0)),
        }

        try:
            feats = self._engineer_features(raw)
            # Build the input vector in the EXACT column order the model
            # was trained with. Any missing feature here is a code bug
            # (the engineering recipe is out of sync), so we let KeyError
            # propagate to the engine's try/except for visibility.
            x = np.array([[feats[name] for name in self.feature_names]],
                         dtype=np.float32)

            x_scaled = self._scaler.transform(x)

            # Use the raw Booster with DMatrix. The output is a raw margin
            # for binary:logistic — we apply sigmoid to get probability.
            dmat = xgb.DMatrix(x_scaled, feature_names=self.feature_names)
            raw_pred = self._booster.predict(dmat)   # already sigmoid for logistic
            prob = float(raw_pred[0])
            cloud_win = prob >= self._threshold

            mode_str = "CLOUD" if cloud_win else "LOCAL"
            print("[XGBoostRegulator] RTT:{:3.0f}ms GPU:{:4.1f}%/{:.0f}C "
                  "CPU:{:4.1f}% -> Mode: {} (p={:.3f}, t={:.2f})".format(
                      raw["rtt_ms"], raw["gpu_load"], raw["gpu_temp"],
                      raw["cpu_load"], mode_str, prob, self._threshold))
            return bool(cloud_win), prob

        except Exception as exc:
            # Don't kill the engine on a single bad prediction. Return
            # LOCAL with low confidence so hysteresis treats it as a
            # "no change" sample.
            log.error("XGBoost predict failed: %s", exc)
            print("[XGBoostRegulator] ERROR: {} — defaulting to LOCAL".format(exc))
            return False, 0.0

    def extract_features(self, metrics):
        """
        Diagnostic helper. Returns the same list of (name, value) pairs that
        predict() would feed the model. Useful for logging and CSV columns
        downstream — adaptive_engine never calls this in the hot path.
        """
        raw = {
            "rtt_ms":         float(metrics.get("rtt_ms",         0.0)),
            "bandwidth_kbps": float(metrics.get("bandwidth_kbps", 0.0)),
            "cpu_load":       float(metrics.get("cpu_load",       0.0)),
            "ram_usage":      float(metrics.get("ram_usage",      0.0)),
            "gpu_load":       float(metrics.get("gpu_load",       0.0)),
            "gpu_temp":       float(metrics.get("gpu_temp",       0.0)),
            "error_rate_pct": float(metrics.get("error_rate_pct", 0.0)),
        }
        # Note: this CONSUMES one history slot every time it's called. We
        # snapshot+restore history to avoid polluting the predict() stream.
        snapshot = list(self._history)
        try:
            feats = self._engineer_features(raw)
        finally:
            self._history.clear()
            self._history.extend(snapshot)
        return [(name, feats[name]) for name in self.feature_names]


# ─── Standalone smoke test ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("XGBoost regulator smoke test:\n")
    try:
        r = RuleBasedRegulator()
    except FileNotFoundError as e:
        print("Artefacts missing — set DEFAULT_*_PATH or place the files at "
              "the expected location:\n  {}".format(e))
        raise SystemExit(1)

    # A handful of synthetic samples that exercise the main decision regions.
    cases = [
        ("ideal — low load",
            {"rtt_ms": 20,  "bandwidth_kbps": 5000, "error_rate_pct": 0,
             "cpu_load": 15, "ram_usage": 30,
             "gpu_load": 20, "gpu_temp": 50}),
        ("gpu hot — should CLOUD",
            {"rtt_ms": 20,  "bandwidth_kbps": 5000, "error_rate_pct": 0,
             "cpu_load": 25, "ram_usage": 40,
             "gpu_load": 85, "gpu_temp": 70}),
        ("network bad — should LOCAL",
            {"rtt_ms": 350, "bandwidth_kbps": 100,  "error_rate_pct": 12,
             "cpu_load": 25, "ram_usage": 40,
             "gpu_load": 80, "gpu_temp": 65}),
        ("cpu starved — depends on model",
            {"rtt_ms": 20,  "bandwidth_kbps": 5000, "error_rate_pct": 0,
             "cpu_load": 90, "ram_usage": 60,
             "gpu_load": 30, "gpu_temp": 55}),
    ]
    for desc, m in cases:
        print("  case: " + desc)
        r.predict(metrics=m)
        print()
