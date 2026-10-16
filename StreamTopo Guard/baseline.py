"""Baseline calibration for Isolation Forest: pre-train on known-normal history.

Serializes a fitted baseline (Pickle or JSON) so TopoGuard can load it at
startup and enter an accurate defence state immediately, instead of waiting
for the 33-sample warm-up to complete.
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_engine import IsolationForestBaseline


class BaselineCalibrator:
    """Fit an Isolation Forest baseline on known-normal rows and serialize it."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForestBaseline(
            contamination=contamination, random_state=random_state
        )
        self.feature_names: list[str] | None = None
        self.trained_at: float | None = None
        self.sample_count: int = 0

    @staticmethod
    def _rows_to_matrix(rows: list[dict[str, float]]) -> tuple[np.ndarray, list[str]]:
        if not rows:
            raise ValueError("Cannot calibrate on an empty row set")
        feature_names = sorted({name for row in rows for name in row})
        matrix = np.asarray(
            [[float(row.get(name, 0.0)) for name in feature_names] for row in rows],
            dtype=np.float64,
        )
        if not np.isfinite(matrix).all():
            raise ValueError("Calibration data contains non-finite values")
        return matrix, feature_names

    def fit(self, rows: list[dict[str, float]]) -> dict[str, Any]:
        matrix, feature_names = self._rows_to_matrix(rows)
        required = IsolationForestBaseline.MIN_BASELINE_SAMPLES + 1
        if matrix.shape[0] < required:
            raise ValueError(
                f"Need at least {required} normal samples for calibration; received {matrix.shape[0]}"
            )
        # Fit on all but the last row; the model is now ready to score new data.
        self.model.evaluate(matrix, tuple(feature_names))
        self.feature_names = feature_names
        self.trained_at = time.time()
        self.sample_count = int(matrix.shape[0])
        return {
            "status": "trained",
            "sample_count": self.sample_count,
            "feature_count": int(matrix.shape[1]),
            "feature_names": list(self.feature_names),
            "contamination": self.contamination,
            "trained_at": self.trained_at,
        }

    @classmethod
    def from_detector(cls, model: IsolationForestBaseline) -> "BaselineCalibrator":
        """Create a persistable snapshot of an already-fitted detector."""
        if model._model is None or model._feature_count is None:
            raise RuntimeError("Isolation Forest baseline has not been trained yet")

        calibrator = cls(
            contamination=model.contamination,
            random_state=model.random_state,
        )
        calibrator.model = model
        calibrator.feature_names = list(model._feature_names or ())
        calibrator.trained_at = model._trained_at or time.time()
        calibrator.sample_count = model._baseline_samples
        return calibrator

    def save(self, path: str | Path, format: str = "pickle") -> None:
        if self.model._model is None:
            raise RuntimeError("Baseline has not been trained yet")
        path = Path(path)
        fmt = format.lower()
        if fmt == "pickle":
            payload = {
                "format": "baseline-pickle",
                "version": 1,
                "contamination": self.contamination,
                "random_state": self.random_state,
                "feature_names": self.feature_names,
                "trained_at": self.trained_at,
                "sample_count": self.sample_count,
                "feature_count": self.model._feature_count,
                "baseline_samples": self.model._baseline_samples,
                "model": self.model._model,
            }
            serialized = pickle.dumps(payload)
        elif fmt == "json":
            model_bytes = pickle.dumps(self.model._model)
            payload = {
                "format": "baseline-json",
                "version": 1,
                "contamination": self.contamination,
                "random_state": self.random_state,
                "feature_names": self.feature_names,
                "trained_at": self.trained_at,
                "sample_count": self.sample_count,
                "feature_count": self.model._feature_count,
                "baseline_samples": self.model._baseline_samples,
                "model_b64": base64.b64encode(model_bytes).decode("ascii"),
            }
            serialized = json.dumps(payload, indent=2).encode("utf-8")
        else:
            raise ValueError(f"Unsupported baseline format: {format}")

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(serialized)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def load(cls, path: str | Path) -> "BaselineCalibrator":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Baseline file not found: {path}")
        if path.suffix.lower() == ".json":
            return cls._load_json(path)
        return cls._load_pickle(path)

    @classmethod
    def _load_pickle(cls, path: Path) -> "BaselineCalibrator":
        payload = pickle.loads(path.read_bytes())
        if payload.get("format") != "baseline-pickle":
            raise ValueError("Not a TopoGuard baseline pickle file")
        calibrator = cls(
            contamination=payload.get("contamination", 0.05),
            random_state=payload.get("random_state", 42),
        )
        calibrator.model._model = payload["model"]
        calibrator.model._feature_count = payload.get("feature_count")
        calibrator.model._feature_names = tuple(payload.get("feature_names") or [])
        calibrator.model._baseline_samples = payload.get("baseline_samples", 0)
        calibrator.model._trained_at = payload.get("trained_at")
        calibrator.feature_names = payload.get("feature_names")
        calibrator.trained_at = payload.get("trained_at")
        calibrator.sample_count = payload.get("sample_count", 0)
        return calibrator

    @classmethod
    def _load_json(cls, path: Path) -> "BaselineCalibrator":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "baseline-json":
            raise ValueError("Not a TopoGuard baseline JSON file")
        calibrator = cls(
            contamination=payload.get("contamination", 0.05),
            random_state=payload.get("random_state", 42),
        )
        from sklearn.ensemble import IsolationForest

        model = pickle.loads(base64.b64decode(payload["model_b64"]))
        if not isinstance(model, IsolationForest):
            raise ValueError("Deserialized baseline model is not an IsolationForest")
        calibrator.model._model = model
        calibrator.model._feature_count = payload.get("feature_count")
        calibrator.model._feature_names = tuple(payload.get("feature_names") or [])
        calibrator.model._baseline_samples = payload.get("baseline_samples", 0)
        calibrator.model._trained_at = payload.get("trained_at")
        calibrator.feature_names = payload.get("feature_names")
        calibrator.trained_at = payload.get("trained_at")
        calibrator.sample_count = payload.get("sample_count", 0)
        return calibrator
