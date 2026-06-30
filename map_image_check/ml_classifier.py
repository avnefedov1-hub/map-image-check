"""
Trainable logistic-regression classifier on detector feature vectors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .detector import FEATURE_NAMES, features_to_vector

_CLASSIFIER_FILENAME = "map_classifier.joblib"
_META_FILENAME = "map_classifier_meta.json"
_MIN_SAMPLES_PER_CLASS = 8
_MODEL_VERSION = "logreg-v1"


def default_classifier_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def default_classifier_path() -> Path:
    return default_classifier_dir() / _CLASSIFIER_FILENAME


def default_classifier_meta_path() -> Path:
    return default_classifier_dir() / _META_FILENAME


@dataclass(slots=True)
class TrainResult:
    success: bool
    message: str
    map_count: int = 0
    not_map_count: int = 0
    holdout_accuracy: float | None = None


@dataclass(slots=True)
class ClassifierMeta:
    version: str
    trained_at: str
    map_count: int
    not_map_count: int
    holdout_accuracy: float | None


class MapMlClassifier:
    def __init__(
        self,
        model_path: str | Path | None = None,
        meta_path: str | Path | None = None,
    ) -> None:
        self._model_path = Path(model_path) if model_path else default_classifier_path()
        self._meta_path = Path(meta_path) if meta_path else default_classifier_meta_path()
        self._pipeline: Pipeline | None = None
        self._meta: ClassifierMeta | None = None
        self.load()

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def meta(self) -> ClassifierMeta | None:
        return self._meta

    def is_ready(self) -> bool:
        return self._pipeline is not None

    def load(self) -> bool:
        self._pipeline = None
        self._meta = None
        if not self._model_path.is_file():
            return False
        try:
            self._pipeline = joblib.load(self._model_path)
        except Exception:
            return False
        if self._meta_path.is_file():
            try:
                raw = json.loads(self._meta_path.read_text(encoding="utf-8"))
                self._meta = ClassifierMeta(
                    version=str(raw.get("version") or _MODEL_VERSION),
                    trained_at=str(raw.get("trained_at") or ""),
                    map_count=int(raw.get("map_count") or 0),
                    not_map_count=int(raw.get("not_map_count") or 0),
                    holdout_accuracy=_maybe_float(raw.get("holdout_accuracy")),
                )
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                self._meta = None
        return self._pipeline is not None

    def predict_proba(self, features: dict[str, float]) -> float | None:
        if self._pipeline is None:
            return None
        vector = np.array([features_to_vector(features)], dtype=np.float64)
        proba = self._pipeline.predict_proba(vector)[0]
        classes = list(self._pipeline.classes_)
        if 1 not in classes:
            return None
        return float(proba[classes.index(1)])

    def train(self, samples: list[tuple[dict[str, float], int]]) -> TrainResult:
        maps = sum(1 for _, label in samples if label == 1)
        not_maps = sum(1 for _, label in samples if label == 0)
        if maps < _MIN_SAMPLES_PER_CLASS or not_maps < _MIN_SAMPLES_PER_CLASS:
            return TrainResult(
                success=False,
                message=(
                    f"Нужно минимум {_MIN_SAMPLES_PER_CLASS} примеров каждого класса. "
                    f"Сейчас: карт={maps}, не карт={not_maps}."
                ),
                map_count=maps,
                not_map_count=not_maps,
            )

        x_rows = [features_to_vector(features) for features, _label in samples]
        y_rows = [int(label) for _features, label in samples]
        x = np.array(x_rows, dtype=np.float64)
        y = np.array(y_rows, dtype=np.int32)

        holdout_accuracy: float | None = None
        if len(samples) >= 20:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.25, random_state=42, stratify=y
            )
        else:
            x_train, y_train = x, y
            x_test, y_test = None, None

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                ),
            ]
        )
        pipeline.fit(x_train, y_train)

        if x_test is not None and y_test is not None and len(y_test) > 0:
            holdout_accuracy = float(pipeline.score(x_test, y_test))

        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self._model_path)
        trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta = ClassifierMeta(
            version=_MODEL_VERSION,
            trained_at=trained_at,
            map_count=maps,
            not_map_count=not_maps,
            holdout_accuracy=holdout_accuracy,
        )
        self._meta_path.write_text(
            json.dumps(
                {
                    "version": meta.version,
                    "trained_at": meta.trained_at,
                    "map_count": meta.map_count,
                    "not_map_count": meta.not_map_count,
                    "holdout_accuracy": meta.holdout_accuracy,
                    "feature_names": list(FEATURE_NAMES),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._pipeline = pipeline
        self._meta = meta

        accuracy_text = (
            f", точность на hold-out: {holdout_accuracy:.0%}"
            if holdout_accuracy is not None
            else ""
        )
        return TrainResult(
            success=True,
            message=(
                f"Модель обучена: карт={maps}, не карт={not_maps}{accuracy_text}."
            ),
            map_count=maps,
            not_map_count=not_maps,
            holdout_accuracy=holdout_accuracy,
        )


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
