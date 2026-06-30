"""
Two-stage map classification: heuristic pre-filter + optional ML + optional LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .detector import (
    DETECTOR_VERSION,
    HYBRID_DETECTOR_VERSION,
    MAP_SCORE_THRESHOLD,
    analyze_terrain_map,
)
from .ml_classifier import MapMlClassifier


DEFAULT_T_LOW = 0.30
DEFAULT_T_ACCEPT = 0.65
DEFAULT_T_REJECT = 0.35


@dataclass(slots=True)
class HybridConfig:
    heuristic_threshold: float = MAP_SCORE_THRESHOLD
    t_low: float = DEFAULT_T_LOW
    t_accept: float = DEFAULT_T_ACCEPT
    t_reject: float = DEFAULT_T_REJECT
    use_llm_gray_zone: bool = False


@dataclass(slots=True)
class HybridDecision:
    is_map: bool
    heuristic_score: float
    ml_score: float | None
    decision_source: str
    features: dict[str, float]
    detector_version: str
    threshold: float
    llm_verdict: bool | None = None

    def as_heuristic_payload(self) -> dict[str, Any]:
        return {
            "is_map": self.is_map,
            "score": self.heuristic_score,
            "threshold": self.threshold,
            "detector_version": self.detector_version,
            "features": self.features,
            "ml_score": self.ml_score,
            "decision_source": self.decision_source,
            "llm_verdict": self.llm_verdict,
        }


def classify_image_path(
    path: str | Path,
    *,
    config: HybridConfig,
    ml_classifier: MapMlClassifier | None = None,
    llm_classify: Callable[[bytes], bool] | None = None,
) -> HybridDecision | None:
    analysis = analyze_terrain_map(path, threshold=config.heuristic_threshold)
    if analysis is None:
        return None

    features = dict(analysis["features"])
    heuristic_score = float(analysis["score"])
    threshold = float(analysis["threshold"])

    if heuristic_score < config.t_low:
        return HybridDecision(
            is_map=False,
            heuristic_score=heuristic_score,
            ml_score=None,
            decision_source="heuristic_low",
            features=features,
            detector_version=DETECTOR_VERSION,
            threshold=threshold,
        )

    ml = ml_classifier if ml_classifier is not None else MapMlClassifier()
    if ml.is_ready():
        ml_score = ml.predict_proba(features)
        if ml_score is not None:
            if ml_score >= config.t_accept:
                return HybridDecision(
                    is_map=True,
                    heuristic_score=heuristic_score,
                    ml_score=ml_score,
                    decision_source="ml",
                    features=features,
                    detector_version=HYBRID_DETECTOR_VERSION,
                    threshold=threshold,
                )
            if ml_score <= config.t_reject:
                return HybridDecision(
                    is_map=False,
                    heuristic_score=heuristic_score,
                    ml_score=ml_score,
                    decision_source="ml",
                    features=features,
                    detector_version=HYBRID_DETECTOR_VERSION,
                    threshold=threshold,
                )

            if config.use_llm_gray_zone and llm_classify is not None:
                try:
                    image_bytes = Path(path).read_bytes()
                    llm_ok = bool(llm_classify(image_bytes))
                    return HybridDecision(
                        is_map=llm_ok,
                        heuristic_score=heuristic_score,
                        ml_score=ml_score,
                        decision_source="llm",
                        features=features,
                        detector_version=HYBRID_DETECTOR_VERSION,
                        threshold=threshold,
                        llm_verdict=llm_ok,
                    )
                except OSError:
                    pass

            return HybridDecision(
                is_map=False,
                heuristic_score=heuristic_score,
                ml_score=ml_score,
                decision_source="ml_gray_reject",
                features=features,
                detector_version=HYBRID_DETECTOR_VERSION,
                threshold=threshold,
            )

    is_map = heuristic_score >= threshold
    return HybridDecision(
        is_map=is_map,
        heuristic_score=heuristic_score,
        ml_score=None,
        decision_source="heuristic",
        features=features,
        detector_version=DETECTOR_VERSION,
        threshold=threshold,
    )


def classify_features_offline(
    features: dict[str, float],
    *,
    heuristic_score: float,
    config: HybridConfig,
    ml_classifier: MapMlClassifier | None = None,
) -> HybridDecision:
    """Classify using precomputed features (for tests)."""
    threshold = config.heuristic_threshold
    if heuristic_score < config.t_low:
        return HybridDecision(
            is_map=False,
            heuristic_score=heuristic_score,
            ml_score=None,
            decision_source="heuristic_low",
            features=features,
            detector_version=DETECTOR_VERSION,
            threshold=threshold,
        )

    ml = ml_classifier if ml_classifier is not None else MapMlClassifier()
    if ml.is_ready():
        ml_score = ml.predict_proba(features)
        if ml_score is not None:
            if ml_score >= config.t_accept:
                return HybridDecision(
                    is_map=True,
                    heuristic_score=heuristic_score,
                    ml_score=ml_score,
                    decision_source="ml",
                    features=features,
                    detector_version=HYBRID_DETECTOR_VERSION,
                    threshold=threshold,
                )
            if ml_score <= config.t_reject:
                return HybridDecision(
                    is_map=False,
                    heuristic_score=heuristic_score,
                    ml_score=ml_score,
                    decision_source="ml",
                    features=features,
                    detector_version=HYBRID_DETECTOR_VERSION,
                    threshold=threshold,
                )

    is_map = heuristic_score >= threshold
    return HybridDecision(
        is_map=is_map,
        heuristic_score=heuristic_score,
        ml_score=ml.predict_proba(features) if ml.is_ready() else None,
        decision_source="heuristic",
        features=features,
        detector_version=DETECTOR_VERSION,
        threshold=threshold,
    )
