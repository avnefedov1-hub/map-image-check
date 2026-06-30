"""Tests for hybrid map classification pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from map_image_check.detector import FEATURE_NAMES
from map_image_check.hybrid_pipeline import HybridConfig, classify_features_offline
from map_image_check.ml_classifier import MapMlClassifier
from map_image_check.image_store import (
    ImageStore,
    LLM_FILTER_NO,
    LLM_FILTER_UNKNOWN,
    LLM_FILTER_YES,
    StoredImageRecord,
    encode_image_for_storage,
    list_records_needing_llm_analysis,
    matches_llm_topographic_filter,
    parse_feature_summary,
    record_has_llm_analysis,
)


def _sample_features(seed: float) -> dict[str, float]:
    return {name: seed + index * 0.01 for index, name in enumerate(FEATURE_NAMES)}


class HybridPipelineTests(unittest.TestCase):
    def test_heuristic_low_rejects_before_ml(self) -> None:
        config = HybridConfig(t_low=0.40, heuristic_threshold=0.45)
        decision = classify_features_offline(
            _sample_features(0.1),
            heuristic_score=0.25,
            config=config,
        )
        self.assertFalse(decision.is_map)
        self.assertEqual(decision.decision_source, "heuristic_low")
        self.assertIsNone(decision.ml_score)

    def test_ml_accept_and_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.joblib"
            meta_path = Path(tmp) / "meta.json"
            clf = MapMlClassifier(model_path=model_path, meta_path=meta_path)

            map_features = _sample_features(0.8)
            not_map_features = _sample_features(0.2)
            samples = [(map_features, 1)] * 10 + [(not_map_features, 0)] * 10
            result = clf.train(samples)
            self.assertTrue(result.success)
            self.assertTrue(clf.is_ready())

            config = HybridConfig(
                t_low=0.20,
                t_accept=0.60,
                t_reject=0.40,
                heuristic_threshold=0.45,
            )
            accept = classify_features_offline(
                map_features,
                heuristic_score=0.50,
                config=config,
                ml_classifier=clf,
            )
            reject = classify_features_offline(
                not_map_features,
                heuristic_score=0.50,
                config=config,
                ml_classifier=clf,
            )
            self.assertTrue(accept.is_map)
            self.assertEqual(accept.decision_source, "ml")
            self.assertFalse(reject.is_map)
            self.assertEqual(reject.decision_source, "ml")

    def test_parse_feature_summary_roundtrip(self) -> None:
        payload = {
            "features": _sample_features(0.5),
            "ml_score": 0.81,
            "decision_source": "ml",
        }
        import json

        parsed = parse_feature_summary(json.dumps(payload, sort_keys=True))
        self.assertAlmostEqual(parsed["ml_score"], 0.81)
        self.assertEqual(parsed["decision_source"], "ml")
        self.assertIn("edge_density", parsed["features"])

    def test_encode_image_for_storage_reduces_large_image(self) -> None:
        img = np.zeros((4000, 3000, 3), dtype=np.uint8)
        data, width, height = encode_image_for_storage(img, max_long_side=2048)
        self.assertLessEqual(max(width, height), 2048)
        self.assertLess(len(data), 4000 * 3000)
        self.assertTrue(data.startswith(b"\xff\xd8"))

    def test_user_labels_and_training_samples(self) -> None:
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        tmp.close()
        db_path = Path(tmp.name)
        store = ImageStore(db_path)
        try:
            features = _sample_features(0.42)
            store.save_user_label(
                sha256="abc123",
                label=1,
                features=features,
                source_path="/tmp/map.png",
            )
            store.save_user_label(
                sha256="def456",
                label=0,
                features=_sample_features(0.11),
                source_path="/tmp/not_map.png",
            )
            maps, not_maps = store.get_label_stats()
            self.assertEqual(maps, 1)
            self.assertEqual(not_maps, 1)
            samples = store.get_training_samples()
            self.assertEqual(len(samples), 2)
            self.assertEqual(store.get_user_label("abc123"), 1)
        finally:
            del store
            try:
                db_path.unlink()
            except OSError:
                pass


    def test_matches_llm_topographic_filter(self) -> None:
        self.assertTrue(
            matches_llm_topographic_filter(True, filter_mode=LLM_FILTER_YES)
        )
        self.assertFalse(
            matches_llm_topographic_filter(False, filter_mode=LLM_FILTER_YES)
        )
        self.assertTrue(
            matches_llm_topographic_filter(None, filter_mode=LLM_FILTER_UNKNOWN)
        )
        self.assertFalse(
            matches_llm_topographic_filter(True, filter_mode=LLM_FILTER_NO)
        )

    def test_record_has_llm_analysis(self) -> None:
        analyzed = StoredImageRecord(
            image_id=1,
            source_path="/tmp/a.png",
            source_host=None,
            scan_scope="test",
            file_size=1,
            width=1,
            height=1,
            sha256="abc",
            is_map=True,
            detector_version="v1",
            score=0.5,
            threshold=0.4,
            feature_summary_json="{}",
            llm_status="completed",
            llm_model_name="m",
            llm_prompt_version="p",
            llm_analysis_text="text",
            llm_structured_json="{}",
            llm_is_topographic_map=True,
        )
        pending = StoredImageRecord(
            image_id=2,
            source_path="/tmp/b.png",
            source_host=None,
            scan_scope="test",
            file_size=1,
            width=1,
            height=1,
            sha256="def",
            is_map=True,
            detector_version="v1",
            score=0.5,
            threshold=0.4,
            feature_summary_json="{}",
            llm_status=None,
            llm_model_name=None,
            llm_prompt_version=None,
            llm_analysis_text=None,
            llm_structured_json=None,
            llm_is_topographic_map=None,
        )
        gray = StoredImageRecord(
            image_id=3,
            source_path="/tmp/c.png",
            source_host=None,
            scan_scope="test",
            file_size=1,
            width=1,
            height=1,
            sha256="ghi",
            is_map=True,
            detector_version="v1",
            score=0.5,
            threshold=0.4,
            feature_summary_json="{}",
            llm_status="gray_zone",
            llm_model_name="m",
            llm_prompt_version="p",
            llm_analysis_text="text",
            llm_structured_json="{}",
            llm_is_topographic_map=True,
        )
        self.assertTrue(record_has_llm_analysis(analyzed))
        self.assertTrue(record_has_llm_analysis(gray))
        self.assertFalse(record_has_llm_analysis(pending))
        self.assertEqual(
            list_records_needing_llm_analysis([analyzed, pending, gray]),
            [pending],
        )

    def test_update_llm_result_persists_analysis(self) -> None:
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        tmp.close()
        db_path = Path(tmp.name)
        store = ImageStore(db_path)
        try:
            img = np.zeros((200, 300, 3), dtype=np.uint8)
            ok, encoded = cv2.imencode(".png", img)
            self.assertTrue(ok)
            png_path = Path(tmp.name + ".png")
            png_path.write_bytes(encoded.tobytes())
            saved = store.save_detected_image(
                png_path,
                scan_scope="test",
                heuristic={
                    "is_map": True,
                    "score": 0.7,
                    "threshold": 0.45,
                    "detector_version": "test",
                    "features": _sample_features(0.5),
                },
            )
            store.update_llm_result(
                saved.record.image_id,
                status="completed",
                model_name="qwen2.5vl:7b",
                prompt_version="map-review-v2",
                analysis_text="1. Тип\nКарта",
                structured_json={"api": "chat"},
                is_topographic_map=True,
            )
            record = store.get_image_record(saved.record.image_id)
            self.assertEqual(record.llm_status, "completed")
            self.assertEqual(record.llm_model_name, "qwen2.5vl:7b")
            self.assertEqual(record.llm_analysis_text, "1. Тип\nКарта")
            self.assertTrue(record.llm_is_topographic_map)
        finally:
            del store
            try:
                db_path.unlink()
            except OSError:
                pass
            try:
                png_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
