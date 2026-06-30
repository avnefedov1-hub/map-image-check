"""Tests for database listing and scan path index."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from map_image_check.image_store import (
    ImageStore,
    ML_FILTER_GT90,
    ML_FILTER_RANGE_80_90,
    lookup_path_in_index,
    matches_ml_score_filter,
)


class ImageStoreViewerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmpdir.cleanup)
        self._store = ImageStore(Path(self._tmpdir.name) / "test.sqlite3")
        self._image_path = Path(self._tmpdir.name) / "map.png"
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(self._image_path), img)

    def tearDown(self) -> None:
        self._store = None  # type: ignore[assignment]

    def _save_map(self, path: Path | None = None) -> int:
        target = path or self._image_path
        saved = self._store.save_detected_image(
            target,
            scan_scope="test",
            heuristic={
                "is_map": True,
                "score": 0.9,
                "threshold": 0.5,
                "detector_version": "test",
            },
        )
        self.assertTrue(saved.inserted)
        return saved.record.image_id

    def test_list_image_records_returns_saved_maps(self) -> None:
        image_id = self._save_map()
        records = self._store.list_image_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].image_id, image_id)
        self.assertEqual(records[0].source_path, str(self._image_path.resolve()))

    def test_build_path_index_and_lookup(self) -> None:
        image_id = self._save_map()
        index = self._store.build_path_index()
        self.assertEqual(lookup_path_in_index(self._image_path, index), image_id)

        self._store.add_path_to_index(index, self._image_path, image_id)
        self.assertEqual(lookup_path_in_index(self._image_path, index), image_id)


    def test_list_distinct_source_hosts(self) -> None:
        image_id = self._save_map()
        hosts = self._store.list_distinct_source_hosts()
        self.assertEqual(len(hosts), 1)
        self.assertIsNone(hosts[0][0])
        self.assertEqual(hosts[0][1], 1)

        filtered = self._store.list_image_records(local_hosts_only=True)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].image_id, image_id)

        self.assertEqual(self._store.list_image_records(source_host_exact="missing"), [])

    def test_matches_ml_score_filter(self) -> None:
        self.assertTrue(matches_ml_score_filter(0.91, filter_mode=ML_FILTER_GT90))
        self.assertFalse(matches_ml_score_filter(0.90, filter_mode=ML_FILTER_GT90))
        self.assertTrue(matches_ml_score_filter(0.85, filter_mode=ML_FILTER_RANGE_80_90))
        self.assertFalse(matches_ml_score_filter(0.90, filter_mode=ML_FILTER_RANGE_80_90))
        self.assertFalse(matches_ml_score_filter(0.80, filter_mode=ML_FILTER_RANGE_80_90))
        self.assertFalse(matches_ml_score_filter(None, filter_mode=ML_FILTER_GT90))


if __name__ == "__main__":
    unittest.main()
