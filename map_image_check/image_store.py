"""
SQLite storage for scanned images and follow-up analysis.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .image_validation import is_valid_image_candidate

_DEFAULT_DB_NAME = "map_image_check.sqlite3"
MAX_STORED_IMAGE_BYTES = 50 * 1024 * 1024
STORED_IMAGE_MAX_LONG_SIDE = 2048
STORED_IMAGE_JPEG_QUALITY = 85
STORED_IMAGE_MIN_LONG_SIDE = 640


class ImageTooLargeForStoreError(ValueError):
    """Raised when an image exceeds the database storage size limit."""


def _bgr_from_decoded(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    channels = int(img.shape[2])
    if channels == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if channels == 3:
        return img
    raise ValueError(f"Unsupported image channel count: {channels}")


def encode_image_for_storage(
    img: np.ndarray,
    *,
    max_long_side: int = STORED_IMAGE_MAX_LONG_SIDE,
    jpeg_quality: int = STORED_IMAGE_JPEG_QUALITY,
    max_bytes: int = MAX_STORED_IMAGE_BYTES,
) -> tuple[bytes, int, int]:
    """Resize and JPEG-compress an image for SQLite storage."""
    bgr = _bgr_from_decoded(img)
    height, width = bgr.shape[:2]
    source_long = max(width, height)
    target_long = min(source_long, max(1, max_long_side))
    quality = int(jpeg_quality)

    while True:
        if source_long <= target_long:
            resized = bgr
        else:
            scale = target_long / source_long
            new_w = max(1, int(round(width * scale)))
            new_h = max(1, int(round(height * scale)))
            resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode(
            ".jpg",
            resized,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            raise ValueError("Failed to encode image for database storage.")

        data = encoded.tobytes()
        stored_h, stored_w = int(resized.shape[0]), int(resized.shape[1])
        if len(data) <= max_bytes:
            return data, stored_w, stored_h

        if quality > 55:
            quality -= 10
            continue
        if target_long > STORED_IMAGE_MIN_LONG_SIDE:
            target_long = max(
                STORED_IMAGE_MIN_LONG_SIDE,
                int(target_long * 0.85),
            )
            quality = int(jpeg_quality)
            continue

        limit_mib = max_bytes // (1024 * 1024)
        raise ImageTooLargeForStoreError(
            f"Image still exceeds database size limit ({limit_mib} MiB) after compression."
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_db_path() -> Path:
    return Path(__file__).resolve().parent.parent / _DEFAULT_DB_NAME


@dataclass(slots=True)
class StoredImageRecord:
    image_id: int
    source_path: str
    source_host: str | None
    scan_scope: str
    file_size: int
    width: int
    height: int
    sha256: str
    is_map: bool
    detector_version: str
    score: float | None
    threshold: float | None
    feature_summary_json: str
    llm_status: str | None
    llm_model_name: str | None
    llm_prompt_version: str | None
    llm_analysis_text: str | None
    llm_structured_json: str | None
    llm_is_topographic_map: bool | None


@dataclass(slots=True)
class SaveImageResult:
    record: StoredImageRecord
    inserted: bool


def _normalize_source_path(path: str | Path) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(Path(path))


def _path_lookup_variants(path: str | Path) -> tuple[str, ...]:
    raw = str(Path(path))
    normalized = _normalize_source_path(path)
    variants = {raw, normalized}
    if sys.platform == "win32":
        variants.add(raw.lower())
        variants.add(normalized.lower())
    return tuple(variants)


def lookup_path_in_index(path: str | Path, index: dict[str, int]) -> int | None:
    """Return image id if path matches a key in a path index from build_path_index()."""
    for variant in _path_lookup_variants(path):
        image_id = index.get(variant)
        if image_id is not None:
            return image_id
    if sys.platform == "win32":
        normalized = _normalize_source_path(path)
        return index.get(normalized.lower())
    return None


class ImageStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL,
                    source_host TEXT,
                    scan_scope TEXT NOT NULL,
                    image_bytes BLOB NOT NULL,
                    file_size INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS heuristic_results (
                    image_id INTEGER PRIMARY KEY,
                    is_map INTEGER NOT NULL,
                    detector_version TEXT NOT NULL,
                    score REAL,
                    threshold REAL,
                    summary_features_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS llm_results (
                    image_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    model_name TEXT,
                    prompt_version TEXT,
                    analysis_text TEXT,
                    structured_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS user_labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sha256 TEXT NOT NULL UNIQUE,
                    source_path TEXT,
                    image_id INTEGER,
                    label INTEGER NOT NULL,
                    features_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE SET NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_images_source_host ON images(source_host)"
            )
            self._migrate_db(conn)

    def _migrate_db(self, conn: sqlite3.Connection) -> None:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(llm_results)").fetchall()
        }
        if "is_topographic_map" not in columns:
            conn.execute(
                "ALTER TABLE llm_results ADD COLUMN is_topographic_map INTEGER"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_is_topographic_map "
            "ON llm_results(is_topographic_map)"
        )
        conn.execute(
            """
            UPDATE llm_results
            SET is_topographic_map = CAST(json_extract(structured_json, '$.llm_verdict') AS INTEGER)
            WHERE is_topographic_map IS NULL
              AND structured_json IS NOT NULL
              AND json_type(structured_json, '$.llm_verdict') IN ('true', 'false', 'integer')
            """
        )

    def find_existing_image_id(
        self,
        source_path: str | Path,
        sha256: str | None = None,
    ) -> int | None:
        """Return image id if the path or content hash is already stored."""
        variants = _path_lookup_variants(source_path)
        with self._connect() as conn:
            for variant in variants:
                row = conn.execute(
                    "SELECT id FROM images WHERE source_path = ?",
                    (variant,),
                ).fetchone()
                if row is not None:
                    return int(row["id"])

            if sys.platform == "win32":
                normalized = _normalize_source_path(source_path)
                row = conn.execute(
                    "SELECT id FROM images WHERE lower(source_path) = lower(?)",
                    (normalized,),
                ).fetchone()
                if row is not None:
                    return int(row["id"])

            if sha256:
                row = conn.execute(
                    "SELECT id FROM images WHERE sha256 = ?",
                    (sha256,),
                ).fetchone()
                if row is not None:
                    return int(row["id"])
        return None

    def save_detected_image(
        self,
        path: str | Path,
        *,
        scan_scope: str,
        heuristic: dict[str, Any],
    ) -> SaveImageResult:
        source_path = _normalize_source_path(path)
        if not is_valid_image_candidate(source_path):
            raise ValueError(f"Not a valid image file (text or unsupported format): {source_path}")
        try:
            on_disk_size = Path(path).stat().st_size
        except OSError as exc:
            raise ValueError(f"Cannot read image size: {source_path}") from exc
        if on_disk_size > MAX_STORED_IMAGE_BYTES:
            limit_mib = MAX_STORED_IMAGE_BYTES // (1024 * 1024)
            raise ImageTooLargeForStoreError(
                f"Image exceeds database size limit ({limit_mib} MiB): {source_path}"
            )
        data = Path(path).read_bytes()
        if not data:
            raise ValueError(f"Image is empty: {source_path}")

        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None or img.size == 0:
            raise ValueError(f"Failed to decode image for database save: {source_path}")

        try:
            data, width, height = encode_image_for_storage(img)
        except ImageTooLargeForStoreError as exc:
            raise ImageTooLargeForStoreError(
                f"{exc} Source: {source_path}"
            ) from exc

        file_size = len(data)
        sha256 = hashlib.sha256(data).hexdigest()
        existing_id = self.find_existing_image_id(source_path, sha256)
        if existing_id is not None:
            return SaveImageResult(
                record=self.get_image_record(existing_id),
                inserted=False,
            )

        source_host = infer_source_host(source_path)
        feature_summary_json = json.dumps(
            _build_feature_summary(heuristic),
            ensure_ascii=False,
            sort_keys=True,
        )
        detector_version = str(heuristic.get("detector_version") or "unknown")
        is_map = bool(heuristic.get("is_map"))
        score = _maybe_float(heuristic.get("score"))
        threshold = _maybe_float(heuristic.get("threshold"))
        now = _utc_now()

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO images (
                    source_path, source_host, scan_scope, image_bytes,
                    file_size, width, height, sha256, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_path,
                    source_host,
                    scan_scope,
                    sqlite3.Binary(data),
                    file_size,
                    width,
                    height,
                    sha256,
                    now,
                    now,
                ),
            )
            image_id = int(cur.lastrowid)

            conn.execute(
                """
                INSERT INTO heuristic_results (
                    image_id, is_map, detector_version, score, threshold,
                    summary_features_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image_id,
                    int(is_map),
                    detector_version,
                    score,
                    threshold,
                    feature_summary_json,
                    now,
                    now,
                ),
            )

        return SaveImageResult(
            record=self.get_image_record(image_id),
            inserted=True,
        )

    def update_llm_result(
        self,
        image_id: int,
        *,
        status: str,
        model_name: str | None,
        prompt_version: str | None,
        analysis_text: str | None,
        structured_json: str | dict[str, Any] | None,
        is_topographic_map: bool | None = None,
    ) -> None:
        now = _utc_now()
        if structured_json is None:
            structured_value = None
        elif isinstance(structured_json, str):
            structured_value = structured_json
        else:
            structured_value = json.dumps(
                structured_json, ensure_ascii=False, sort_keys=True
            )
        topo_value = (
            None if is_topographic_map is None else int(bool(is_topographic_map))
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_results (
                    image_id, status, model_name, prompt_version,
                    analysis_text, structured_json, is_topographic_map,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    status = excluded.status,
                    model_name = excluded.model_name,
                    prompt_version = excluded.prompt_version,
                    analysis_text = excluded.analysis_text,
                    structured_json = excluded.structured_json,
                    is_topographic_map = CASE
                        WHEN excluded.is_topographic_map IS NOT NULL
                        THEN excluded.is_topographic_map
                        ELSE llm_results.is_topographic_map
                    END,
                    updated_at = excluded.updated_at
                """,
                (
                    image_id,
                    status,
                    model_name,
                    prompt_version,
                    analysis_text,
                    structured_value,
                    topo_value,
                    now,
                    now,
                ),
            )

    _RECORD_SELECT_SQL = """
        SELECT
            i.id,
            i.source_path,
            i.source_host,
            i.scan_scope,
            i.file_size,
            i.width,
            i.height,
            i.sha256,
            hr.is_map,
            hr.detector_version,
            hr.score,
            hr.threshold,
            hr.summary_features_json,
            lr.status AS llm_status,
            lr.model_name AS llm_model_name,
            lr.prompt_version AS llm_prompt_version,
            lr.analysis_text AS llm_analysis_text,
            lr.structured_json AS llm_structured_json,
            lr.is_topographic_map AS llm_is_topographic_map
        FROM images i
        JOIN heuristic_results hr ON hr.image_id = i.id
        LEFT JOIN llm_results lr ON lr.image_id = i.id
    """

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> StoredImageRecord:
        return StoredImageRecord(
            image_id=int(row["id"]),
            source_path=str(row["source_path"]),
            source_host=row["source_host"],
            scan_scope=str(row["scan_scope"]),
            file_size=int(row["file_size"]),
            width=int(row["width"]),
            height=int(row["height"]),
            sha256=str(row["sha256"]),
            is_map=bool(row["is_map"]),
            detector_version=str(row["detector_version"]),
            score=_maybe_float(row["score"]),
            threshold=_maybe_float(row["threshold"]),
            feature_summary_json=str(row["summary_features_json"]),
            llm_status=row["llm_status"],
            llm_model_name=row["llm_model_name"],
            llm_prompt_version=row["llm_prompt_version"],
            llm_analysis_text=row["llm_analysis_text"],
            llm_structured_json=row["llm_structured_json"],
            llm_is_topographic_map=(
                None
                if row["llm_is_topographic_map"] is None
                else bool(row["llm_is_topographic_map"])
            ),
        )

    def get_image_record(self, image_id: int) -> StoredImageRecord:
        with self._connect() as conn:
            row = conn.execute(
                f"{self._RECORD_SELECT_SQL} WHERE i.id = ?",
                (image_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Image record {image_id} not found.")
        return self._row_to_record(row)

    def list_image_records(
        self,
        *,
        source_host_exact: str | None = None,
        local_hosts_only: bool = False,
        host_name_contains: str | None = None,
    ) -> list[StoredImageRecord]:
        """All stored map records, newest first. Optional filters by source host."""
        clauses: list[str] = []
        params: list[Any] = []
        if local_hosts_only:
            clauses.append("i.source_host IS NULL")
        elif source_host_exact is not None:
            clauses.append("lower(i.source_host) = lower(?)")
            params.append(source_host_exact)
        elif host_name_contains:
            needle = host_name_contains.strip().lower()
            if needle:
                clauses.append("lower(i.source_host) LIKE ?")
                params.append(f"%{needle}%")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"{self._RECORD_SELECT_SQL} {where} ORDER BY i.id DESC",
                params,
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_distinct_source_hosts(self) -> list[tuple[str | None, int]]:
        """Distinct source hosts with record counts (None = local paths)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source_host, COUNT(*) AS cnt
                FROM images
                GROUP BY source_host
                ORDER BY source_host IS NULL, lower(source_host)
                """
            ).fetchall()
        return [(row["source_host"], int(row["cnt"])) for row in rows]

    def add_path_to_index(
        self,
        index: dict[str, int],
        path: str | Path,
        image_id: int,
    ) -> None:
        """Register a newly saved path in an existing scan index."""
        for variant in _path_lookup_variants(path):
            index.setdefault(variant, image_id)
        if sys.platform == "win32":
            index.setdefault(_normalize_source_path(path).lower(), image_id)

    def build_path_index(self) -> dict[str, int]:
        """Map path lookup variants to image ids for fast scan skipping."""
        index: dict[str, int] = {}
        with self._connect() as conn:
            rows = conn.execute("SELECT id, source_path FROM images").fetchall()
        for row in rows:
            image_id = int(row["id"])
            source_path = str(row["source_path"])
            for variant in _path_lookup_variants(source_path):
                index.setdefault(variant, image_id)
            if sys.platform == "win32":
                normalized = _normalize_source_path(source_path)
                index.setdefault(normalized.lower(), image_id)
        return index

    def get_image_bytes(self, image_id: int) -> bytes:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT image_bytes FROM images WHERE id = ?",
                (image_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Image record {image_id} not found.")
        return bytes(row["image_bytes"])

    def count_images(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM images").fetchone()
        return int(row["cnt"]) if row else 0

    def db_file_size_bytes(self) -> int:
        try:
            return int(self._db_path.stat().st_size)
        except OSError:
            return 0

    def total_stored_image_bytes(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(file_size), 0) AS total FROM images"
            ).fetchone()
        return int(row["total"]) if row else 0

    def delete_image(self, image_id: int) -> bool:
        """Delete one image and related heuristic/LLM rows."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
            return cur.rowcount > 0

    def clear_all(self) -> int:
        """Delete all stored images and analysis results."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM images").fetchone()
            total = int(row["cnt"]) if row else 0
            conn.execute("DELETE FROM images")
        return total

    def save_user_label(
        self,
        *,
        sha256: str,
        label: int,
        features: dict[str, Any],
        source_path: str | None = None,
        image_id: int | None = None,
    ) -> None:
        if label not in (0, 1):
            raise ValueError("label must be 0 (not map) or 1 (map)")
        features_json = json.dumps(features, ensure_ascii=False, sort_keys=True)
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_labels (
                    sha256, source_path, image_id, label, features_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sha256) DO UPDATE SET
                    source_path = excluded.source_path,
                    image_id = excluded.image_id,
                    label = excluded.label,
                    features_json = excluded.features_json,
                    updated_at = excluded.updated_at
                """,
                (
                    sha256,
                    source_path,
                    image_id,
                    int(label),
                    features_json,
                    now,
                    now,
                ),
            )

    def get_user_label(self, sha256: str) -> int | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT label FROM user_labels WHERE sha256 = ?",
                (sha256,),
            ).fetchone()
        if row is None:
            return None
        return int(row["label"])

    def get_label_stats(self) -> tuple[int, int]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END), 0) AS maps,
                    COALESCE(SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END), 0) AS not_maps
                FROM user_labels
                """
            ).fetchone()
        if row is None:
            return 0, 0
        return int(row["maps"]), int(row["not_maps"])

    def get_training_samples(self) -> list[tuple[dict[str, float], int]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT label, features_json FROM user_labels ORDER BY id"
            ).fetchall()
        samples: list[tuple[dict[str, float], int]] = []
        for row in rows:
            try:
                features = json.loads(str(row["features_json"]))
            except json.JSONDecodeError:
                continue
            if not isinstance(features, dict):
                continue
            samples.append((features, int(row["label"])))
        return samples

    def get_features_for_image(self, image_id: int) -> dict[str, float]:
        record = self.get_image_record(image_id)
        return parse_feature_summary(record.feature_summary_json)["features"]


def format_data_size(num_bytes: int) -> str:
    """Human-readable size for UI (binary units)."""
    if num_bytes < 1024:
        return f"{num_bytes} Б"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} КиБ"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} МиБ"
    return f"{num_bytes / (1024 ** 3):.2f} ГиБ"


def infer_source_host(source_path: str) -> str | None:
    if source_path.startswith("\\\\"):
        stripped = source_path.lstrip("\\")
        host = stripped.split("\\", 1)[0].strip()
        return host or None
    return None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


ML_FILTER_ALL = "all"
ML_FILTER_GT90 = "gt90"
ML_FILTER_RANGE_80_90 = "80_90"

LLM_FILTER_ALL = "all"
LLM_FILTER_YES = "topo_yes"
LLM_FILTER_NO = "topo_no"
LLM_FILTER_UNKNOWN = "topo_unknown"

LLM_ANALYZED_STATUSES = frozenset({"completed", "gray_zone"})


def record_ml_score(record: StoredImageRecord) -> float | None:
    return parse_feature_summary(record.feature_summary_json).get("ml_score")


def record_llm_is_topographic_map(record: StoredImageRecord) -> bool | None:
    return record.llm_is_topographic_map


def record_has_llm_analysis(record: StoredImageRecord) -> bool:
    """True if LLM already produced a stored verdict (full or gray-zone)."""
    return bool(record.llm_status in LLM_ANALYZED_STATUSES)


def list_records_needing_llm_analysis(
    records: list[StoredImageRecord],
) -> list[StoredImageRecord]:
    return [record for record in records if not record_has_llm_analysis(record)]


def matches_ml_score_filter(
    ml_score: float | None,
    *,
    filter_mode: str,
) -> bool:
    """Filter ML probability (0..1). gt90: >0.90; 80_90: >0.80 and <0.90."""
    if filter_mode in ("", ML_FILTER_ALL):
        return True
    if ml_score is None:
        return False
    if filter_mode == ML_FILTER_GT90:
        return ml_score > 0.90
    if filter_mode == ML_FILTER_RANGE_80_90:
        return 0.80 < ml_score < 0.90
    return True


def matches_llm_topographic_filter(
    llm_is_topographic_map: bool | None,
    *,
    filter_mode: str,
) -> bool:
    if filter_mode in ("", LLM_FILTER_ALL):
        return True
    if filter_mode == LLM_FILTER_YES:
        return llm_is_topographic_map is True
    if filter_mode == LLM_FILTER_NO:
        return llm_is_topographic_map is False
    if filter_mode == LLM_FILTER_UNKNOWN:
        return llm_is_topographic_map is None
    return True


def _build_feature_summary(heuristic: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"features": heuristic.get("features", {})}
    for key in ("ml_score", "decision_source", "llm_verdict", "llm_response_text"):
        value = heuristic.get(key)
        if value is not None:
            payload[key] = value
    return payload


def parse_feature_summary(feature_summary_json: str) -> dict[str, Any]:
    try:
        data = json.loads(feature_summary_json)
    except json.JSONDecodeError:
        return {
            "features": {},
            "ml_score": None,
            "decision_source": None,
            "llm_verdict": None,
        }
    if not isinstance(data, dict):
        return {
            "features": {},
            "ml_score": None,
            "decision_source": None,
            "llm_verdict": None,
        }
    if "features" in data:
        return {
            "features": dict(data.get("features") or {}),
            "ml_score": _maybe_float(data.get("ml_score")),
            "decision_source": data.get("decision_source"),
            "llm_verdict": data.get("llm_verdict"),
        }
    return {
        "features": data,
        "ml_score": None,
        "decision_source": None,
        "llm_verdict": None,
    }
