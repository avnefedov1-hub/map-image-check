"""
Offline heuristic classifier: terrain / topographic map vs non-map image.
Returns False if the path is not an existing file, read fails, or decode fails.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np

from .image_validation import is_text_extension, is_valid_image_candidate

# --- Tunable thresholds (calibrate on your own map / non-map samples) ---

_MAX_LONG_SIDE = 384
_CANNY_LOW = 40
_CANNY_HIGH = 140

_EDGE_DENSITY_LOW = 0.025
_EDGE_DENSITY_IDEAL = 0.095
_EDGE_DENSITY_HIGH = 0.22

_CONTOUR_DENSITY_LOW = 12.0
_CONTOUR_DENSITY_IDEAL = 45.0
_CONTOUR_DENSITY_HIGH = 130.0

_ENTROPY_HIGH = 5.0
_TEXTURE_CAP = 38.0
_STRAIGHT_LINE_CAP = 1.6

_COLORS_LOW = 6
_COLORS_IDEAL = 22
_COLORS_HIGH = 54

_FLAT_LOW = 0.00
_FLAT_IDEAL = 0.18
_FLAT_HIGH = 0.94

# Weights (sum = 1.0)
_W_EDGE = 0.08
_W_CONTOUR = 0.12
_W_LOW_ENTROPY = 0.14
_W_LOW_TEXTURE = 0.06
_W_STRAIGHT = 0.10
_W_ORTHOGONAL = 0.12
_W_COLORS_BAND = 0.14
_W_FLAT_BAND = 0.06
_W_EARTH_TONE = 0.10
_W_MUTED = 0.08

MAP_SCORE_THRESHOLD = 0.45
DETECTOR_VERSION = "heuristic-v2"
HYBRID_DETECTOR_VERSION = "hybrid-v1"

FEATURE_NAMES: tuple[str, ...] = (
    "edge_density",
    "contour_density",
    "color_entropy",
    "mean_local_std",
    "straight_line_ratio",
    "orthogonal_line_score",
    "unique_colors",
    "flat_region_ratio",
    "earth_tone_ratio",
    "muted_palette_score",
)


def features_to_vector(features: dict[str, float]) -> list[float]:
    return [float(features[name]) for name in FEATURE_NAMES]


def imread_unicode(path: str | Path) -> np.ndarray | None:
    """Load image with OpenCV; supports non-ASCII paths on Windows."""
    path = Path(path)
    if not path.is_file():
        return None
    if is_text_extension(path) or not is_valid_image_candidate(path):
        return None
    try:
        with open(path, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def _resize_long_side(bgr: np.ndarray, max_long: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long:
        return bgr
    scale = max_long / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _bandpass(value: float, low: float, ideal: float, high: float) -> float:
    if value <= low or value >= high:
        return 0.0
    if value <= ideal:
        return (value - low) / (ideal - low)
    return (high - value) / (high - ideal)


def _mean_local_std_gray(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    k = 9
    mu = cv2.blur(g, (k, k))
    mu2 = cv2.blur(g * g, (k, k))
    var = np.clip(mu2 - mu * mu, 0.0, None)
    return float(np.sqrt(var).mean())


def _edge_and_contour_features(gray: np.ndarray) -> tuple[float, float, np.ndarray]:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, _CANNY_LOW, _CANNY_HIGH)
    h, w = gray.shape
    area = float(h * w)
    edge_density = float(np.mean(edges > 0))
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    total_len = sum(float(cv2.arcLength(c, closed=False)) for c in contours)
    contour_density = total_len / (area ** 0.5 + 1e-6)
    return edge_density, contour_density, edges


def _line_segments(edges: np.ndarray) -> np.ndarray | None:
    edge_px = int(np.count_nonzero(edges))
    if edge_px < 12:
        return None
    return cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=18,
        minLineLength=12,
        maxLineGap=4,
    )


def _straight_line_ratio(edges: np.ndarray) -> float:
    lines = _line_segments(edges)
    if lines is None:
        return 0.0
    edge_px = int(np.count_nonzero(edges))
    total_len = 0.0
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        total_len += np.hypot(x2 - x1, y2 - y1)
    return float(total_len / (edge_px + 1e-6))


def _orthogonal_line_score(edges: np.ndarray) -> float:
    """Share of detected line length aligned to horizontal/vertical axes."""
    lines = _line_segments(edges)
    if lines is None:
        return 0.0
    ortho_len = 0.0
    total_len = 0.0
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = np.hypot(dx, dy)
        if length < 1.0:
            continue
        total_len += length
        angle = abs(np.degrees(np.arctan2(dy, dx)) % 180.0)
        if angle < 14.0 or angle > 166.0 or (76.0 < angle < 104.0):
            ortho_len += length
    if total_len < 1.0:
        return 0.0
    return float(ortho_len / total_len)


def _unique_colors_quantized(bgr: np.ndarray) -> int:
    small = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
    q = (small.astype(np.int32) // 64).clip(0, 3)
    idx = q[:, :, 0] * 16 + q[:, :, 1] * 4 + q[:, :, 2]
    return int(len(np.unique(idx)))


def _flat_region_ratio(gray: np.ndarray, threshold: float = 5.0) -> float:
    g = gray.astype(np.float32)
    k = 9
    mu = cv2.blur(g, (k, k))
    mu2 = cv2.blur(g * g, (k, k))
    local_std = np.sqrt(np.clip(mu2 - mu * mu, 0.0, None))
    return float(np.mean(local_std < threshold))


def _color_entropy_feature(bgr: np.ndarray) -> float:
    small = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
    q = (small.astype(np.int32) // 32).clip(0, 7)
    idx = q[:, :, 0] * 64 + q[:, :, 1] * 8 + q[:, :, 2]
    hist, _ = np.histogram(idx.ravel(), bins=512, range=(0, 512))
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p[p > 0] / s
    return float(-np.sum(p * np.log2(p + 1e-12)))


def _earth_tone_ratio(bgr: np.ndarray) -> float:
    """Pixels in typical topographic palette: green, brown, tan, water blue."""
    small = cv2.resize(bgr, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    earth = (
        ((h >= 22) & (h <= 95) & (s >= 12) & (v >= 30))
        | ((h <= 22) & (s >= 12) & (v >= 30))
        | ((h >= 95) & (h <= 135) & (s >= 12) & (v >= 22))
    )
    return float(np.mean(earth))


def _muted_palette_score(bgr: np.ndarray) -> float:
    """Printed/scanned maps use subdued saturation, unlike vivid photos."""
    small = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0
    return _bandpass(float(sat.mean()), 0.07, 0.26, 0.62)


def _features(bgr: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edge_d, contour_d, edges = _edge_and_contour_features(gray)
    ent = _color_entropy_feature(bgr)
    tex = _mean_local_std_gray(gray)
    sl = _straight_line_ratio(edges)
    ortho = _orthogonal_line_score(edges)
    uc = _unique_colors_quantized(bgr)
    fr = _flat_region_ratio(gray)
    earth = _earth_tone_ratio(bgr)
    muted = _muted_palette_score(bgr)
    return {
        "edge_density": edge_d,
        "contour_density": contour_d,
        "color_entropy": ent,
        "mean_local_std": tex,
        "straight_line_ratio": sl,
        "orthogonal_line_score": ortho,
        "unique_colors": float(uc),
        "flat_region_ratio": fr,
        "earth_tone_ratio": earth,
        "muted_palette_score": muted,
    }


def _has_cartographic_evidence(features: dict[str, float]) -> bool:
    """Require at least one strong sign of a cartographic / topographic sheet."""
    if features["orthogonal_line_score"] >= 0.30:
        return True
    if features["straight_line_ratio"] >= 0.60:
        return True
    if (
        features["earth_tone_ratio"] >= 0.20
        and features["edge_density"] >= 0.04
        and features["color_entropy"] <= 4.9
        and features["contour_density"] >= 16.0
    ):
        return True
    if (
        features["contour_density"] >= 28.0
        and features["muted_palette_score"] >= 0.35
        and features["color_entropy"] <= 5.0
        and features["mean_local_std"] <= 34.0
    ):
        return True
    return False


def _effective_threshold(threshold: float | None) -> float:
    return MAP_SCORE_THRESHOLD if threshold is None else float(threshold)


def _score(features: dict[str, float]) -> float:
    eb = _bandpass(
        features["edge_density"],
        _EDGE_DENSITY_LOW,
        _EDGE_DENSITY_IDEAL,
        _EDGE_DENSITY_HIGH,
    )
    cb = _bandpass(
        features["contour_density"],
        _CONTOUR_DENSITY_LOW,
        _CONTOUR_DENSITY_IDEAL,
        _CONTOUR_DENSITY_HIGH,
    )
    low_ent = 1.0 - min(1.0, features["color_entropy"] / _ENTROPY_HIGH)
    low_tex = 1.0 - min(1.0, features["mean_local_std"] / _TEXTURE_CAP)
    sl = min(1.0, features["straight_line_ratio"] / _STRAIGHT_LINE_CAP)
    ortho = min(1.0, features["orthogonal_line_score"])
    col_band = _bandpass(features["unique_colors"], _COLORS_LOW, _COLORS_IDEAL, _COLORS_HIGH)
    flat_band = _bandpass(features["flat_region_ratio"], _FLAT_LOW, _FLAT_IDEAL, _FLAT_HIGH)
    earth = min(1.0, features["earth_tone_ratio"] / 0.55)
    muted = features["muted_palette_score"]

    raw = (
        _W_EDGE * eb
        + _W_CONTOUR * cb
        + _W_LOW_ENTROPY * low_ent
        + _W_LOW_TEXTURE * low_tex
        + _W_STRAIGHT * sl
        + _W_ORTHOGONAL * ortho
        + _W_COLORS_BAND * col_band
        + _W_FLAT_BAND * flat_band
        + _W_EARTH_TONE * earth
        + _W_MUTED * muted
    )

    if col_band < 0.01 and flat_band < 0.01:
        raw *= 0.50
    elif col_band < 0.01:
        raw *= 0.78

    if (
        features["edge_density"] < 0.035
        and features["mean_local_std"] < 6.0
        and features["flat_region_ratio"] > 0.82
    ):
        raw *= 0.40

    if features["mean_local_std"] > 42.0 and features["flat_region_ratio"] < 0.02:
        raw *= 0.45

    if (
        features["color_entropy"] > 5.2
        and features["mean_local_std"] > 30.0
        and features["earth_tone_ratio"] < 0.12
    ):
        raw *= 0.55

    if (
        features["flat_region_ratio"] < 0.02
        and features["mean_local_std"] > 28.0
        and features["color_entropy"] > 4.2
        and features["straight_line_ratio"] < 0.75
        and features["orthogonal_line_score"] < 0.25
    ):
        raw *= 0.65

    if (
        features["edge_density"] < 0.08
        and features["contour_density"] < 22.0
        and features["straight_line_ratio"] < 0.62
        and features["orthogonal_line_score"] < 0.22
    ):
        raw *= 0.68

    if (
        features["color_entropy"] > 5.3
        and features["mean_local_std"] > 28.0
        and features["flat_region_ratio"] < 0.06
        and features["earth_tone_ratio"] < 0.15
    ):
        raw *= 0.60

    if earth >= 0.45 and (ortho >= 0.35 or cb >= 0.55):
        raw = min(1.0, raw * 1.08 + 0.04)

    if not _has_cartographic_evidence(features):
        raw *= 0.62

    return float(min(1.0, raw))


def _terrain_map_from_bgr(bgr: np.ndarray, threshold: float | None = None) -> bool:
    bgr = _resize_long_side(bgr, _MAX_LONG_SIDE)
    feats = _features(bgr)
    s = _score(feats)
    return bool(s >= _effective_threshold(threshold))


def analyze_terrain_map_bgr(
    bgr: np.ndarray, threshold: float | None = None
) -> dict[str, object]:
    effective = _effective_threshold(threshold)
    resized = _resize_long_side(bgr, _MAX_LONG_SIDE)
    feats = _features(resized)
    score = _score(feats)
    return {
        "is_map": bool(score >= effective),
        "score": float(score),
        "threshold": float(effective),
        "detector_version": DETECTOR_VERSION,
        "features": {key: float(value) for key, value in feats.items()},
        "analyzed_width": int(resized.shape[1]),
        "analyzed_height": int(resized.shape[0]),
    }


def analyze_terrain_map(
    path: str | Path, threshold: float | None = None
) -> dict[str, object] | None:
    img = imread_unicode(path)
    if img is None or img.size == 0:
        return None
    try:
        return analyze_terrain_map_bgr(img, threshold=threshold)
    except (cv2.error, ValueError, TypeError) as ex:
        warnings.warn(f"analyze_terrain_map: feature error: {ex}", stacklevel=2)
        return None


_MSG_FILE_NOT_FOUND = "Файл не найден"
_MSG_IS_DIRECTORY = "Указан каталог, а не файл"
_MSG_NOT_A_FILE = "Неверный путь: не файл"
_MSG_READ_FAILED = "Не удалось прочитать файл"
_MSG_DECODE_FAILED = "Не удалось декодировать изображение"
_MSG_LIKE_MAP = "Похоже на карту"
_MSG_NOT_LIKE_MAP = "Не похоже на карту"
_MSG_FEATURE_ERROR = "Ошибка при анализе изображения"


def is_terrain_map(path: str | Path) -> bool:
    img = imread_unicode(path)
    if img is None or img.size == 0:
        return False
    try:
        return _terrain_map_from_bgr(img)
    except (cv2.error, ValueError, TypeError) as ex:
        warnings.warn(f"is_terrain_map: feature error: {ex}", stacklevel=2)
        return False


def is_terrain_map_with_reason(path: str | Path) -> tuple[bool, str]:
    p = Path(path)
    if not p.exists():
        return False, _MSG_FILE_NOT_FOUND
    if p.is_dir():
        return False, _MSG_IS_DIRECTORY
    if not p.is_file():
        return False, _MSG_NOT_A_FILE
    try:
        with open(p, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
    except OSError:
        return False, _MSG_READ_FAILED
    if buf.size == 0:
        return False, _MSG_READ_FAILED
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return False, _MSG_DECODE_FAILED
    try:
        ok = _terrain_map_from_bgr(img)
        return (ok, _MSG_LIKE_MAP if ok else _MSG_NOT_LIKE_MAP)
    except (cv2.error, ValueError, TypeError) as ex:
        warnings.warn(f"is_terrain_map_with_reason: feature error: {ex}", stacklevel=2)
        return False, _MSG_FEATURE_ERROR


def _synthetic_topo_map(width: int = 480, height: int = 360) -> np.ndarray:
    """Topo-like sheet: hypsometric fills, grid, brown contour lines."""
    img = np.full((height, width, 3), (210, 225, 195), dtype=np.uint8)
    rng = np.random.RandomState(7)

    for y0 in range(0, height, 60):
        cv2.line(img, (0, y0), (width, y0), (120, 120, 120), 1)
    for x0 in range(0, width, 60):
        cv2.line(img, (x0, 0), (x0, height), (120, 120, 120), 1)

    zones = [
        ((0, 0, 160, 120), (180, 210, 170)),
        ((160, 0, 320, 120), (200, 230, 190)),
        ((0, 120, 160, 240), (170, 200, 230)),
        ((160, 120, 320, 240), (190, 215, 175)),
        ((0, 240, width, height), (160, 190, 220)),
    ]
    for (x1, y1, x2, y2), color in zones:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    yy, xx = np.mgrid[0:height, 0:width]
    for level in range(4, 14):
        phase = rng.uniform(0, 6.28)
        field = np.sin(xx / 38.0 + phase) + 0.7 * np.cos(yy / 29.0 + phase * 0.6)
        mask = np.abs(field - (level * 0.18)) < 0.06
        img[mask] = (40, 80, 140)

    return img


def _smoke_test() -> None:
    import os
    import tempfile

    missing = Path(__file__).parent / "__no_such_image_for_smoke__.png"
    ok_miss, msg_miss = is_terrain_map_with_reason(missing)
    assert ok_miss is False
    assert msg_miss == _MSG_FILE_NOT_FOUND, msg_miss

    grid = np.ones((240, 320, 3), dtype=np.uint8) * 250
    colors = [
        (40, 40, 40), (180, 220, 180), (200, 230, 255),
        (255, 240, 200), (220, 200, 180), (200, 255, 200),
        (180, 200, 220), (240, 210, 210), (210, 240, 230),
        (230, 230, 200), (190, 210, 190), (220, 220, 240),
    ]
    cell_h, cell_w = 40, 40
    ci = 0
    for y0 in range(0, 240, cell_h):
        for x0 in range(0, 320, cell_w):
            grid[y0:y0 + cell_h, x0:x0 + cell_w] = colors[ci % len(colors)]
            ci += 1
    for i in range(0, 320, cell_w):
        grid[:, i:i + 2] = (40, 40, 40)
    for j in range(0, 240, cell_h):
        grid[j:j + 2, :] = (40, 40, 40)

    fd, grid_path = tempfile.mkstemp(suffix=".png")
    try:
        os.close(fd)
        cv2.imwrite(grid_path, grid)
        g = is_terrain_map(grid_path)
        print(f"is_terrain_map(synthetic grid): {g}")
    finally:
        try:
            os.unlink(grid_path)
        except OSError:
            pass

    topo = _synthetic_topo_map()
    fd_topo, topo_path = tempfile.mkstemp(suffix=".png")
    try:
        os.close(fd_topo)
        cv2.imwrite(topo_path, topo)
        g_topo = is_terrain_map(topo_path)
        details = analyze_terrain_map(topo_path)
        print(f"is_terrain_map(synthetic topo): {g_topo}, score={details['score']:.3f}")
    finally:
        try:
            os.unlink(topo_path)
        except OSError:
            pass

    yy, xx = np.meshgrid(
        np.linspace(0, 255, 256), np.linspace(0, 255, 256), indexing="ij"
    )
    grad = np.stack([yy, xx, np.full_like(yy, 128)], axis=2).astype(np.uint8)
    fd2, grad_path = tempfile.mkstemp(suffix=".png")
    try:
        os.close(fd2)
        cv2.imwrite(grad_path, grad)
        g2 = is_terrain_map(grad_path)
        print(f"is_terrain_map(smooth gradient): {g2}")
    finally:
        try:
            os.unlink(grad_path)
        except OSError:
            pass

    rng = np.random.RandomState(42)
    noise = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    fd3, noise_path = tempfile.mkstemp(suffix=".png")
    try:
        os.close(fd3)
        cv2.imwrite(noise_path, noise)
        g3 = is_terrain_map(noise_path)
        print(f"is_terrain_map(random noise): {g3}")
    finally:
        try:
            os.unlink(noise_path)
        except OSError:
            pass

    solid = np.full((256, 256, 3), 255, dtype=np.uint8)
    fd4, solid_path = tempfile.mkstemp(suffix=".png")
    try:
        os.close(fd4)
        cv2.imwrite(solid_path, solid)
        g4 = is_terrain_map(solid_path)
        print(f"is_terrain_map(solid white): {g4}")
    finally:
        try:
            os.unlink(solid_path)
        except OSError:
            pass

    print(f"  grid={g}, topo={g_topo}, gradient={g2}, noise={g3}, solid={g4}")
    assert g is True
    assert g_topo is True
    assert g2 is False
    assert g3 is False
    assert g4 is False
    print("smoke test finished (no crash).")


if __name__ == "__main__":
    _smoke_test()
