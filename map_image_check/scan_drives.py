"""
CLI: scan image files under fixed drives or given roots; classify with is_terrain_map; write CSV.

Only paths classified as terrain maps are written to CSV (single column path).

Run from project root: python -m map_image_check.scan_drives --output report.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from map_image_check.detector import analyze_terrain_map
from map_image_check.hybrid_pipeline import (
    DEFAULT_T_ACCEPT,
    DEFAULT_T_LOW,
    DEFAULT_T_REJECT,
    HybridConfig,
    classify_features_offline,
    classify_image_path,
)
from map_image_check.image_store import MAX_STORED_IMAGE_BYTES
from map_image_check.ml_classifier import MapMlClassifier
from map_image_check.image_validation import (
    IMAGE_SUFFIXES as _IMAGE_SUFFIXES,
    is_text_extension,
    is_valid_image_candidate,
)

# Directory base names to prune (case-insensitive)
_EXCLUDED_DIR_NAMES_LOWER = frozenset(
    {
        "$recycle.bin",
        "system volume information",
        "winsxs",
        "system32",
        "syswow64",
        "program files",
        "program files (x86)",
        "node_modules",
        ".git",
        "packages",
        "windows",
        "appdata",
        "programdata",
        "visualelements",
        "readme",
        "media",
    }
)

_PROGRESS_EVERY = 500
_DEFAULT_OUTPUT = "map_scan_results.csv"

_EXCLUDED_PATH_SUBSTRINGS_LOWER = (
    "\\appdata\\roaming\\cursor\\user\\workspacestorage\\",
    "\\.cursor\\projects\\",
    "\\appdata\\local\\programs\\python\\",
    "\\appdata\\local\\yandex\\yandexbrowser\\user data\\default\\extensions\\",
    "\\.vscode\\extensions\\",
    "\\docs\\manual\\images\\",
    "\\tcl\\tk8.6\\demos\\images\\",
    "\\lib\\site-packages\\matplotlib\\mpl-data\\sample_data\\",
)


def _should_exclude_path(path: Path) -> bool:
    try:
        text = str(path).lower()
        parts = {part.lower() for part in path.parts}
    except OSError:
        return False
    except ValueError:
        parts = set()
    if "appdata" in parts or "programdata" in parts:
        return True
    return any(part in text for part in _EXCLUDED_PATH_SUBSTRINGS_LOWER)


# Skip tiny files / thumbnails and very large images
_MIN_FILE_BYTES = 50 * 1024
_MAX_FILE_BYTES = MAX_STORED_IMAGE_BYTES
_MIN_IMAGE_WIDTH = 200
_MIN_IMAGE_HEIGHT = 200


def _passes_size_filters(
    path: Path,
    *,
    min_file_bytes: int | None = None,
    max_file_bytes: int | None = None,
    min_width: int | None = None,
    min_height: int | None = None,
) -> bool:
    """True if file passes size and minimum dimension filters."""
    if is_text_extension(path) or not is_valid_image_candidate(path):
        return False
    min_bytes = _MIN_FILE_BYTES if min_file_bytes is None else min_file_bytes
    max_bytes = _MAX_FILE_BYTES if max_file_bytes is None else max_file_bytes
    min_w = _MIN_IMAGE_WIDTH if min_width is None else min_width
    min_h = _MIN_IMAGE_HEIGHT if min_height is None else min_height
    try:
        st = path.stat()
    except OSError:
        return False
    if st.st_size < min_bytes or st.st_size > max_bytes:
        return False
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if len(data) < min_bytes or len(data) > max_bytes:
        return False
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        return False
    h, w = int(img.shape[0]), int(img.shape[1])
    return w >= min_w and h >= min_h


def _fixed_drive_roots() -> list[Path]:
    roots: list[Path] = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        p = Path(f"{letter}:/")
        try:
            if p.exists():
                roots.append(p)
        except OSError:
            continue
    return roots


def _keep_child_dir(parent: Path, name: str) -> bool:
    n = name.lower()
    if n in _EXCLUDED_DIR_NAMES_LOWER:
        return False
    try:
        parent_parts = {x.lower() for x in parent.parts}
    except (ValueError, OSError):
        parent_parts = set()

    child_path = parent / name
    if _should_exclude_path(child_path):
        return False

    if n == "temp" and parent.name.lower() == "local":
        if "appdata" in parent_parts:
            return False

    # Skip common application resource trees that produce logos/demo images,
    # not user-owned maps.
    if n == "visualelements" and "appdata" in parent_parts:
        return False
    if n in {"readme", "media"} and {"resources", "extensions"}.issubset(parent_parts):
        return False
    if n == "_images" and {"doc", "html"}.issubset(parent_parts):
        return False
    if n == "images" and ({"docs", "manual"}.issubset(parent_parts) or "extensions" in parent_parts):
        return False
    if n == "assets" and ".cursor" in parent_parts:
        return False
    if n == "workspacestorage" and "cursor" in parent_parts:
        return False

    return True


def _walk_onerror(_err: OSError) -> None:
    """Skip folders we cannot list (permissions, etc.)."""
    return


def _walk_images(root: Path):
    """Yield Path to each image file under root."""
    root = root.resolve()
    if not root.exists():
        return
    if _should_exclude_path(root):
        return
    for dirpath, dirnames, filenames in os.walk(
        root, topdown=True, followlinks=False, onerror=_walk_onerror
    ):
        try:
            parent = Path(dirpath)
            dirnames[:] = [
                d for d in dirnames if _keep_child_dir(parent, d)
            ]
        except (OSError, ValueError):
            dirnames[:] = []

        for fname in filenames:
            fp = Path(dirpath) / fname
            if is_text_extension(fp):
                continue
            suf = fp.suffix.lower()
            if suf not in _IMAGE_SUFFIXES:
                continue
            try:
                if fp.is_file() and not _should_exclude_path(fp):
                    if is_valid_image_candidate(fp):
                        yield fp
            except OSError:
                continue


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scan images; write paths of detected terrain maps to CSV only."
    )
    p.add_argument(
        "--output",
        "-o",
        default=_DEFAULT_OUTPUT,
        help=f"CSV output path (UTF-8 with BOM for Excel). Default: {_DEFAULT_OUTPUT}",
    )
    p.add_argument(
        "--roots",
        nargs="*",
        default=None,
        metavar="PATH",
        help="If set, only scan these directories (one or more). "
        "If omitted, scan all available drive letters (A:-Z: that exist).",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=_PROGRESS_EVERY,
        help=f"Print progress every N classified images. Default: {_PROGRESS_EVERY}",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Heuristic threshold (fallback when ML not trained).",
    )
    p.add_argument(
        "--t-low",
        type=float,
        default=DEFAULT_T_LOW,
        help="Fast reject if heuristic score below this.",
    )
    p.add_argument(
        "--t-accept",
        type=float,
        default=DEFAULT_T_ACCEPT,
        help="ML probability above this accepts as map.",
    )
    p.add_argument(
        "--t-reject",
        type=float,
        default=DEFAULT_T_REJECT,
        help="ML probability below this rejects as not map.",
    )
    p.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Skip ML/LLM; use legacy is_terrain_map heuristic only.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_path = Path(args.output).resolve()

    if args.roots:
        scan_roots = [Path(r).expanduser().resolve() for r in args.roots]
    else:
        scan_roots = _fixed_drive_roots()

    if not scan_roots:
        print("No roots to scan (no drives or empty --roots).", file=sys.stderr)
        return 1

    progress_every = max(1, int(args.progress_every))
    classified = 0
    maps = 0
    skipped = 0

    from map_image_check.detector import MAP_SCORE_THRESHOLD

    hybrid_config = HybridConfig(
        heuristic_threshold=float(args.threshold or MAP_SCORE_THRESHOLD),
        t_low=float(args.t_low),
        t_accept=float(args.t_accept),
        t_reject=float(args.t_reject),
        use_llm_gray_zone=False,
    )
    ml_classifier = MapMlClassifier() if not args.heuristic_only else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["path"])

        for root in scan_roots:
            label = str(root)
            print(f"Scanning: {label}", flush=True)
            drive_files = 0
            skipped_here = 0
            try:
                for fp in _walk_images(root):
                    drive_files += 1
                    if not _passes_size_filters(fp):
                        skipped += 1
                        skipped_here += 1
                        continue
                    try:
                        if args.heuristic_only:
                            analysis = analyze_terrain_map(
                                fp, threshold=hybrid_config.heuristic_threshold
                            )
                            ok = bool(analysis and analysis.get("is_map"))
                        else:
                            decision = classify_image_path(
                                fp,
                                config=hybrid_config,
                                ml_classifier=ml_classifier,
                            )
                            ok = bool(decision and decision.is_map)
                    except Exception:
                        ok = False
                    classified += 1
                    if ok:
                        maps += 1
                        w.writerow([str(fp)])
                    if classified % progress_every == 0:
                        print(
                            f"  classified {classified} (maps so far: {maps})",
                            flush=True,
                        )
            except KeyboardInterrupt:
                print("\nInterrupted.", file=sys.stderr, flush=True)
                raise SystemExit(130) from None
            print(
                f"  done under {label}: {drive_files} image paths, "
                f"skipped (<{_MIN_FILE_BYTES // 1024} KiB or <{_MIN_IMAGE_WIDTH}x{_MIN_IMAGE_HEIGHT}px): "
                f"{skipped_here}",
                flush=True,
            )

    print(
        f"Finished. Maps in CSV: {maps}, classified (checked): {classified}, "
        f"skipped (size/pixels): {skipped}"
    )
    print(f"CSV: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
