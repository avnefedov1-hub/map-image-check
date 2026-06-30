"""
Reject text and non-image files before map detection.
"""

from __future__ import annotations

from pathlib import Path

# Allowed image extensions (lowercase, with dot)
IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
)

# Text and document extensions — never scan, even if misnamed elsewhere
TEXT_FILE_SUFFIXES = frozenset(
    {
        ".txt",
        ".text",
        ".log",
        ".csv",
        ".tsv",
        ".md",
        ".markdown",
        ".json",
        ".jsonl",
        ".xml",
        ".html",
        ".htm",
        ".xhtml",
        ".css",
        ".js",
        ".mjs",
        ".ts",
        ".tsx",
        ".jsx",
        ".py",
        ".pyw",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".ini",
        ".cfg",
        ".conf",
        ".config",
        ".yaml",
        ".yml",
        ".toml",
        ".properties",
        ".env",
        ".rtf",
        ".bat",
        ".cmd",
        ".ps1",
        ".psm1",
        ".sh",
        ".bash",
        ".zsh",
        ".sql",
        ".doc",
        ".docx",
        ".docm",
        ".odt",
        ".pdf",
        ".epub",
        ".mobi",
        ".tex",
        ".rst",
        ".adoc",
        ".srt",
        ".vtt",
        ".nfo",
        ".diz",
        ".reg",
        ".lst",
        ".inf",
    }
)

_HEAD_READ_BYTES = 8192
_TEXT_SAMPLE_BYTES = 4096


def file_suffix_lower(path: str | Path) -> str:
    return Path(path).suffix.lower()


def is_text_extension(path: str | Path) -> bool:
    """True if the path has a known text/document extension."""
    suffix = file_suffix_lower(path)
    if suffix in TEXT_FILE_SUFFIXES:
        return True
    # Double extension: report.txt.png -> still image suffix wins in caller,
    # but report.png.txt must be rejected.
    name = Path(path).name.lower()
    for text_suf in TEXT_FILE_SUFFIXES:
        if name.endswith(text_suf):
            return True
    return False


def is_allowed_image_extension(path: str | Path) -> bool:
    suffix = file_suffix_lower(path)
    return suffix in IMAGE_SUFFIXES and not is_text_extension(path)


def has_image_magic(data: bytes) -> bool:
    """True if bytes start with a known raster image signature."""
    if len(data) < 12:
        return False
    if data.startswith(b"\xff\xd8\xff"):
        return True
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if data.startswith((b"GIF87a", b"GIF89a")):
        return True
    if data.startswith(b"BM"):
        return True
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return True
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return True
    return False


def looks_like_text_content(data: bytes) -> bool:
    """Heuristic: plain UTF-8/ASCII text (e.g. a .jpg file that is really a log)."""
    if not data:
        return True
    sample = data[:_TEXT_SAMPLE_BYTES]
    if b"\x00" in sample:
        return False
    try:
        text = sample.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = sample.decode("latin-1")
        except UnicodeDecodeError:
            return False
    if not text:
        return True
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\r\n\t")
    return printable / len(text) >= 0.97


def read_file_head(path: Path, max_bytes: int = _HEAD_READ_BYTES) -> bytes | None:
    try:
        with open(path, "rb") as handle:
            return handle.read(max_bytes)
    except OSError:
        return None


def is_valid_image_candidate(path: str | Path) -> bool:
    """
    True only for allowed image extensions with real image magic bytes.
    Text extensions and text-like content are always rejected.
    """
    path = Path(path)
    if not path.is_file():
        return False
    if is_text_extension(path):
        return False
    if not is_allowed_image_extension(path):
        return False
    head = read_file_head(path)
    if head is None or not head:
        return False
    if looks_like_text_content(head):
        return False
    return has_image_magic(head)
