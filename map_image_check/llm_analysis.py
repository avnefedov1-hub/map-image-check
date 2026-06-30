"""
Manual detailed image analysis through a local Ollama endpoint.
"""

from __future__ import annotations

import base64
import json
import socket
from typing import Any
from urllib import error, parse, request

import cv2
import numpy as np

from .image_store import StoredImageRecord

DEFAULT_LOCAL_MODEL = "llama3.2-vision"
DEFAULT_PROMPT_VERSION = "map-review-v1"
DEFAULT_YES_NO_PROMPT_VERSION = "map-yes-no-v1"
DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_ENDPOINT = f"{DEFAULT_OLLAMA_BASE}/api/generate"
DEFAULT_LLM_GRAY_TIMEOUT = 60
DEFAULT_LLM_ANALYSIS_TIMEOUT = 180
DEFAULT_LLM_MAX_LONG_SIDE = 1536
DEFAULT_LLM_JPEG_QUALITY = 85


def normalize_ollama_base_url(base_url: str) -> str:
    text = (base_url or DEFAULT_OLLAMA_BASE).strip().rstrip("/")
    if not text:
        return DEFAULT_OLLAMA_BASE
    parsed = parse.urlparse(text)
    if not parsed.scheme:
        text = f"http://{text}"
    return text.rstrip("/")


def ollama_chat_url(base_url: str) -> str:
    return f"{normalize_ollama_base_url(base_url)}/api/chat"


def ollama_tags_url(base_url: str) -> str:
    return f"{normalize_ollama_base_url(base_url)}/api/tags"


def build_chat_payload(
    *,
    model_name: str,
    content: str,
    image_bytes: bytes,
    stream: bool = False,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": content,
                "images": [base64.b64encode(image_bytes).decode("ascii")],
            }
        ],
        "stream": stream,
    }


def extract_chat_response_text(data: dict[str, Any]) -> str:
    message = data.get("message")
    if isinstance(message, dict):
        text = str(message.get("content") or "").strip()
        if text:
            return text
    return str(data.get("response") or "").strip()


def format_ollama_error(exc: BaseException, *, base_url: str = DEFAULT_OLLAMA_BASE) -> str:
    normalized = normalize_ollama_base_url(base_url)
    if isinstance(exc, error.HTTPError):
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except OSError:
            body = ""
        api_message = ""
        if body:
            try:
                payload = json.loads(body)
                api_message = str(payload.get("error") or "").strip()
            except json.JSONDecodeError:
                api_message = body.strip()[:400]
        if api_message:
            lowered = api_message.lower()
            if "not found" in lowered and "model" in lowered:
                return (
                    f"Модель не установлена в Ollama: {api_message}. "
                    f"Выполните: ollama pull <имя-модели>"
                )
            return f"Ollama вернула ошибку HTTP {exc.code}: {api_message}"
        return f"Ollama вернула ошибку HTTP {exc.code} ({normalized})."

    reason = getattr(exc, "reason", None)
    if isinstance(exc, error.URLError):
        inner = exc.reason
        if isinstance(inner, ConnectionRefusedError):
            return (
                f"Ollama не отвечает на {normalized}. "
                "Проверьте, что приложение Ollama запущено."
            )
        if isinstance(inner, socket.timeout):
            return f"Таймаут подключения к Ollama ({normalized})."
        if isinstance(inner, TimeoutError):
            return f"Таймаут запроса к Ollama ({normalized})."
        if isinstance(inner, OSError):
            return f"Ollama недоступна ({normalized}): {inner}"
        if reason:
            return f"Ollama недоступна ({normalized}): {reason}"

    return f"Ollama недоступна ({normalized}): {exc}"


def _http_json_request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        raise RuntimeError(format_ollama_error(exc, base_url=url)) from exc
    except error.URLError as exc:
        raise RuntimeError(format_ollama_error(exc, base_url=url)) from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Некорректный ответ Ollama: {raw[:400]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Ollama вернула неожиданный формат ответа.")
    return parsed


def ollama_chat(
    *,
    base_url: str,
    model_name: str,
    content: str,
    image_bytes: bytes,
    timeout: int = DEFAULT_LLM_ANALYSIS_TIMEOUT,
) -> str:
    payload = build_chat_payload(
        model_name=model_name,
        content=content,
        image_bytes=image_bytes,
        stream=False,
    )
    data = _http_json_request(
        ollama_chat_url(base_url),
        method="POST",
        payload=payload,
        timeout=timeout,
    )
    text = extract_chat_response_text(data)
    if not text:
        raise RuntimeError(
            f"Ollama ({model_name}) вернула пустой результат анализа."
        )
    return text


def ollama_health_check(base_url: str) -> dict[str, Any]:
    normalized = normalize_ollama_base_url(base_url)
    try:
        data = _http_json_request(ollama_tags_url(normalized), timeout=10)
    except RuntimeError as exc:
        return {
            "reachable": False,
            "base_url": normalized,
            "models": [],
            "error": str(exc),
        }

    models: list[str] = []
    for item in data.get("models") or []:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if name:
                models.append(name)
    return {
        "reachable": True,
        "base_url": normalized,
        "models": models,
        "error": None,
    }


def model_is_available(model_name: str, installed_models: list[str]) -> bool:
    target = model_name.strip().lower()
    if not target:
        return False
    for name in installed_models:
        lowered = name.lower()
        if lowered == target or lowered.split(":", 1)[0] == target:
            return True
    return False


def format_health_check_message(
    health: dict[str, Any],
    *,
    model_name: str,
) -> str:
    if not health.get("reachable"):
        return str(health.get("error") or "Ollama недоступна.")
    models = list(health.get("models") or [])
    base = f"Ollama доступна ({health.get('base_url')}), моделей: {len(models)}."
    if model_is_available(model_name, models):
        return f"{base} Модель «{model_name}»: найдена."
    return (
        f"{base} Модель «{model_name}»: не найдена. "
        f"Выполните: ollama pull {model_name}"
    )


def prepare_image_bytes_for_llm(
    image_bytes: bytes,
    *,
    max_long_side: int = DEFAULT_LLM_MAX_LONG_SIDE,
    jpeg_quality: int = DEFAULT_LLM_JPEG_QUALITY,
) -> bytes:
    if not image_bytes:
        raise ValueError("Пустые данные изображения для LLM.")

    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return image_bytes

    height, width = img.shape[:2]
    long_side = max(width, height)
    if long_side > max_long_side:
        scale = max_long_side / long_side
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, encoded = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        return image_bytes
    return encoded.tobytes()


def build_analysis_prompt(record: StoredImageRecord) -> str:
    return (
        "You are analyzing a potentially map-like image.\n"
        "Describe what the image contains in detail. Decide whether it is a real terrain or topographic map, "
        "a poster, a screenshot, a logo, a photo, a document illustration, or something else.\n"
        "Return concise but useful analysis with these sections:\n"
        "1. Type\n"
        "2. Evidence\n"
        "3. MapLikelihood\n"
        "4. ImportantDetails\n\n"
        f"Source path: {record.source_path}\n"
        f"Scan scope: {record.scan_scope}\n"
        f"Heuristic detector version: {record.detector_version}\n"
        f"Heuristic detector is_map: {record.is_map}\n"
        f"Heuristic score: {record.score}\n"
        f"Heuristic threshold: {record.threshold}\n"
        f"Heuristic features JSON: {record.feature_summary_json}\n"
    )


def analyze_image_with_local_llm(
    *,
    image_bytes: bytes,
    record: StoredImageRecord,
    model_name: str = DEFAULT_LOCAL_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE,
    endpoint: str | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> dict[str, Any]:
    del endpoint  # legacy alias; base_url is used instead
    prompt = build_analysis_prompt(record)
    prepared = prepare_image_bytes_for_llm(image_bytes)
    analysis_text = ollama_chat(
        base_url=base_url,
        model_name=model_name,
        content=prompt,
        image_bytes=prepared,
        timeout=DEFAULT_LLM_ANALYSIS_TIMEOUT,
    )
    return {
        "status": "completed",
        "model_name": model_name,
        "prompt_version": prompt_version,
        "analysis_text": analysis_text,
        "structured_json": {"api": "chat", "base_url": normalize_ollama_base_url(base_url)},
    }


def classify_map_yes_no(
    *,
    image_bytes: bytes,
    model_name: str = DEFAULT_LOCAL_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE,
    endpoint: str | None = None,
    prompt_version: str = DEFAULT_YES_NO_PROMPT_VERSION,
    timeout: int = DEFAULT_LLM_GRAY_TIMEOUT,
) -> bool:
    """Fast yes/no: is this a topographic or terrain map?"""
    del endpoint  # legacy alias; base_url is used instead
    prompt = (
        "Look at this image. Is it a real topographic map, terrain map, "
        "or military/topographic chart with elevation/contour lines?\n"
        "Answer with exactly one word: yes or no.\n"
        "Do not explain."
    )
    prepared = prepare_image_bytes_for_llm(image_bytes)
    text = ollama_chat(
        base_url=base_url,
        model_name=model_name,
        content=prompt,
        image_bytes=prepared,
        timeout=timeout,
    ).lower()
    if text.startswith("yes") or text == "y":
        return True
    if text.startswith("no") or text == "n":
        return False
    return "yes" in text.split() and "no" not in text.split()[:1]
