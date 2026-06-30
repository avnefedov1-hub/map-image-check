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

DEFAULT_LOCAL_MODEL = "qwen2.5vl:7b"
DEFAULT_PROMPT_VERSION = "map-review-v2"
DEFAULT_YES_NO_PROMPT_VERSION = "map-yes-no-v2"
DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_ENDPOINT = f"{DEFAULT_OLLAMA_BASE}/api/generate"
DEFAULT_LLM_GRAY_TIMEOUT = 60
DEFAULT_LLM_ANALYSIS_TIMEOUT = 180
DEFAULT_LLM_MAX_LONG_SIDE = 1024
DEFAULT_LLM_JPEG_QUALITY = 85
DEFAULT_LLM_NUM_CTX = 8192


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
    num_ctx: int = DEFAULT_LLM_NUM_CTX,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
    if num_ctx > 0:
        payload["options"] = {"num_ctx": int(num_ctx)}
    return payload


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
            if "exceed_context_size" in lowered or "exceeds the available context" in lowered:
                return (
                    "Запрос с изображением превышает размер контекста модели (num_ctx). "
                    "Увеличьте context length до 8192 или выше "
                    "(Open WebUI: настройки чата → Advanced Params → Context Length)."
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
        "Вы анализируете изображение, которое может быть картой.\n"
        "Подробно опишите, что изображено. Определите, является ли это реальной "
        "топографической или terrain-картой, постером, скриншотом, логотипом, "
        "фотографией, иллюстрацией в документе или чем-то другим.\n"
        "Ответ дайте полностью на русском языке. Верните краткий, но полезный "
        "анализ со следующими разделами:\n"
        "1. Тип\n"
        "2. Доказательства\n"
        "3. ВероятностьКарты\n"
        "4. ВажныеДетали\n\n"
        f"Путь к файлу: {record.source_path}\n"
        f"Область сканирования: {record.scan_scope}\n"
        f"Версия эвристического детектора: {record.detector_version}\n"
        f"Эвристика is_map: {record.is_map}\n"
        f"Эвристический score: {record.score}\n"
        f"Порог эвристики: {record.threshold}\n"
        f"Признаки эвристики (JSON): {record.feature_summary_json}\n"
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


def _parse_yes_no_response(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        raise RuntimeError("LLM вернула пустой ответ.")
    if normalized.startswith("да") or normalized.startswith("yes") or normalized == "y":
        return True
    if normalized.startswith("нет") or normalized.startswith("no") or normalized == "n":
        return False
    words = normalized.split()
    if words and words[0] in ("да", "yes"):
        return True
    if words and words[0] in ("нет", "no"):
        return False
    return "да" in words and "нет" not in words[:1]


def classify_map_yes_no_with_response(
    *,
    image_bytes: bytes,
    model_name: str = DEFAULT_LOCAL_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE,
    endpoint: str | None = None,
    prompt_version: str = DEFAULT_YES_NO_PROMPT_VERSION,
    timeout: int = DEFAULT_LLM_GRAY_TIMEOUT,
) -> tuple[bool, str]:
    """Fast yes/no with raw model response text."""
    del endpoint, prompt_version  # legacy aliases
    prompt = (
        "Посмотрите на изображение. Это настоящая топографическая карта, terrain-карта "
        "или военно-топографическая схема с линиями рельефа и контурами?\n"
        "Ответьте ровно одним словом на русском: да или нет.\n"
        "Не объясняйте."
    )
    prepared = prepare_image_bytes_for_llm(image_bytes)
    text = ollama_chat(
        base_url=base_url,
        model_name=model_name,
        content=prompt,
        image_bytes=prepared,
        timeout=timeout,
    )
    return _parse_yes_no_response(text), text.strip()


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
    verdict, _response = classify_map_yes_no_with_response(
        image_bytes=image_bytes,
        model_name=model_name,
        base_url=base_url,
        endpoint=endpoint,
        prompt_version=prompt_version,
        timeout=timeout,
    )
    return verdict


def build_gray_zone_llm_result(
    *,
    model_name: str,
    llm_verdict: bool,
    llm_response_text: str | None,
    ml_score: float | None,
    heuristic_score: float,
) -> dict[str, Any]:
    verdict_label = "да, это карта" if llm_verdict else "нет, это не карта"
    response = (llm_response_text or verdict_label).strip()
    ml_part = f"{ml_score:.2f}" if ml_score is not None else "—"
    return {
        "status": "gray_zone",
        "model_name": model_name,
        "prompt_version": DEFAULT_YES_NO_PROMPT_VERSION,
        "is_topographic_map": llm_verdict,
        "analysis_text": (
            f"Быстрая проверка LLM (серая зона ML): {verdict_label}.\n"
            f"Ответ модели: {response}\n"
            f"ML score: {ml_part}\n"
            f"Эвристика: {heuristic_score:.2f}"
        ),
        "structured_json": {
            "kind": "gray_zone",
            "llm_verdict": llm_verdict,
            "llm_response_text": llm_response_text,
            "ml_score": ml_score,
            "heuristic_score": heuristic_score,
        },
    }
