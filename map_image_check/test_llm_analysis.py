"""Tests for Ollama LLM analysis helpers."""

from __future__ import annotations

import io
import json
import unittest
from urllib import error

import cv2
import numpy as np

from map_image_check.llm_analysis import (
    build_chat_payload,
    extract_chat_response_text,
    format_health_check_message,
    format_ollama_error,
    model_is_available,
    normalize_ollama_base_url,
    ollama_health_check,
    prepare_image_bytes_for_llm,
)


class LlmAnalysisTests(unittest.TestCase):
    def test_normalize_ollama_base_url(self) -> None:
        self.assertEqual(
            normalize_ollama_base_url("127.0.0.1:11434"),
            "http://127.0.0.1:11434",
        )
        self.assertEqual(
            normalize_ollama_base_url("http://127.0.0.1:11434/"),
            "http://127.0.0.1:11434",
        )

    def test_build_chat_payload(self) -> None:
        payload = build_chat_payload(
            model_name="llama3.2-vision",
            content="describe",
            image_bytes=b"abc",
        )
        self.assertEqual(payload["model"], "llama3.2-vision")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][0]["content"], "describe")
        self.assertTrue(payload["messages"][0]["images"])

    def test_extract_chat_response_text(self) -> None:
        self.assertEqual(
            extract_chat_response_text({"message": {"content": "hello"}}),
            "hello",
        )
        self.assertEqual(
            extract_chat_response_text({"response": "legacy"}),
            "legacy",
        )

    def test_format_ollama_error_http_model_not_found(self) -> None:
        body = json.dumps({"error": "model 'llama3.2-vision' not found"}).encode(
            "utf-8"
        )
        exc = error.HTTPError(
            url="http://127.0.0.1:11434/api/chat",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=io.BytesIO(body),
        )
        message = format_ollama_error(exc)
        self.assertIn("не установлена", message.lower())
        self.assertIn("ollama pull", message.lower())

    def test_model_is_available(self) -> None:
        models = ["llama3.2-vision:latest", "gemma3:4b"]
        self.assertTrue(model_is_available("llama3.2-vision", models))
        self.assertTrue(model_is_available("llama3.2-vision:latest", models))
        self.assertFalse(model_is_available("llava", models))

    def test_format_health_check_message(self) -> None:
        ok = format_health_check_message(
            {
                "reachable": True,
                "base_url": "http://127.0.0.1:11434",
                "models": ["llama3.2-vision:latest"],
            },
            model_name="llama3.2-vision",
        )
        self.assertIn("найдена", ok.lower())

        missing = format_health_check_message(
            {
                "reachable": True,
                "base_url": "http://127.0.0.1:11434",
                "models": ["gemma3:4b"],
            },
            model_name="llama3.2-vision",
        )
        self.assertIn("не найдена", missing.lower())

    def test_prepare_image_bytes_for_llm_resizes_large_image(self) -> None:
        img = np.zeros((4000, 3000, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", img)
        self.assertTrue(ok)
        prepared = prepare_image_bytes_for_llm(encoded.tobytes(), max_long_side=1536)
        buf = np.frombuffer(prepared, dtype=np.uint8)
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        assert decoded is not None
        h, w = decoded.shape[:2]
        self.assertLessEqual(max(h, w), 1536)

    def test_ollama_health_check_unreachable(self) -> None:
        health = ollama_health_check("http://127.0.0.1:59999")
        self.assertFalse(health["reachable"])
        self.assertIsNotNone(health["error"])


if __name__ == "__main__":
    unittest.main()
