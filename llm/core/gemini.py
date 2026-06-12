import json
import os
from typing import Generator

import requests


class Gemini:
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, model: str | None = None):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def _contents(self, history: list[dict]) -> list[dict]:
        contents = []
        for m in history:
            if m["role"] not in ("user", "assistant"):
                continue
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return contents

    def _payload(self, history: list[dict], system: str) -> dict:
        payload = {
            "contents": self._contents(history),
            "generationConfig": {"maxOutputTokens": 512},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    def _extract_text(self, data: dict) -> str:
        if "error" in data:
            raise RuntimeError(data["error"].get("message", data["error"]))
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return ""

    def chat_messages(self, history: list[dict], system: str = "") -> str:
        response = requests.post(
            f"{self.BASE_URL}/{self.model}:generateContent",
            params={"key": self.api_key},
            json=self._payload(history, system),
            timeout=60,
        )
        return self._extract_text(response.json())

    def chat(self, prompt: str, system: str = "") -> str:
        return self.chat_messages([{"role": "user", "content": prompt}], system)

    def stream_messages(self, history: list[dict], system: str = "") -> Generator[str, None, None]:
        response = requests.post(
            f"{self.BASE_URL}/{self.model}:streamGenerateContent",
            params={"key": self.api_key, "alt": "sse"},
            json=self._payload(history, system),
            timeout=120,
            stream=True,
        )
        for line in response.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            text = self._extract_text(json.loads(line[6:]))
            if text:
                yield text

    def stream(self, prompt: str, system: str = "") -> Generator[str, None, None]:
        yield from self.stream_messages([{"role": "user", "content": prompt}], system)
