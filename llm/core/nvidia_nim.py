import json
import os
from typing import Generator

import requests


class NvidiaNIM:
    URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(self, model: str | None = None):
        self.api_key = os.getenv("NVIDIA_NIM_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_NIM_API_KEY is not set")
        self.model = model or os.getenv("NVIDIA_NIM_MODEL", "moonshotai/kimi-k2.6")

    def _build_messages(self, history: list[dict], system: str) -> list[dict]:
        msgs = [{"role": "system", "content": system}] if system else []
        for m in history:
            if m["role"] in ("user", "assistant"):
                msgs.append({"role": m["role"], "content": m["content"]})
        return msgs

    def _payload(self, history: list[dict], system: str, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": self._build_messages(history, system),
            "max_tokens": 512,
            "stream": stream,
        }

    def _headers(self, stream: bool) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream" if stream else "application/json",
        }

    def chat_messages(self, history: list[dict], system: str = "") -> str:
        response = requests.post(
            self.URL,
            headers=self._headers(stream=False),
            json=self._payload(history, system, stream=False),
            timeout=120,
        )
        data = response.json()
        if "error" in data:
            raise RuntimeError(data["error"].get("message", data["error"]))
        return data["choices"][0]["message"]["content"]

    def chat(self, prompt: str, system: str = "") -> str:
        return self.chat_messages([{"role": "user", "content": prompt}], system)

    def stream_messages(self, history: list[dict], system: str = "") -> Generator[str, None, None]:
        response = requests.post(
            self.URL,
            headers=self._headers(stream=True),
            json=self._payload(history, system, stream=True),
            timeout=120,
            stream=True,
        )
        for line in response.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload.strip() == b"[DONE]":
                break
            data = json.loads(payload)
            if "error" in data:
                raise RuntimeError(data["error"].get("message", data["error"]))
            choices = data.get("choices") or []
            if not choices:
                continue
            content = choices[0].get("delta", {}).get("content")
            if content:
                yield content

    def stream(self, prompt: str, system: str = "") -> Generator[str, None, None]:
        yield from self.stream_messages([{"role": "user", "content": prompt}], system)
