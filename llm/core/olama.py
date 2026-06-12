import os
from typing import Generator

from ollama import ResponseError, chat, pull


class Ollama:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "tinyllama")

    def _build_messages(self, history: list[dict], system: str) -> list[dict]:
        msgs = [{"role": "system", "content": system}] if system else []
        for m in history:
            if m["role"] in ("user", "assistant"):
                msgs.append({"role": m["role"], "content": m["content"]})
        return msgs

    def _call(self, history: list[dict], system: str, stream: bool):
        messages = self._build_messages(history, system)
        try:
            return chat(model=self.model, messages=messages, stream=stream)
        except ResponseError as e:
            if e.status_code != 404:
                raise
            print(f"Model '{self.model}' not found locally. Pulling...")
            pull(self.model)
            return chat(model=self.model, messages=messages, stream=stream)

    def chat_messages(self, history: list[dict], system: str = "") -> str:
        return self._call(history, system, stream=False).message.content

    def chat(self, prompt: str, system: str = "") -> str:
        return self.chat_messages([{"role": "user", "content": prompt}], system)

    def stream_messages(self, history: list[dict], system: str = "") -> Generator[str, None, None]:
        for chunk in self._call(history, system, stream=True):
            yield chunk.message.content

    def stream(self, prompt: str, system: str = "") -> Generator[str, None, None]:
        yield from self.stream_messages([{"role": "user", "content": prompt}], system)
