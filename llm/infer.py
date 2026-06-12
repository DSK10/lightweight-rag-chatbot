from typing import Generator

from dotenv import load_dotenv

load_dotenv()


class LLM:
    def __init__(self, backend: str, model: str | None = None):
        if backend == "ollama":
            from llm.core.olama import Ollama
            self.provider = Ollama(model=model)
        elif backend == "huggingface":
            from llm.core.huggingface import HuggingFace
            self.provider = HuggingFace(model=model)
        elif backend == "gemini":
            from llm.core.gemini import Gemini
            self.provider = Gemini(model=model)
        elif backend == "nvidia_nim":
            from llm.core.nvidia_nim import NvidiaNIM
            self.provider = NvidiaNIM(model=model)
        elif backend == "openai":
            from llm.core.openai_gpt import OpenAIGPT
            self.provider = OpenAIGPT(model=model)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.backend = backend
        self.model = self.provider.model

    def chat_messages(self, history: list[dict], system: str = "") -> str:
        try:
            return self.provider.chat_messages(history, system)
        except Exception as e:
            print(f"[ERROR] Error chatting with {self.backend} / {self.model}: {e}")
            return ""

    def chat(self, prompt: str, system: str = "") -> str:
        return self.chat_messages([{"role": "user", "content": prompt}], system)

    def stream_messages(self, history: list[dict], system: str = "") -> Generator[str, None, None]:
        try:
            return self.provider.stream_messages(history, system)
        except Exception as e:
            print(f"[ERROR] Error streaming from {self.backend} / {self.model}: {e}")
            return iter(())

    def stream(self, prompt: str, system: str = "") -> Generator[str, None, None]:
        yield from self.stream_messages([{"role": "user", "content": prompt}], system)


if __name__ == "__main__":
    llm = LLM("ollama")
    print(llm.chat("Why is the sky blue?"))
