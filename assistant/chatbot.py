from dataclasses import dataclass, field
from typing import Generator, Literal

from llm.infer import LLM
from assistant.rag import RAG
from assistant.router import needs_rag
from assistant.guardrails import (
    should_abstain,
    abstention_message,
    sanitize_chunks,
    answer_has_citation,
    citation_warning,
)
from assistant.clarification import needs_clarification, generate_clarifying_question

MAX_HISTORY_MESSAGES = 20

GENERAL_SYSTEM = """You are a helpful assistant. Answer clearly and concisely.
If you don't know something, say so — don't make things up."""


@dataclass
class ChatResponse:
    kind: Literal["general", "answer", "abstain", "clarify"]
    text: str
    chunks: list[dict] = field(default_factory=list)
    used_rag: bool = False


class Chatbot:
    def __init__(self, llm: LLM, rag: RAG):
        self.llm = llm
        self.rag = rag
        self.last_chunks: list[dict] = []
        self.last_route_scores: dict = {}

    @staticmethod
    def _trim_history(history: list[dict]) -> list[dict]:
        trimmed = []
        for m in history:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                trimmed.append({"role": m["role"], "content": m["content"]})
        return trimmed[-MAX_HISTORY_MESSAGES:]

    def _messages_with_query(self, history: list[dict], query: str) -> list[dict]:
        return self._trim_history(history) + [{"role": "user", "content": query}]

    def _fetch_context(self, query: str, chunks: list[dict] | None = None) -> tuple[str, list[dict]]:
        if chunks is None:
            chunks = self.rag.retrieve_hybrid(query, top_k=5)
        chunks = sanitize_chunks(chunks)
        self.last_chunks = chunks

        blocks = []
        for c in chunks:
            fname = c["metadata"].get("filename", "unknown")
            section = c["metadata"].get("section_title") or c["metadata"].get("title", "")
            tag = f"[{fname} §{section}]" if section else f"[{fname}]"
            blocks.append(f"{tag}\n{c['text']}")
        context = "\n\n---\n\n".join(blocks)

        system = f"""You are a helpful assistant. Answer ONLY from the context below.
If the context doesn't contain the answer, say you don't know — don't make things up.
When you state a fact, cite the source tag in square brackets exactly as shown.

CONTEXT:
{context}
"""
        return system, chunks

    def respond(
        self,
        query: str,
        history: list[dict] | None = None,
        *,
        force_rag: bool = False,
        rag_enabled: bool = True,
    ) -> ChatResponse:
        history = history or []

        if rag_enabled:
            use_rag, route_scores = needs_rag(
                query,
                self.rag.corpus_index,
                self.rag.embed,
                self.llm,
                force_rag=force_rag,
            )
            self.last_route_scores = route_scores
            if not use_rag:
                return ChatResponse(kind="general", text="", chunks=[], used_rag=False)
        else:
            return ChatResponse(kind="general", text="", chunks=[], used_rag=False)

        chunks = sanitize_chunks(self.rag.retrieve_hybrid(query, top_k=5))
        self.last_chunks = chunks

        if should_abstain(chunks):
            return ChatResponse(
                kind="abstain",
                text=abstention_message(),
                chunks=[],
                used_rag=True,
            )

        if needs_clarification(chunks):
            question = generate_clarifying_question(query, chunks, self.llm)
            return ChatResponse(
                kind="clarify",
                text=question,
                chunks=chunks,
                used_rag=True,
            )

        return ChatResponse(kind="answer", text="", chunks=chunks, used_rag=True)

    def stream_response(
        self,
        query: str,
        result: ChatResponse,
        history: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        history = history or []
        messages = self._messages_with_query(history, query)

        if result.kind == "clarify" or result.kind == "abstain":
            yield result.text
            return

        if result.kind == "general":
            self.last_chunks = []
            yield from self.llm.stream_messages(messages, system=GENERAL_SYSTEM)
            return

        system, chunks = self._fetch_context(query, chunks=result.chunks)
        full = ""
        for token in self.llm.stream_messages(messages, system=system):
            full += token
            yield token
        if chunks and not answer_has_citation(full):
            yield citation_warning()
