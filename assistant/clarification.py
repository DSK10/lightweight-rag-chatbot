from assistant.guardrails import MIN_USEFUL_SCORE, AMBIGUITY_SCORE_GAP
from llm.infer import LLM


def _chunk_label(chunk: dict) -> str:
    meta = chunk.get("metadata", {})
    fname = meta.get("filename", "unknown")
    section = meta.get("section_title") or meta.get("title", "")
    return f"{fname} — {section}" if section else fname


def needs_clarification(chunks: list[dict], top_n: int = 3) -> bool:
    if len(chunks) < 2:
        return False

    top = chunks[:top_n]
    if top[0].get("rerank_score", float("-inf")) < MIN_USEFUL_SCORE:
        return False

    labels = {_chunk_label(c) for c in top}
    if len(labels) < 2:
        return False

    top_score = top[0]["rerank_score"]
    close = [
        c for c in top[1:]
        if top_score - c["rerank_score"] <= AMBIGUITY_SCORE_GAP
    ]
    if not close:
        return False

    close_labels = {_chunk_label(c) for c in close}
    return len(close_labels) >= 1


def generate_clarifying_question(query: str, chunks: list[dict], llm: LLM) -> str:
    options = list(dict.fromkeys(_chunk_label(c) for c in chunks[:3]))
    options_text = "\n".join(f"- {o}" for o in options)

    prompt = f"""The user asked: "{query}"

Retrieval found several equally relevant sections:
{options_text}

Write ONE short clarifying question (one sentence) asking the user which topic they mean.
Do not answer the original question. Do not list bullet points — just the question."""

    return llm.chat(prompt).strip()