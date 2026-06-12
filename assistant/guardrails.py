import re

MIN_USEFUL_SCORE = -5.0
AMBIGUITY_SCORE_GAP = 1.5

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?prior",
    r"you\s+are\s+now",
    r"system\s+prompt",
    r"<\s*/?\s*system\s*>",
]
_INJECTION_RES = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

CITATION_TAG_RE = re.compile(r"\[[^\]]+\]")


def should_abstain(chunks: list[dict]) -> bool:
    """True when retrieval confidence is too low to answer safely from documents."""
    if not chunks:
        return True
    top_score = chunks[0].get("rerank_score", float("-inf"))
    return top_score < MIN_USEFUL_SCORE


def abstention_message() -> str:
    return (
        "I don't have enough information in the provided documents to answer that confidently. "
        "Try rephrasing your question or uploading a relevant document."
    )


def sanitize_chunk_text(text: str) -> str:
    cleaned = text
    for pattern in _INJECTION_RES:
        cleaned = pattern.sub("[removed]", cleaned)
    return cleaned


def sanitize_chunks(chunks: list[dict]) -> list[dict]:
    out = []
    for c in chunks:
        copy = dict(c)
        copy["text"] = sanitize_chunk_text(c.get("text", ""))
        out.append(copy)
    return out


def answer_has_citation(answer: str) -> bool:
    return bool(CITATION_TAG_RE.search(answer))


def citation_warning() -> str:
    return "\n\n*(Note: this answer may not be fully grounded — no source citation was detected.)*"
