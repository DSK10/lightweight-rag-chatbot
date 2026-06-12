import json
from llm.infer import LLM

ENRICHMENT_PROMPT = """You are a document analysis assistant. Given the document text below, respond ONLY with valid JSON (no markdown, no preamble) in this exact format:

{{
  "summary": "2-sentence summary of the document",
  "keywords": ["keyword1", "keyword2", "..."],
  "intent_tags": ["tag1", "tag2", "tag3"]
}}

Choose intent_tags from: policy, financial, hr, technical, legal, operations, communication, general.
Generate 8-10 relevant keywords.

DOCUMENT TEXT (truncated):
{text}
"""

def enrich_document(full_text: str, max_chars: int = 4000) -> dict:
    """
    Returns {"summary": str, "keywords": [...], "intent_tags": [...]}
    Falls back to safe defaults if the LLM doesn't return valid JSON.
    """
    truncated = full_text[:max_chars]
    prompt = ENRICHMENT_PROMPT.format(text=truncated)

    raw = LLM("openai", "gpt-5-nano").chat(prompt)

    # Strip markdown fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        return {
            "summary": data.get("summary", ""),
            "keywords": data.get("keywords", []),
            "intent_tags": data.get("intent_tags", []),
        }
    except (json.JSONDecodeError, AttributeError):
        return {"summary": "", "keywords": [], "intent_tags": []}