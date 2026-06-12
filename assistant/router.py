import json, os, re, glob
import numpy as np
from rank_bm25 import BM25Okapi

SIDECAR_DIR = "data/sidecar"
TOKEN_RE = re.compile(r"\w+")

# Only skip obvious chitchat — NOT document-domain terms
CHITCHAT_RE = re.compile(
    r"^(hi|hello|hey|thanks|thank you|bye|good morning|good night)\b",
    re.IGNORECASE,
)

RAG_HIGH = 0.45   # tune on your corpus
RAG_LOW  = 0.20


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _doc_profile(sidecar: dict) -> str:
    """One searchable string per document — built from enrichment, not hand-coded."""
    keywords = " ".join(sidecar.get("keywords") or [])
    tags = " ".join(sidecar.get("intent_tags") or [])
    summary = sidecar.get("summary") or ""
    filename = sidecar.get("filename") or sidecar.get("document_id") or ""
    return f"{filename}\n{tags}\n{keywords}\n{summary}"


class CorpusIndex:
    """Document-level index for routing. Rebuild after ingest."""

    def __init__(self, sidecar_dir: str = SIDECAR_DIR):
        self.sidecar_dir = sidecar_dir
        self.docs: list[dict] = []       # {document_id, filename, profile, keywords, intent_tags}
        self.bm25 = None
        self.profile_embeddings: np.ndarray | None = None

    def load(self, embed_fn) -> None:
        """embed_fn(text) -> list[float] — reuse RAG's SentenceTransformer."""
        self.docs = []
        for path in glob.glob(os.path.join(self.sidecar_dir, "*.json")):
            if os.path.basename(path).startswith("_"):
                continue  # skip _hashes.json
            with open(path) as f:
                sc = json.load(f)
            profile = _doc_profile(sc)
            if not profile.strip():
                continue
            self.docs.append({
                "document_id": sc.get("document_id", ""),
                "filename": sc.get("filename", ""),
                "profile": profile,
                "keywords": sc.get("keywords") or [],
                "intent_tags": sc.get("intent_tags") or [],
            })

        if not self.docs:
            self.bm25 = None
            self.profile_embeddings = None
            return

        tokenized = [_tokenize(d["profile"]) for d in self.docs]
        self.bm25 = BM25Okapi(tokenized)
        self.profile_embeddings = np.array([embed_fn(d["profile"]) for d in self.docs])

    def keyword_score(self, query: str) -> float:
        """BM25 best score across document profiles, normalized to ~0–1."""
        if not self.bm25 or not self.docs:
            return 0.0
        scores = self.bm25.get_scores(_tokenize(query))
        raw = float(np.max(scores)) if len(scores) else 0.0
        # BM25 is unbounded — squash for routing (tune divisor on your corpus)
        return min(raw / 10.0, 1.0)

    def semantic_score(self, query: str, embed_fn) -> float:
        """Max cosine similarity between query and any document profile."""
        if self.profile_embeddings is None or len(self.docs) == 0:
            return 0.0
        q = np.array(embed_fn(query))
        norms = np.linalg.norm(self.profile_embeddings, axis=1) * np.linalg.norm(q)
        sims = np.dot(self.profile_embeddings, q) / np.maximum(norms, 1e-9)
        return float(np.max(sims))  # already 0–1 for cosine

    def tag_overlap(self, query: str) -> float:
        """Bonus: fraction of query tokens that hit any intent_tag or keyword."""
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return 0.0
        best = 0.0
        for d in self.docs:
            corpus_tokens = set(_tokenize(" ".join(d["keywords"] + d["intent_tags"])))
            overlap = len(q_tokens & corpus_tokens) / len(q_tokens)
            best = max(best, overlap)
        return best

    def score(self, query: str, embed_fn) -> dict:
        kw = self.keyword_score(query)
        sem = self.semantic_score(query, embed_fn)
        tag = self.tag_overlap(query)
        combined = max(kw, sem, tag)  # any strong signal → document query
        return {"keyword": kw, "semantic": sem, "tag": tag, "combined": combined}

from llm.infer import LLM

def needs_rag_llm(query: str, llm: LLM, corpus_hint: str = "") -> bool:
    """Called ONLY when scores fall between RAG_LOW and RAG_HIGH."""
    prompt = f"""You route questions for a company document assistant.

Indexed document topics (sample):
{corpus_hint}

User question: "{query}"

Does answering this require searching the uploaded company documents?
Reply with ONLY: yes or no"""
    return llm.chat(prompt).strip().lower().startswith("y")


def needs_rag(
    query: str,
    index: CorpusIndex,
    embed_fn,
    llm: LLM | None = None,
    *,
    force_rag: bool = False,
) -> tuple[bool, dict]:
    """
    Returns (use_rag, score_breakdown).
    force_rag=True for clarification follow-ups.
    """
    if force_rag:
        return True, {"reason": "clarification_followup"}

    q = query.strip()
    if len(q) < 2 or CHITCHAT_RE.match(q):
        return False, {"reason": "chitchat"}

    if not index.docs:
        return True, {"reason": "empty_index_default_rag"}

    scores = index.score(q, embed_fn)
    combined = scores["combined"]

    if combined >= RAG_HIGH:
        return True, {**scores, "reason": "above_high"}
    if combined <= RAG_LOW:
        return False, {**scores, "reason": "below_low"}

    # Doubtful band — LLM breaks the tie
    if llm is not None:
        hint = "\n".join(
            f"- {d['filename']}: {', '.join(d['intent_tags'][:3])}"
            for d in index.docs[:8]
        )
        use_rag = needs_rag_llm(q, llm, hint)
        return use_rag, {**scores, "reason": "llm_tiebreak", "llm_said": use_rag}

    return True, {**scores, "reason": "doubtful_default_rag"}