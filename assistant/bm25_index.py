import os, pickle, re
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"\w+")

def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())

class BM25Index:
    def __init__(self, cache_path: str = "chroma_db/bm25_index.pkl"):
        self.cache_path = cache_path
        self.bm25 = None
        self.chunk_ids: list[str] = []
        self.texts: list[str] = []

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        self.chunk_ids = list(chunk_ids)
        self.texts = list(texts)
        tokenized = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None
        self._save()

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self.bm25 or not self.chunk_ids:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump({"ids": self.chunk_ids, "texts": self.texts}, f)

    def load(self) -> bool:
        if not os.path.exists(self.cache_path):
            return False
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)
        self.build(data["ids"], data["texts"])
        return True