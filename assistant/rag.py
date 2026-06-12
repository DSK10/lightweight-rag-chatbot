import chromadb, sentence_transformers
import os
from threading import Thread
import time
from tqdm import tqdm
from assistant.parsers import parse_file
from assistant.chunking import Chunker
from assistant.metadata import enrich_document
import uuid
import json
from assistant.bm25_index import BM25Index
from assistant.reranker import Reranker
from assistant.dedup import is_duplicate, record
from assistant.router import CorpusIndex


class RAG:
    def __init__(self, db_path: str = "chroma_db"):
        print(f"[INFO] Initializing RAG with database path: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedder = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(name="rag_collection", metadata={"hnsw:space": "cosine"})
        self.chunker = Chunker()

        self.bm25 = BM25Index()
        self.bm25.load()
        self.reranker = Reranker()

        self.corpus_index = CorpusIndex()
        self.corpus_index.load(self.embed)

        

        # insert following documents
        # self.ingest([
        #     "/Users/deepeshsingh/Desktop/Career/Development/SutraAI_Assignment/data/code_of_conduct.txt"
        # ])

        # self.ingest([f"data/{x}" for x in os.listdir("data") if x.endswith(".txt")])

    def embed(self, texts: str | list[str]):
        vecs = self.embedder.encode(texts, show_progress_bar=True)
        if isinstance(texts, str):
            return vecs.tolist()
        return [vec.tolist() for vec in vecs]
    
    # def format_text(self, file_path: str):
    #     document_id = os.path.basename(file_path)
    #     with open(file_path, 'r') as file:
    #         text = file.read().splitlines()
    #     return [{"text": line, "document_id" : document_id, "chunk_index" : i} for i, line in enumerate(text)]

    def ingest_chunks(self, chunks: list[dict]) -> int:
        """
        chunks: list of dicts with keys:
            - text       (str)  required
            - doc_id     (str)  required — unique per source document
            - chunk_index (int) required
            - metadata   (dict) optional extra fields
        Returns number of new chunks added.
        """
        if not chunks:
            return 0

        ids, texts, metadatas, embeddings = [], [], [], []

        for c in chunks:
            chunk_id = f"{c['document_id']}__chunk_{c['chunk_index']}"


            existing = self.collection.get(ids=[chunk_id])
            if existing["ids"]:
                continue

            ids.append(chunk_id)
            texts.append(c["content"])
            meta = c.get("metadata", {})
            meta.update({"document_id": c["document_id"], "chunk_index": c["chunk_index"]})
            metadatas.append(meta)

        if not ids:
            return 0

        embeddings = self.embed(texts)
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        return len(ids)

    def ingest(self, file_paths: list[str]):
        try:
            print(f"[INFO] Ingesting {len(file_paths)} files")
            for file_path in tqdm(file_paths):
                if os.path.isdir(file_path):
                    continue

                if is_duplicate(file_path):
                    print(f"[SKIP] {file_path} — already indexed")
                    yield {"file": file_path, "status": "duplicate_skipped"}
                    continue
                print(f"[INFO] Ingesting {file_path}")

                document_id = os.path.basename(file_path)
                filename = os.path.basename(file_path)

                sections = parse_file(file_path)
                if not sections:
                    print(f"[INFO] No sections found for {file_path}")
                    continue
                
                raw_chunks = self.chunker.chunk_sections(sections)
                full_texts = "\n".join([chunk["content"] for chunk in raw_chunks])
                metadata = enrich_document(full_texts)
                document_type = filename.split(".")[-1]

                print(f"[INFO] Sections found for {file_path}: {len(sections)}")
        

                vector_chunks = []
                for chunk in raw_chunks:
                    meta = {
                        "document_id": document_id,
                        "section_title": chunk["title"],
                        "filename": filename,
                        "document_type": document_type,
                        "chunk_index": chunk["chunk_index"],
                        "summary": metadata["summary"],
                        "keywords": metadata["keywords"],
                        "intent_tags": metadata["intent_tags"],
                        "section_count": len(sections),
                        "chunk_count": len(raw_chunks),
                    }
                    meta.update(chunk["metadata"])
                    vector_chunks.append({
                        "content": chunk["content"],
                        "metadata": meta,
                        "document_id": document_id,
                        "chunk_index": chunk["chunk_index"],
                    })
                
                added = self.ingest_chunks(vector_chunks)
                if added == 0:
                    print(f"[INFO] No chunks added for {file_path}")
                    continue
                
                # save json sidecar
                sidecar = {
                    "document_id": document_id,
                    "filename": filename,
                    "document_type": document_type,
                    "section_count": len(sections),
                    "chunk_count": len(raw_chunks),
                    "summary": metadata["summary"],
                    "keywords": metadata["keywords"],
                    "intent_tags": metadata["intent_tags"],
                    "chunks": [
                        {
                            "chunk_index": chunk["chunk_index"],
                            "metadata": chunk["metadata"],
                            "content": chunk["content"],
                        }
                        for chunk in raw_chunks
                    ]
                }
                sidecar_path = os.path.join("data", "sidecar", f"{document_id}.json")
                os.makedirs(os.path.dirname(sidecar_path), exist_ok=True)
                with open(sidecar_path, "w") as f:
                    json.dump(sidecar, f, indent=2)
                
                record(file_path, document_id)
                yield {"file": file_path, "status": "added", "chunks": added}
            all_chunks = self.collection.get(include=["documents"])
            self.bm25.build(all_chunks["ids"], all_chunks["documents"])
            self.corpus_index.load(self.embed)
            # return {
            #     "status": "ok",
            #     "filename": filename,
            #     "document_id": document_id,
            #     "chunks_added": added,
            #     "chunks_total": len(raw_chunks),
            #     "summary": metadata["summary"],
            #     "keywords": metadata["keywords"],
            #     "intent_tags": metadata["intent_tags"],
            # }
        except Exception as e:
            print(f"[ERROR] Error ingesting {file_path}: {e}")
            yield {"file": file_path, "status": "error", "error": str(e)}
        
    def query(self, query: str, n_results: int = 5):
        if not self.collection:
            return []
        query_embedding = self.embed(query)
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["distances", "documents", "metadatas"],
        )
        return results
    

    def retrieve_hybrid(self, query: str, top_k: int = 5,
                    dense_k: int = 10, bm25_k: int = 10,
                    rrf_k: int = 60) -> list[dict]:

        dense = self.collection.query(
            query_embeddings=[self.embed(query)],
            n_results=dense_k,
            include=["documents", "metadatas"],
        )
        dense_ids = dense["ids"][0]
        dense_docs = dense["documents"][0]
        dense_metas = dense["metadatas"][0]


        bm25_hits = self.bm25.search(query, top_k=bm25_k)

        rrf: dict[str, float] = {}
        for rank, cid in enumerate(dense_ids):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        for rank, (cid, _) in enumerate(bm25_hits):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (rrf_k + rank)

        merged_ids = sorted(rrf, key=rrf.get, reverse=True)[: dense_k + bm25_k]

        fetched = self.collection.get(ids=merged_ids, include=["documents", "metadatas"])
        by_id = {i: (d, m) for i, d, m in zip(fetched["ids"], fetched["documents"], fetched["metadatas"])}
        candidates = [
            {"chunk_id": cid, "text": by_id[cid][0], "metadata": by_id[cid][1], "rrf_score": rrf[cid]}
            for cid in merged_ids if cid in by_id
        ]

        return self.reranker.rerank(query, candidates, top_k=top_k)

