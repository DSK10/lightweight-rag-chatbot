from assistant.rag import RAG
import os
try:
    rag = RAG()
    for data in rag.ingest([f"data/{x}" for x in os.listdir("data")]):
        print(data)
except Exception as e:
    print(f"Error ingesting documents: {e}")
    raise e