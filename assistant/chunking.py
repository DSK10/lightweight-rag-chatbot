from assistant.parsers.base import Section
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: list[str] = ["\n\n", "\n", ". ", " ", ""]):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk_sections(self, sections: list[Section]) -> list[dict]:
        chunks = []
        global_idx = 0
        for section in sections:
            pieces = self.splitter.split_text(section.content)
            for piece in pieces:
                chunks.append({
                    "content": piece,
                    "title": section.title,
                    "chunk_index": global_idx,
                    "metadata": section.metadata,
                })
                global_idx += 1
        return chunks