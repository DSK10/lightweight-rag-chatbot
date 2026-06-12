import fitz
from assistant.parsers.base import Section
from pprint import pprint
import os

def parse_pdf(filepath: str) -> list[Section]:
    sections = []
    doc = fitz.open(filepath)
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if not text:
            continue
        sections.append(Section(
            title=f"Page {page_num}",
            content=text,
            metadata={"page_number": page_num},
        ))
    doc.close()
    return sections