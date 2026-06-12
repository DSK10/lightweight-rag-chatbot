from docx import Document
from assistant.parsers.base import Section
from pprint import pprint

def parse_docx(filepath: str) -> list[Section]:
    sections = []
    doc = Document(filepath)
    for index, paragraph in enumerate(doc.paragraphs):
        if paragraph.text.strip() == "":
            continue
        sections.append(Section(
            title=f"Paragraph {index + 1}",
            content=paragraph.text,
            metadata={"paragraph_number": index + 1},
        ))
    return sections