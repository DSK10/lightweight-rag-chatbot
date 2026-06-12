from pathlib import Path
from assistant.parsers.pdf import parse_pdf
from assistant.parsers.docx import parse_docx
from assistant.parsers.excel import parse_excel
from assistant.parsers.csv import parse_csv
from assistant.parsers.txt import parse_txt
from assistant.parsers.base import Section

SUPPORTED_FILE_EXTENSIONS = {
    ".txt": parse_txt,
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".xls": parse_excel,
    ".xlsx": parse_excel,
    ".csv": parse_csv,
    ".eml": parse_txt,
    ".md": parse_txt,
}

def parse_file(file_path: str) -> list[Section]:
    extension = Path(file_path).suffix.lower()
    if extension not in SUPPORTED_FILE_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {extension}")
    return SUPPORTED_FILE_EXTENSIONS[extension](file_path)