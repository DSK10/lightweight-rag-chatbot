from assistant.parsers.base import Section
import os
def parse_txt(filepath: str) -> list[Section]:
    with open(filepath, 'r') as file:
        text = file.read().splitlines()
    return [Section(
        title=f"Text File {os.path.basename(filepath)}",
        content="\n".join(text),
        metadata={"file_name": os.path.basename(filepath)},
    )]