from assistant.parsers.base import Section
import pandas as pd
import os

def _df_to_text(df: pd.DataFrame, max_rows: int = 200) -> str:
    df = df.head(max_rows)
    lines = [f"Columns: {', '.join(str(c) for c in df.columns)}"]
    for _, row in df.iterrows():
        row_str = " | ".join(f"{col}: {val}" for col, val in row.items())
        lines.append(row_str)
    return "\n".join(lines)


def parse_excel(filepath: str) -> list[Section]:
    sheets = pd.read_excel(filepath, sheet_name=None)  # dict of {sheet_name: df}
    sections = []
    for sheet_name, df in sheets.items():
        if df.empty:
            continue
        sections.append(Section(
            title=f"{os.path.basename(filepath)} - Sheet: {sheet_name}",
            content=_df_to_text(df),
            metadata={
                "sheet_name": sheet_name,
                "columns": list(df.columns),
                "row_count": len(df),
            },
        ))
    return sections