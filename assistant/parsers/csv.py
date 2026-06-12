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


def parse_csv(filepath: str) -> list[Section]:
    df = pd.read_csv(filepath)
    return [Section(
        title=f"CSV Data {os.path.basename(filepath)}",
        content=_df_to_text(df),
        metadata={
            "columns": list(df.columns),
            "row_count": len(df),
        },
    )]