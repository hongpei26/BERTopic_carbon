import pandas as pd
import json
import numpy as np
from pathlib import Path

src = Path("/home/carbon/carbon/data/part-000000000000.parquet")
out = Path("/home/carbon/carbon/data/part-000000000000_dedup.json")

df = pd.read_parquet(src)

# 保留原始順序，作為最後 tie-breaker
df = df.reset_index(names="_original_order")

# 將 0 視為缺失日期，避免被當成最早日期
sort_df = df.assign(
    _priority_sort=df["priority_date"].where(df["priority_date"] != 0, 99999999),
    _filing_sort=df["filing_date"].where(df["filing_date"] != 0, 99999999),
    _publication_sort=df["publication_date"].where(df["publication_date"] != 0, 99999999),
)

dedup = (
    sort_df
    .sort_values(
        [
            "application_number",
            "_priority_sort",
            "_filing_sort",
            "_publication_sort",
            "_original_order",
        ],
        kind="mergesort",
    )
    .drop_duplicates("application_number", keep="first")
    .drop(
        columns=[
            "_priority_sort",
            "_filing_sort",
            "_publication_sort",
            "_original_order",
        ]
    )
    .reset_index(drop=True)
)

def clean_json_value(value):
    if isinstance(value, np.ndarray):
        return [clean_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [clean_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_json_value(item) for key, item in value.items()}
    if pd.isna(value):
        return None
    return value


records = [
    {key: clean_json_value(value) for key, value in record.items()}
    for record in dedup.to_dict(orient="records")
]

out.write_text(
    json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False),
    encoding="utf-8",
)

print("source_rows", len(df))
print("dedup_rows", len(dedup))
print("unique_app", dedup["application_number"].nunique())
print("removed_rows", len(df) - len(dedup))
print("output", out)