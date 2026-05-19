import pandas as pd
import json
import numpy as np
from pathlib import Path

# =============================================================================
# 依 application_number 對多個 Parquet 專利資料去重
#
# 去重邏輯：
# 1. 合併 data_global 資料夾下所有 .parquet 檔案。
# 2. 同一個 application_number 若出現多筆，保留時間最早的一筆。
# 3. 日期排序優先順序為 priority_date → filing_date → publication_date。
# 4. 若日期仍相同，保留原始讀入順序最前的紀錄。
# 5. 最後輸出為 JSON，並把 numpy / pandas 的特殊值轉成 JSON 可序列化格式。
# =============================================================================
src_dir = Path("/home/carbon/carbon/data_global_v2/Carbon_onlycpc_global_morecpc_v2")
out = Path("/home/carbon/carbon/data_global_v2/Carbon_onlycpc_global_morecpc_v2/global_application_dedup.json")

# 收集來源資料夾中的所有 Parquet 檔，排序後讀取可讓結果更穩定可重現。
parquet_files = sorted(src_dir.glob("*.parquet"))
if not parquet_files:
    raise FileNotFoundError(f"在 {src_dir} 找不到任何 .parquet 檔案")

# 將所有 parquet 合併成單一 DataFrame，ignore_index=True 讓列索引重新編號。
df = pd.concat(
    [pd.read_parquet(path) for path in parquet_files],
    ignore_index=True,
)

# 保留原始順序，作為所有排序欄位都相同時的最後 tie-breaker。
df = df.reset_index(names="_original_order")

# 專利日期欄位若為 0，代表缺失值；轉成 99999999，避免排序時被誤認為最早日期。
sort_df = df.assign(
    _priority_sort=df["priority_date"].where(df["priority_date"] != 0, 99999999),
    _filing_sort=df["filing_date"].where(df["filing_date"] != 0, 99999999),
    _publication_sort=df["publication_date"].where(df["publication_date"] != 0, 99999999),
)

# 對同一 application_number 排序後取第一筆：
# priority_date 最早者優先，其次 filing_date、publication_date，最後用原始順序決定。
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
    """將 pandas/numpy 物件轉成標準 JSON 可序列化型別。"""
    if isinstance(value, np.ndarray):
        return [clean_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [clean_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_json_value(item) for key, item in value.items()}
    if pd.isna(value):
        return None
    return value


# DataFrame 轉 dict 後逐格清理，避免 json.dumps 遇到 ndarray、NaN 等非標準 JSON 值。
records = [
    {key: clean_json_value(value) for key, value in record.items()}
    for record in dedup.to_dict(orient="records")
]

# allow_nan=False 可強制阻擋 NaN / Infinity 進入輸出 JSON，確保檔案符合標準 JSON。
out.write_text(
    json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False),
    encoding="utf-8",
)

# 輸出基本統計，方便確認去重前後筆數與移除量。
print("source_rows", len(df))
print("dedup_rows", len(dedup))
print("unique_app", dedup["application_number"].nunique())
print("removed_rows", len(df) - len(dedup))
print("output", out)
