import json
from pathlib import Path

# =============================================================================
# 第三階段：依 abstract_en 對專利資料做最後去重
#
# 使用情境：
# 前一步已經用 family_id 去重，這支腳本做最後掃蕩，處理「不同專利家族但摘要完全相同」
# 的極端重複資料，避免同一段英文摘要在語料中重複出現，影響後續 BERTopic 主題建模。
#
# 去重邏輯：
# 1. 讀取 global_family_dedup.json。
# 2. 移除 abstract_en 為 null 或空字串的紀錄。
# 3. 以完整 abstract_en 文字作為 key。
# 4. 若多筆資料摘要完全相同，保留日期最早的一筆：
#    priority_date → filing_date → publication_date。
# 5. 若日期仍相同，保留原始讀入順序最前的紀錄作為 tie-breaker。
# 6. 輸出 global_abstract_dedup.json (供給後續模型使用)。
# =============================================================================

# 輸入資料：前一步 family_id 去重後的 JSON
src = Path("/home/carbon/carbon/data_global_v2/Carbon_onlycpc_global_morecpc_v2/global_family_dedup.json")
# 輸出資料：再依摘要去重後的 JSON
out = Path("/home/carbon/carbon/data_global_v2/Carbon_onlycpc_global_morecpc_v2/global_abstract_dedup.json")

PREFERRED_OUTPUT_COLUMNS = [
    "publication_number",
    "application_number",
    "family_id",
    "country_code",
    "kind_code",
    "publication_date",
    "filing_date",
    "priority_date",
    "title_en",
    "abstract_en",
    "matched_cpc_codes",
    "all_cpc_codes",
    "all_ipc_codes",
]


def date_sort_value(value):
    """將日期欄位轉成可排序數值；缺失日期放到最後。"""
    # 0 / 空值通常代表缺失日期，排序時放到最後，避免被誤判為最早日期。
    if value in (None, "", 0, "0"):
        return 99999999
    return int(value)


def record_sort_key(record):
    """回傳同摘要資料之間的排序鍵，用來決定要保留哪一筆。"""
    # 當多筆資料摘要相同時，用日期最早的那筆保留；
    # 若日期也相同，則保留原始資料中較前面的那筆。
    return (
        date_sort_value(record.get("priority_date")),
        date_sort_value(record.get("filing_date")),
        date_sort_value(record.get("publication_date")),
        record["_original_order"],
    )


# 讀取原始 JSON 資料
records = json.loads(src.read_text(encoding="utf-8"))

# 用 abstract_en 當 key，記錄每個摘要目前要保留的最佳資料
best_by_abstract = {}
removed_null_abstract = 0

# enumerate 產生 original_order，讓後面可以在日期完全相同時保留原始順序較前者。
for original_order, record in enumerate(records):
    abstract = record.get("abstract_en")

    # abstract_en 是 null 或空字串，直接排除，不參與去重。
    if abstract is None or abstract == "":
        removed_null_abstract += 1
        continue

    # 暫存原始順序，作為最後的 tie-breaker。
    record["_original_order"] = original_order

    current = best_by_abstract.get(abstract)

    # 若第一次遇到這個摘要，或這筆資料排序更優先，則更新保留對象。
    if current is None or record_sort_key(record) < record_sort_key(current):
        best_by_abstract[abstract] = record


# 取出去重後的所有資料
deduped = list(best_by_abstract.values())

# 保持輸出大致依照原始資料順序排列
deduped.sort(key=lambda record: record["_original_order"])

# 清除中途加入的暫存欄位，避免寫入最終檔案。
for record in deduped:
    del record["_original_order"]


missing_preferred = [
    column for column in PREFERRED_OUTPUT_COLUMNS
    if not any(column in record for record in deduped)
]
if missing_preferred:
    print(f"warning_missing_preferred_columns {missing_preferred}")

ordered_deduped = []
for record in deduped:
    ordered_record = {}
    for column in PREFERRED_OUTPUT_COLUMNS:
        if column in record:
            ordered_record[column] = record[column]
    for key, value in record.items():
        if key not in ordered_record:
            ordered_record[key] = value
    ordered_deduped.append(ordered_record)


# 輸出去重結果；allow_nan=False 確保輸出符合標準 JSON，不允許 NaN / Infinity。
out.write_text(
    json.dumps(ordered_deduped, ensure_ascii=False, indent=2, allow_nan=False),
    encoding="utf-8",
)

# 統計被刪除的重複摘要筆數
duplicate_abstract_removed = len(records) - removed_null_abstract - len(deduped)

# 印出執行結果摘要
print()
print("=== Abstract 去重結果 ===")
print(f"輸入總筆數：{len(records):,}")
print(f"刪除 abstract_en 為 null / 空字串：{removed_null_abstract:,}")
print(f"刪除 abstract_en 重複資料：{duplicate_abstract_removed:,}")
print(f"總共刪除筆數：{len(records) - len(deduped):,}")
print(f"最後輸出筆數：{len(deduped):,}")
print(f"輸出檔案：{out}")
print("========================")
