import json
from pathlib import Path


src = Path("/home/carbon/carbon/data/part-000000000000_dedup.json")
out = Path("/home/carbon/carbon/data/part-000000000000_dedup_by_abstract.json")


def date_sort_value(value):
    # 這份資料裡 0 通常代表缺失日期，所以排到最後
    if value in (None, "", 0, "0"):
        return 99999999
    return int(value)


def record_sort_key(record):
    return (
        date_sort_value(record.get("priority_date")),
        date_sort_value(record.get("filing_date")),
        date_sort_value(record.get("publication_date")),
        record["_original_order"],
    )


records = json.loads(src.read_text(encoding="utf-8"))

best_by_abstract = {}
removed_null_abstract = 0

for original_order, record in enumerate(records):
    abstract = record.get("abstract_en")

    # abstract_en 是 null 或空字串，直接去除
    if abstract is None or abstract == "":
        removed_null_abstract += 1
        continue

    record["_original_order"] = original_order

    current = best_by_abstract.get(abstract)

    if current is None or record_sort_key(record) < record_sort_key(current):
        best_by_abstract[abstract] = record


deduped = list(best_by_abstract.values())

# 保持輸出大致依照原始資料順序排列
deduped.sort(key=lambda record: record["_original_order"])

for record in deduped:
    del record["_original_order"]


out.write_text(
    json.dumps(deduped, ensure_ascii=False, indent=2, allow_nan=False),
    encoding="utf-8",
)

duplicate_abstract_removed = len(records) - removed_null_abstract - len(deduped)

print()
print("=== Abstract 去重結果 ===")
print(f"輸入總筆數：{len(records):,}")
print(f"刪除 abstract_en 為 null / 空字串：{removed_null_abstract:,}")
print(f"刪除 abstract_en 重複資料：{duplicate_abstract_removed:,}")
print(f"總共刪除筆數：{len(records) - len(deduped):,}")
print(f"最後輸出筆數：{len(deduped):,}")
print(f"輸出檔案：{out}")
print("========================")

