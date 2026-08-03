import json
import re
from pathlib import Path

# =============================================================================
# 第二階段：依 family_id 對專利資料做專利族去重
#
# 使用情境：
# 前一步已完成：
#   1. application_number 去重
#
# 本步驟處理「同一技術族在不同國家/地區公開」的跨國同族版本，
# 避免同一發明因 AU / CA / CN / EP / US / WO 等多國公開而重複進入
# BERTopic，造成主題頻率被放大。
#
# 去重邏輯：
# 1. 讀取 global_application_dedup.json。
# 2. 若 family_id 為空，保留不去重。
# 3. 若多筆資料具有相同 family_id，只保留一筆代表案。
# 4. 保留規則：
#    - 優先保留 abstract_en 品質較高者
#    - 其次保留英文文本品質較穩定國別：US → WO → EP → CA → AU → GB → CN → JP → KR
#    - 再依 priority_date → filing_date → publication_date 較早者
#    - 最後用原始順序作 tie-breaker
# 5. 輸出 global_family_dedup.json。
# =============================================================================

src = Path(
    "/home/carbon/carbon/data_global_v2/"
    "Carbon_onlycpc_global_morecpc_v2/"
    "global_application_dedup.json"
)

out = Path(
    "/home/carbon/carbon/data_global_v2/"
    "Carbon_onlycpc_global_morecpc_v2/"
    "global_family_dedup.json"
)

audit_out = Path(
    "/home/carbon/carbon/data_global_v2/"
    "Carbon_onlycpc_global_morecpc_v2/"
    "global_family_dedup_audit.json"
)

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

# 英文文本品質與後續分析可讀性排序，不是法律優先順序
PREFERRED_COUNTRIES = {
    "US": 0,
    "WO": 1,
    "EP": 2,
    "CA": 3,
    "AU": 4,
    "GB": 5,
    "CN": 6,
    "JP": 7,
    "KR": 8,
    "TW": 9,
    "MX": 10,
    "EA": 11,
    "RU": 12,
}


def date_sort_value(value):
    """將日期欄位轉成可排序數值；缺失日期放到最後。"""
    if value in (None, "", 0, "0"):
        return 99999999
    try:
        return int(value)
    except Exception:
        return 99999999


def normalize_family_id(value):
    """統一 family_id 格式，避免 int / str 混用造成分組錯誤。"""
    if value in (None, "", 0, "0"):
        return None
    return str(value).strip()


def country_rank(record):
    """國別排序；未列入者排後面。"""
    country = str(record.get("country_code") or "").upper()
    return PREFERRED_COUNTRIES.get(country, 99)


def safe_text(value):
    if value is None:
        return ""
    return str(value)


def abstract_quality_score(record):
    """
    摘要品質分數。
    分數越高，越適合保留為 family 代表案。

    設計原則：
    - 摘要完整度重要，word_count 較高通常較佳。
    - 出現核心鋼鐵/減碳技術詞加分。
    - 明顯截斷、PCT/OCR 殘留、俄文摘要格式詞等扣分。
    """
    title = safe_text(record.get("title_en"))
    abstract = safe_text(record.get("abstract_en"))
    text = f"{title} {abstract}"
    text_lower = text.lower()

    words = abstract.split()
    word_count = len(words)

    score = 0

    # 1. 摘要長度：作為唯一的分數，不再設定上限
    score += word_count

    return score


def record_sort_key(record):
    """
    family_id 內部排序鍵。
    Python sort 是由小到大，所以品質分數取負號。
    """
    return (
        country_rank(record),
        -abstract_quality_score(record),
        date_sort_value(record.get("priority_date")),
        date_sort_value(record.get("filing_date")),
        date_sort_value(record.get("publication_date")),
        record["_original_order"],
    )


def order_record_columns(record):
    """依照偏好欄位順序輸出，其餘欄位保留在後面。"""
    ordered = {}

    for column in PREFERRED_OUTPUT_COLUMNS:
        if column in record:
            ordered[column] = record[column]

    for key, value in record.items():
        if key not in ordered and key != "_original_order":
            ordered[key] = value

    return ordered


# =============================================================================
# 主流程
# =============================================================================

records = json.loads(src.read_text(encoding="utf-8"))

best_by_family = {}
members_by_family = {}
no_family_records = []

for original_order, record in enumerate(records):
    record["_original_order"] = original_order

    family_id = normalize_family_id(record.get("family_id"))

    # 沒有 family_id 的資料不要硬合併，直接保留
    if family_id is None:
        no_family_records.append(record)
        continue

    members_by_family.setdefault(family_id, []).append(record)

    current = best_by_family.get(family_id)

    if current is None or record_sort_key(record) < record_sort_key(current):
        best_by_family[family_id] = record


# 去重後資料 = 每個 family 的最佳代表案 + 無 family_id 的資料
deduped = list(best_by_family.values()) + no_family_records

# 保持大致原始順序
deduped.sort(key=lambda r: r["_original_order"])

ordered_deduped = [order_record_columns(record) for record in deduped]

# 輸出 family 去重結果
out.write_text(
    json.dumps(ordered_deduped, ensure_ascii=False, indent=2, allow_nan=False),
    encoding="utf-8",
)

# =============================================================================
# Audit：記錄哪些 family 被合併、保留哪一筆、刪除哪些公開案
# =============================================================================

audit_rows = []

for family_id, members in members_by_family.items():
    if len(members) <= 1:
        continue

    kept = best_by_family[family_id]

    removed = [
        m for m in members
        if m.get("publication_number") != kept.get("publication_number")
    ]

    audit_rows.append({
        "family_id": family_id,
        "family_size_in_input": len(members),
        "kept_publication_number": kept.get("publication_number"),
        "kept_application_number": kept.get("application_number"),
        "kept_country_code": kept.get("country_code"),
        "kept_kind_code": kept.get("kind_code"),
        "kept_priority_date": kept.get("priority_date"),
        "kept_publication_date": kept.get("publication_date"),
        "kept_title_en": kept.get("title_en"),
        "kept_abstract_word_count": len(safe_text(kept.get("abstract_en")).split()),
        "kept_quality_score": abstract_quality_score(kept),
        "removed_publication_numbers": [
            m.get("publication_number") for m in removed
        ],
        "removed_country_codes": [
            m.get("country_code") for m in removed
        ],
        "all_publication_numbers": [
            m.get("publication_number") for m in sorted(members, key=record_sort_key)
        ],
    })

audit_rows.sort(
    key=lambda x: (-x["family_size_in_input"], x["family_id"])
)

audit_out.write_text(
    json.dumps(audit_rows, ensure_ascii=False, indent=2, allow_nan=False),
    encoding="utf-8",
)

# =============================================================================
# 統計輸出
# =============================================================================

families_with_duplicates = sum(
    1 for members in members_by_family.values()
    if len(members) > 1
)

removed_by_family = sum(
    len(members) - 1
    for members in members_by_family.values()
    if len(members) > 1
)

print()
print("=== Family ID 去重結果 ===")
print(f"輸入總筆數：{len(records):,}")
print(f"有 family_id 的 family 數：{len(best_by_family):,}")
print(f"無 family_id 保留筆數：{len(no_family_records):,}")
print(f"重複 family 數：{families_with_duplicates:,}")
print(f"依 family_id 刪除跨國/同族公開案：{removed_by_family:,}")
print(f"最後輸出筆數：{len(ordered_deduped):,}")
print(f"輸出檔案：{out}")
print(f"Audit 檔案：{audit_out}")
print("==========================")