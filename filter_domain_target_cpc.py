import json
from pathlib import Path

# =============================================================================
# 依 CPC 代碼篩選「鋼鐵領域」且「碳中和/減碳目標」相關的專利
#
# 篩選邏輯：
# 1. 讀取已完成 application_number 與 abstract 去重後的 JSON。
# 2. 從每筆資料的 matched_cpc_codes 取出 CPC 碼。
# 3. 第一群組 Domain：判斷是否屬於鋼鐵/鐵製造領域。
# 4. 第二群組 Target：判斷是否屬於低碳製程、CO2 捕集、氣體分離等目標技術。
# 5. 只有同時命中 Domain AND Target 的資料才輸出。
# 6. 另外把實際命中的 prefix 寫回每筆資料，方便後續檢查與分析。
# =============================================================================

# 輸入資料：前一步依 abstract_en 去重後的資料。
src = Path("/home/carbon/carbon/data_global/global_abstract_dedup.json")
# 輸出資料：同時命中 Domain 與 Target CPC 條件的資料。
out = Path("/home/carbon/carbon/data_global/global_cpc_domain_target_intersection.json")


# 第一群組：鋼鐵 / 鐵製造相關 CPC prefix。
DOMAIN_PREFIXES = [
    "C21B",  # 高爐與鐵製造
    "C21C",  # 煉鋼精煉
]

# 第二群組：碳中和、製程減碳、CO2 捕集/處理、氣體分離相關 CPC prefix。
TARGET_PREFIXES = [
    "Y02P10/10",
    "Y02P10/122",
    "Y02P10/134",
    "Y02P10/20",
    "Y02P10/25",
    "Y02C20/40",
    "B01D53",
]


def normalize_cpc(code):
    """統一 CPC 字串格式：轉字串、去前後空白、轉大寫。"""
    return str(code).strip().upper()


def match_prefix(code, prefixes):
    """判斷單一 CPC code 是否命中任一指定 prefix。"""
    code = normalize_cpc(code)
    return any(code.startswith(prefix) for prefix in prefixes)


def matched_prefixes(codes, prefixes):
    """回傳一組 CPC codes 實際命中的 prefix 清單。"""
    hits = set()

    for code in codes:
        code = normalize_cpc(code)

        for prefix in prefixes:
            if code.startswith(prefix):
                hits.add(prefix)

    return sorted(hits)


def get_cpc_codes(record):
    """從單筆資料取出 CPC codes，並相容 list 或逗號分隔字串格式。"""
    codes = record.get("matched_cpc_codes")

    if codes is None:
        return []

    if isinstance(codes, list):
        return codes

    if isinstance(codes, str):
        return [code.strip() for code in codes.split(",") if code.strip()]

    return []


# 讀取前處理後的 JSON 專利資料。
records = json.loads(src.read_text(encoding="utf-8"))

filtered = []

# 分別統計命中 Domain / Target 的筆數，幫助檢查篩選條件是否過寬或過窄。
domain_hit_count = 0
target_hit_count = 0

for record in records:
    cpc_codes = get_cpc_codes(record)

    # 分別找出命中的 Domain prefix 與 Target prefix。
    domain_hits = matched_prefixes(cpc_codes, DOMAIN_PREFIXES)
    target_hits = matched_prefixes(cpc_codes, TARGET_PREFIXES)

    if domain_hits:
        domain_hit_count += 1

    if target_hits:
        target_hit_count += 1

    # 第一群組 AND 第二群組：
    # 同一筆專利必須同時具備鋼鐵領域 CPC 與碳中和/減碳目標 CPC。
    if domain_hits and target_hits:
        output_record = dict(record)
        output_record["domain_matched_prefixes"] = domain_hits
        output_record["target_matched_prefixes"] = target_hits
        filtered.append(output_record)


# 輸出符合交集條件的資料；allow_nan=False 確保輸出符合標準 JSON。
out.write_text(
    json.dumps(filtered, ensure_ascii=False, indent=2, allow_nan=False),
    encoding="utf-8",
)


# 印出篩選統計，方便確認 Domain、Target 與交集的資料量。
print()
print("=== CPC 第一群組 AND 第二群組篩選結果 ===")
print(f"輸入總筆數：{len(records):,}")
print(f"命中第一群組 Domain 的筆數：{domain_hit_count:,}")
print(f"命中第二群組 Target 的筆數：{target_hit_count:,}")
print(f"同時命中 Domain AND Target 的輸出筆數：{len(filtered):,}")
print(f"輸出檔案：{out}")
print("======================================")
