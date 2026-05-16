import json
import re
from pathlib import Path

# =============================================================================
# 依 CPC 代碼篩選「鋼鐵領域」且「碳中和/減碳目標」相關的專利
#
# 篩選順序(嚴格依此順序執行):
#   步驟一、時間過濾:priority_date 年份 ∈ [2006, 2025]
#           - 空值、None、0、"0"、"00000000" 等一律跳過
#           - 必須最先執行,後續所有計算都以時間過濾後的母體為準
#   步驟二、CPC 篩選:
#           減碳目標 CPC
#           AND
#           C21B/C21C  (鋼鐵領域 CPC)
# =============================================================================

# 輸入資料:前一步依 abstract_en 去重後的資料。
src = Path("/home/carbon/carbon/data_globalmorecpc/global_abstract_dedup.json")
# 輸出資料:同時命中減碳目標 CPC 與鋼鐵領域 CPC 的資料。
out = Path("/home/carbon/carbon/data_globalmorecpc/global_onlycpc_domain_target_intersection.json")


# =============================================================================
# 時間過濾設定
# =============================================================================
PRIORITY_YEAR_MIN = 2006
PRIORITY_YEAR_MAX = 2025


# 第一群組:鋼鐵 / 鐵製造相關 CPC 前綴。
DOMAIN_PREFIXES = [
    "C21B",  # 高爐與鐵製造
    "C21C",  # 煉鋼精煉
]

# 第二群組:碳中和、製程減碳、CO2 捕集/處理、氣體分離相關 CPC 前綴。
TARGET_PREFIXES = [
    # B01D:氣體分離 / 廢氣淨化 / CO2 捕集
    "B01D53",
    "B01D2053",

    # Y02C:溫室氣體捕捉或處置
    "Y02C20/40",

    # Y02P10:金屬加工減碳
    "Y02P10/10",
    "Y02P10/122",
    "Y02P10/134",
    "Y02P10/143",
    "Y02P10/146",
    "Y02P10/20",
    "Y02P10/25",
    "Y02P10/32",

    # Y02P20:化工產業減碳
    "Y02P20/129",
    "Y02P20/143",
    "Y02P20/145",
    "Y02P20/151",
    "Y02P20/582",
    "Y02P20/584",

    # Y02P40:礦物加工減碳
    "Y02P40/10",
    "Y02P40/121",
    "Y02P40/125",
    "Y02P40/18",

    # Y02P70:最終產品製程減碳
    "Y02P70/10",

    # Y02P80:跨產業能源效率、廢棄物減量、材料節省
    "Y02P80/10",
    "Y02P80/15",
    "Y02P80/30",
    "Y02P80/40",

    # Y02P90:智慧製造、氫能、燃料電池、能源管理、GHG 管理
    "Y02P90/02",
    "Y02P90/30",
    "Y02P90/40",
    "Y02P90/45",
    "Y02P90/50",
    "Y02P90/80",
    "Y02P90/82",
    "Y02P90/84",
    "Y02P90/845",
]


# =============================================================================
# 時間過濾相關函式
# =============================================================================

# 用於從 priority_date 字串開頭抓 4 位年份的 regex。
# 涵蓋常見格式:"20020709"、"2002-07-09"、"2002/07/09"、"2002.07.09"、"2002"
_YEAR_PREFIX_RE = re.compile(r"(\d{4})")


def extract_priority_year(record):
    """從 record 的 priority_date 欄位取出 4 位年份。

    以下情況一律視為無效,回傳 None(不會報錯):
        - 欄位不存在 / None
        - 空字串 / 全空白字串
        - 0 / "0" / "00000000" 等全為 0 的值
        - 解析後年份不在 1900~2100 合理範圍

    支援格式:
        - "20020709" 或 20020709(int)→ 2002
        - "2002-07-09" / "2002/07/09" / "2002.07.09" → 2002
        - "2002" → 2002
    """
    raw = record.get("priority_date")

    # (1) 缺失值。
    if raw is None:
        return None

    # (2) 數值 0(int 或 float)。
    if isinstance(raw, (int, float)) and raw == 0:
        return None

    # (3) 轉字串並去頭尾空白。
    s = str(raw).strip()
    if not s:
        return None

    # (4) 純 0 或全 0 字串(例如 "0"、"00000000"、"0000-00-00")。
    digits_only = re.sub(r"\D", "", s)
    if not digits_only or set(digits_only) == {"0"}:
        return None

    # (5) 抓字串開頭的 4 位數字當年份。
    match = _YEAR_PREFIX_RE.match(s)
    if not match:
        return None

    year = int(match.group(1))

    # (6) 合理性檢查,避免年份為 0001、9999 之類雜訊。
    if year < 1900 or year > 2100:
        return None

    return year


# =============================================================================
# CPC 處理相關函式
# =============================================================================

def normalize_cpc(code):
    """統一 CPC 字串格式:轉字串、去前後空白、轉大寫。"""
    return str(code).strip().upper()


def matched_prefixes(codes, prefixes):
    """回傳一組 CPC 代碼實際命中的前綴清單。"""
    hits = set()

    for code in codes:
        code = normalize_cpc(code)

        for prefix in prefixes:
            if code.startswith(prefix):
                hits.add(prefix)

    return sorted(hits)


def get_cpc_codes(record):
    """從單筆資料取出 CPC 代碼,並相容 list 或逗號分隔字串格式。"""
    codes = record.get("matched_cpc_codes")

    if codes is None:
        return []

    if isinstance(codes, list):
        return codes

    if isinstance(codes, str):
        return [code.strip() for code in codes.split(",") if code.strip()]

    return []


# =============================================================================
# 主流程
# =============================================================================

# 讀取前處理後的 JSON 專利資料。
records = json.loads(src.read_text(encoding="utf-8"))
total_input = len(records)


# -----------------------------------------------------------------------------
# 步驟一:時間過濾(必須最先執行)
#   priority_date 年份 ∈ [PRIORITY_YEAR_MIN, PRIORITY_YEAR_MAX]
#   空值、0、無法解析者一律跳過,不會進入後續 CPC 篩選。
# -----------------------------------------------------------------------------
records_in_range = []
year_missing = 0       # priority_date 缺失/空值/0/無法解析
year_out_of_range = 0  # 年份在區間外

for record in records:
    year = extract_priority_year(record)

    # 空值或無效值:跳過,不報錯。
    if year is None:
        year_missing += 1
        continue

    # 年份不在區間內:跳過。
    if year < PRIORITY_YEAR_MIN or year > PRIORITY_YEAR_MAX:
        year_out_of_range += 1
        continue

    # 通過時間過濾。
    records_in_range.append(record)


# -----------------------------------------------------------------------------
# 步驟二:CPC 篩選(只跑時間範圍內的資料)
# -----------------------------------------------------------------------------
filtered = []

# 各條件命中筆數(以時間過濾後的母體計算)。
domain_hit_count = 0
target_hit_count = 0
steel_domain_cpc_count = 0

for record in records_in_range:
    cpc_codes = get_cpc_codes(record)

    domain_hits = matched_prefixes(cpc_codes, DOMAIN_PREFIXES)
    target_hits = matched_prefixes(cpc_codes, TARGET_PREFIXES)

    if domain_hits:
        domain_hit_count += 1
    if target_hits:
        target_hit_count += 1

    is_steel_domain = bool(domain_hits)

    if is_steel_domain:
        steel_domain_cpc_count += 1

    # 篩選條件:減碳目標 CPC AND 鋼鐵領域 CPC。
    if is_steel_domain and target_hits:
        output_record = dict(record)
        output_record["priority_year"] = extract_priority_year(record)
        output_record["domain_matched_prefixes"] = domain_hits
        output_record["target_matched_prefixes"] = target_hits

        output_record["domain_source"] = "cpc_domain"

        filtered.append(output_record)


# 輸出符合條件的資料。
out.write_text(
    json.dumps(filtered, ensure_ascii=False, indent=2, allow_nan=False),
    encoding="utf-8",
)


# =============================================================================
# 印出篩選統計
# =============================================================================

# 計算最終輸出的年份分布。
year_distribution = {}
for record in filtered:
    year = record.get("priority_year")
    if year is not None:
        year_distribution[year] = year_distribution.get(year, 0) + 1

print()
print("=== 鋼鐵場域 AND 減碳目標 CPC 篩選結果 ===")
print(f"輸入總筆數:{total_input:,}")
print()
print(f"[步驟一] 時間過濾:priority_date 年份 ∈ [{PRIORITY_YEAR_MIN}, {PRIORITY_YEAR_MAX}]")
print(f"  通過:{len(records_in_range):,}")
print(f"  剔除(priority_date 缺失/空值/0/無法解析):{year_missing:,}")
print(f"  剔除(年份在區間外):{year_out_of_range:,}")
print()
print("[步驟二] 各條件命中筆數(以時間過濾後母體計算,彼此可能重複)")
print(f"  命中鋼鐵領域 CPC (C21B/C21C):{domain_hit_count:,}")
print(f"  命中減碳目標 CPC:{target_hit_count:,}")
print()
print("[鋼鐵場域條件]")
print(f"  鋼鐵領域 CPC:{steel_domain_cpc_count:,}")
print()
print("[最終輸出]")
print(f"  減碳目標 CPC AND 鋼鐵領域 CPC AND 年份範圍:{len(filtered):,}")
if year_distribution:
    print(f"  年份分布:")
    for year in sorted(year_distribution):
        print(f"    {year}: {year_distribution[year]:,}")
print(f"  輸出檔案:{out}")
print("======================================")
