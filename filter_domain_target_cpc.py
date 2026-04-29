import json
from pathlib import Path

src = Path("/home/carbon/carbon/data/dedup_by_abstract_2.json")
out = Path("/home/carbon/carbon/data/cpc_domain_target_intersection_3.json")


DOMAIN_PREFIXES = [
    "C21B",  # 高爐與鐵製造
    "C21C",  # 煉鋼精煉
]

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
    return str(code).strip().upper()


def match_prefix(code, prefixes):
    code = normalize_cpc(code)
    return any(code.startswith(prefix) for prefix in prefixes)


def matched_prefixes(codes, prefixes):
    hits = set()

    for code in codes:
        code = normalize_cpc(code)

        for prefix in prefixes:
            if code.startswith(prefix):
                hits.add(prefix)

    return sorted(hits)


def get_cpc_codes(record):
    codes = record.get("matched_cpc_codes")

    if codes is None:
        return []

    if isinstance(codes, list):
        return codes

    if isinstance(codes, str):
        return [code.strip() for code in codes.split(",") if code.strip()]

    return []


records = json.loads(src.read_text(encoding="utf-8"))

filtered = []

domain_hit_count = 0
target_hit_count = 0

for record in records:
    cpc_codes = get_cpc_codes(record)

    domain_hits = matched_prefixes(cpc_codes, DOMAIN_PREFIXES)
    target_hits = matched_prefixes(cpc_codes, TARGET_PREFIXES)

    if domain_hits:
        domain_hit_count += 1

    if target_hits:
        target_hit_count += 1

    # 第一群組 AND 第二群組
    if domain_hits and target_hits:
        output_record = dict(record)
        output_record["domain_matched_prefixes"] = domain_hits
        output_record["target_matched_prefixes"] = target_hits
        filtered.append(output_record)


out.write_text(
    json.dumps(filtered, ensure_ascii=False, indent=2, allow_nan=False),
    encoding="utf-8",
)


print()
print("=== CPC 第一群組 AND 第二群組篩選結果 ===")
print(f"輸入總筆數：{len(records):,}")
print(f"命中第一群組 Domain 的筆數：{domain_hit_count:,}")
print(f"命中第二群組 Target 的筆數：{target_hit_count:,}")
print(f"同時命中 Domain AND Target 的輸出筆數：{len(filtered):,}")
print(f"輸出檔案：{out}")
print("======================================")
