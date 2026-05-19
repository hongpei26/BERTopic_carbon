import json
from pathlib import Path


SRC = Path(
    "/home/carbon/carbon/data_global_v2/"
    "Carbon_onlycpc_global_morecpc_v2/"
    "global_onlycpc_domain_target_intersection.json"
)

OUT_DIR = SRC.parent
OUT_TOP1 = OUT_DIR / "global_onlycpc_domain_target_intersection_top1.json"
OUT_TOP2 = OUT_DIR / "global_onlycpc_domain_target_intersection_top2.json"

KEEP_COLUMNS = [
    "publication_number",
    "application_number",
    "country_code",
    "kind_code",
    "priority_date",
    "title_en",
    "abstract_en",
    "matched_cpc_codes",
]


def keep_selected_fields(record):
    return {column: record.get(column) for column in KEEP_COLUMNS}


def main():
    print(f"Reading: {SRC}")
    records = json.loads(SRC.read_text(encoding="utf-8"))

    if isinstance(records, dict):
        records = list(records.values())

    total = len(records)
    midpoint = (total + 1) // 2

    top1 = [keep_selected_fields(record) for record in records[:midpoint]]
    top2 = [keep_selected_fields(record) for record in records[midpoint:]]

    OUT_TOP1.write_text(
        json.dumps(top1, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    OUT_TOP2.write_text(
        json.dumps(top2, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Total records: {total:,}")
    print(f"Top1 records: {len(top1):,} -> {OUT_TOP1}")
    print(f"Top2 records: {len(top2):,} -> {OUT_TOP2}")


if __name__ == "__main__":
    main()
