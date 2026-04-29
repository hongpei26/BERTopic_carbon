import json
from collections import Counter
from pathlib import Path


INPUT_PATH = Path(
    "/home/carbon/carbon/data/part-000000000000_carbon_neutral_keywords.json"
)


def extract_year(value) -> int | None:
    if value is None:
        return None

    text = str(value).strip()
    if len(text) < 4 or not text[:4].isdigit():
        return None

    return int(text[:4])


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)

    year_counts = Counter()
    invalid_count = 0

    for record in records:
        year = extract_year(record.get("priority_date"))
        if year is None:
            invalid_count += 1
            continue
        year_counts[year] += 1

    range_total = 0

    for year in range(2006, 2026):
        count = year_counts.get(year, 0)
        range_total += count
        print(f"{year}: {count}筆")

    print()
    print(f"整體數量: {len(records)}筆")
    print(f"2006-2025數量: {range_total}筆")
    print(f"無法判讀priority_date年份: {invalid_count}筆")


if __name__ == "__main__":
    main()
