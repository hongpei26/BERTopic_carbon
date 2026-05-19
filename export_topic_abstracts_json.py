import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output_specter2_3000"

DEFAULT_PATENT_PATH = OUTPUT_DIR / "patent_with_topics.parquet"
DEFAULT_KEYWORDS_PATH = OUTPUT_DIR / "topic_keywords.csv"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "topic_patent_abstracts.json"


def clean_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def build_topic_lookup(topic_keywords_path: Path) -> dict:
    if not topic_keywords_path.exists():
        return {}

    topic_df = pd.read_csv(topic_keywords_path)
    topic_df["Topic_ID"] = topic_df["Topic_ID"].astype(int)

    lookup = {}
    for _, row in topic_df.iterrows():
        tid = int(row["Topic_ID"])
        lookup[tid] = {
            "topic_id": tid,
            "english_label": clean_value(row.get("English_Label")),
            "doc_count": int(row.get("Doc_Count", 0)),
            "top10_keywords": clean_value(row.get("Top10_Keywords")),
        }
    return lookup


def export_topic_abstracts(
    patent_path: Path,
    topic_keywords_path: Path,
    output_path: Path,
    include_noise: bool = False,
    max_per_topic: int | None = None,
) -> None:
    if not patent_path.exists():
        raise FileNotFoundError(f"Cannot find patent topic file: {patent_path}")

    df = pd.read_parquet(patent_path)
    if "topic_id" not in df.columns:
        raise ValueError("patent_with_topics.parquet must contain topic_id")
    if "abstract_clean" not in df.columns:
        raise ValueError("patent_with_topics.parquet must contain abstract_clean")

    df = df.copy()
    df["topic_id"] = df["topic_id"].astype(int)

    if not include_noise:
        df = df[df["topic_id"] != -1].copy()

    topic_lookup = build_topic_lookup(topic_keywords_path)
    topics = []

    for topic_id, topic_df in df.groupby("topic_id", sort=True):
        topic_id = int(topic_id)
        topic_df = topic_df.sort_values(
            [c for c in ["priority_date", "application_number"] if c in topic_df.columns]
        )
        if max_per_topic is not None:
            topic_df = topic_df.head(max_per_topic)

        topic_meta = topic_lookup.get(
            topic_id,
            {
                "topic_id": topic_id,
                "english_label": "NOISE" if topic_id == -1 else "",
                "doc_count": int(len(topic_df)),
                "top10_keywords": "",
            },
        ).copy()

        patents = []
        for _, row in topic_df.iterrows():
            patents.append(
                {
                    "application_number": clean_value(row.get("application_number")),
                    "publication_number": clean_value(row.get("publication_number")),
                    "priority_date": clean_value(row.get("priority_date")),
                    "time_segment": clean_value(row.get("time_segment")),
                    "title": clean_value(row.get("title_clean")),
                    "abstract": clean_value(row.get("abstract_clean")),
                }
            )

        topic_meta["exported_doc_count"] = len(patents)
        topic_meta["patents"] = patents
        topics.append(topic_meta)

    payload = {
        "source_patent_file": str(patent_path),
        "source_topic_keywords_file": str(topic_keywords_path),
        "include_noise": include_noise,
        "max_per_topic": max_per_topic,
        "topic_count": len(topics),
        "topics": topics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(topics)} topics to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export each BERTopic topic and its patent abstracts to JSON."
    )
    parser.add_argument("--patent-path", type=Path, default=DEFAULT_PATENT_PATH)
    parser.add_argument("--topic-keywords-path", type=Path, default=DEFAULT_KEYWORDS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--include-noise",
        action="store_true",
        help="Include topic_id=-1 noise documents in the JSON output.",
    )
    parser.add_argument(
        "--max-per-topic",
        type=int,
        default=None,
        help="Export only the first N patent abstracts per topic.",
    )
    args = parser.parse_args()

    export_topic_abstracts(
        patent_path=args.patent_path,
        topic_keywords_path=args.topic_keywords_path,
        output_path=args.output,
        include_noise=args.include_noise,
        max_per_topic=args.max_per_topic,
    )


if __name__ == "__main__":
    main()
