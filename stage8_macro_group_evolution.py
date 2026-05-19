# =============================================================================
# Stage 8: Macro Group Assignment and Evolution Analysis
# -----------------------------------------------------------------------------
# 功能:
#   1. 讀取 output_specter2_3000/topic_keywords.csv
#   2. 使用 LLM 將每個 Topic_ID 指派到預先定義好的 Macro_Group
#   3. 輸出 topic_macro_groups.csv
#   4. 輸出 topic_macro_groups_review.csv 供人工檢查與修正
#   5. 讀取人工修正後結果，合併回 patent_with_topics.parquet
#   6. 不覆蓋原始 patent_with_topics.parquet
#   7. 輸出 patent_with_macro_groups.parquet
#   8. 計算 Macro Group 的時間演化
# =============================================================================

import os
import re
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# =============================================================================
# 路徑設定
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output_specter2_3000"

TOPIC_KEYWORDS_CSV = OUTPUT_DIR / "topic_keywords.csv"
PATENT_WITH_TOPICS_PARQUET = OUTPUT_DIR / "patent_with_topics.parquet"

TOPIC_MACRO_GROUPS_CSV = OUTPUT_DIR / "topic_macro_groups.csv"
TOPIC_MACRO_GROUPS_REVIEW_CSV = OUTPUT_DIR / "topic_macro_groups_review.csv"
TOPIC_MACRO_GROUPS_REVIEWED_CSV = OUTPUT_DIR / "topic_macro_groups_reviewed.csv"

PATENT_WITH_MACRO_GROUPS_PARQUET = OUTPUT_DIR / "patent_with_macro_groups.parquet"

MACRO_TOPIC_MEMBERS_CSV = OUTPUT_DIR / "macro_group_topic_members.csv"
MACRO_SEGMENT_COUNTS_CSV = OUTPUT_DIR / "macro_group_segment_counts.csv"
MACRO_SEGMENT_SHARES_CSV = OUTPUT_DIR / "macro_group_segment_shares.csv"
MACRO_YEARLY_COUNTS_CSV = OUTPUT_DIR / "macro_group_yearly_counts.csv"
MACRO_YEARLY_SHARES_CSV = OUTPUT_DIR / "macro_group_yearly_shares.csv"
MACRO_EVOLUTION_SUMMARY_CSV = OUTPUT_DIR / "macro_group_evolution_summary.csv"


SEGMENTS = [
    "SEG_A_2006_2010",
    "SEG_B_2011_2015",
    "SEG_C_2016_2020",
    "SEG_D_2021_2025",
]

SEGMENT_MIDPOINTS = np.array([2008, 2013, 2018, 2023], dtype=float)


# =============================================================================
# Macro Group 定義
# =============================================================================

MACRO_GROUPS = {
    "BF_Energy_Gas_Optimization": {
        "zh": "高爐能源效率與爐氣／廢熱回收",
        "definition": (
            "Blast furnace energy efficiency, hot blast stove, blast furnace gas "
            "recovery, gas purification, combustion optimization, waste heat recovery "
            "from blast-furnace-related processes."
        ),
    },
    "Slag_Byproduct_Valorization": {
        "zh": "爐渣與冶金副產物資源化",
        "definition": (
            "Steel slag, converter slag, blast furnace slag, red mud, metallurgical "
            "dust, zinc/lead/vanadium recovery, by-product recycling, slag valorization, "
            "and metal recovery from waste."
        ),
    },
    "EAF_Scrap_Recycling": {
        "zh": "電弧爐與廢鋼循環利用",
        "definition": (
            "Electric arc furnace, electric furnace steelmaking, scrap preheating, "
            "scrap recycling, cyclic EAF process, and EAF process optimization."
        ),
    },
    "LowCarbon_Hydrogen_Ironmaking": {
        "zh": "低碳煉鐵與氫基還原",
        "definition": (
            "Hydrogen-based direct reduction, DRI, sponge iron, hydrogen-rich reducing gas, "
            "low-carbon blast furnace, biomass-based reduction, and alternative reductants "
            "for ironmaking."
        ),
    },
    "CCUS_Carbon_Utilization": {
        "zh": "CCUS 與碳利用技術",
        "definition": (
            "CO2 capture, CO2 sequestration, carbon utilization, steel slag carbonation, "
            "carbon fixation, and molten carbonate fuel cell CO2 capture."
        ),
    },
    "Peripheral_Metallurgy_Materials": {
        "zh": "邊緣冶金與材料回收主題",
        "definition": (
            "Metallurgical or material-related topics that are not clearly part of the five "
            "core carbon-neutral steelmaking pathways, such as alloying, specialty steel "
            "composition, lithium battery metal recovery, manganese alloying, nitrogen control, "
            "or generic hot rolled steel production."
        ),
    },
}


VALID_GROUPS = set(MACRO_GROUPS.keys())


# =============================================================================
# 環境變數
# =============================================================================

def load_openai_key() -> str:
    """
    從 .env 或環境變數讀取 OPENAI_API_KEY。
    """
    if load_dotenv is not None:
        load_dotenv(PROJECT_DIR / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "找不到 OPENAI_API_KEY。請在 /home/carbon/carbon/.env 中設定，"
            "或先 export OPENAI_API_KEY='你的 key'"
        )
    return api_key


# =============================================================================
# LLM Prompt
# =============================================================================

def build_macro_prompt(topic_row: dict) -> str:
    """
    針對單一 topic 建立 LLM 分類 prompt。
    """

    macro_text = "\n".join(
        [
            f"{i + 1}. {name}\n"
            f"Chinese name: {meta['zh']}\n"
            f"Definition: {meta['definition']}"
            for i, (name, meta) in enumerate(MACRO_GROUPS.items())
        ]
    )

    prompt = f"""
You are an expert in carbon-neutral ironmaking, steelmaking, and metallurgy patent classification.

Your task is to assign one BERTopic topic to exactly ONE predefined macro group.

Predefined macro groups:

{macro_text}

Topic information:
Topic_ID: {topic_row.get("Topic_ID", "")}
English_Label: {topic_row.get("English_Label", "")}
Doc_Count: {topic_row.get("Doc_Count", "")}
Top10_Keywords: {topic_row.get("Top10_Keywords", "")}

Classification rules:
- Assign exactly one Macro_Group.
- Use the topic label and Top10 keywords.
- Prefer the most technically specific group.
- If a topic is about slag, dust, red mud, metallurgical waste, or metal recovery from waste, choose Slag_Byproduct_Valorization unless CO2 carbonation or carbon fixation is explicit.
- If a topic is about CO2 capture, sequestration, carbon fixation, or carbonation, choose CCUS_Carbon_Utilization.
- If a topic is about hydrogen, DRI, sponge iron, reducing gas, biomass reductants, or alternative reductants for ironmaking, choose LowCarbon_Hydrogen_Ironmaking.
- If a topic is about electric arc furnace, electric furnace steelmaking, or scrap preheating, choose EAF_Scrap_Recycling.
- If a topic is about blast furnace gas, hot blast, combustion, gas purification, or BF-related waste heat recovery, choose BF_Energy_Gas_Optimization.
- If none of the above fits clearly, choose Peripheral_Metallurgy_Materials.

Return JSON only with this exact schema:
{{
  "Topic_ID": <integer>,
  "Macro_Group": "<one of the predefined macro groups>",
  "Macro_Group_ZH": "<Chinese macro group name>",
  "Confidence": "high | medium | low",
  "Needs_Review": true | false,
  "Reason": "<one short sentence explaining the classification>"
}}
""".strip()

    return prompt


def extract_json_object(text: str) -> dict:
    """
    從 LLM 回覆中抽取 JSON object。
    """
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def classify_one_topic_with_llm(client, topic_row: dict, model: str, max_retries: int = 3) -> dict:
    """
    使用 OpenAI 對單一 topic 指派 Macro_Group。
    """
    prompt = build_macro_prompt(topic_row)

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict patent technology classification assistant. "
                            "You must return valid JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            content = response.choices[0].message.content
            data = extract_json_object(content)

            topic_id = int(topic_row["Topic_ID"])
            macro_group = str(data.get("Macro_Group", "")).strip()

            if macro_group not in VALID_GROUPS:
                raise ValueError(f"Invalid Macro_Group: {macro_group}")

            macro_zh = MACRO_GROUPS[macro_group]["zh"]

            return {
                "Topic_ID": topic_id,
                "English_Label": topic_row.get("English_Label", ""),
                "Doc_Count": topic_row.get("Doc_Count", ""),
                "Top10_Keywords": topic_row.get("Top10_Keywords", ""),
                "Macro_Group": macro_group,
                "Macro_Group_ZH": macro_zh,
                "Confidence": str(data.get("Confidence", "medium")).strip().lower(),
                "Needs_Review": bool(data.get("Needs_Review", False)),
                "Reason": str(data.get("Reason", "")).strip(),
            }

        except Exception as e:
            last_error = e
            wait = 2 * attempt
            print(f"  ⚠ Topic {topic_row.get('Topic_ID')} 第 {attempt} 次失敗: {e}")
            time.sleep(wait)

    topic_id = int(topic_row["Topic_ID"])
    return {
        "Topic_ID": topic_id,
        "English_Label": topic_row.get("English_Label", ""),
        "Doc_Count": topic_row.get("Doc_Count", ""),
        "Top10_Keywords": topic_row.get("Top10_Keywords", ""),
        "Macro_Group": "Peripheral_Metallurgy_Materials",
        "Macro_Group_ZH": MACRO_GROUPS["Peripheral_Metallurgy_Materials"]["zh"],
        "Confidence": "low",
        "Needs_Review": True,
        "Reason": f"LLM classification failed; defaulted to peripheral group. Error: {last_error}",
    }


# =============================================================================
# Step 1–4: LLM 指派 Macro Group
# =============================================================================

def assign_macro_groups_with_llm(
    topic_keywords_path: Path = TOPIC_KEYWORDS_CSV,
    output_path: Path = TOPIC_MACRO_GROUPS_CSV,
    review_path: Path = TOPIC_MACRO_GROUPS_REVIEW_CSV,
    model: str = "gpt-4o-mini",
    sleep_seconds: float = 0.5,
):
    """
    讀取 topic_keywords.csv，使用 LLM 指派 Macro_Group，輸出 topic_macro_groups.csv。
    """

    print("=" * 80)
    print("STEP 1–4: 使用 LLM 指派 Topic_ID 至 Macro_Group")
    print("=" * 80)

    if not topic_keywords_path.exists():
        raise FileNotFoundError(f"找不到 {topic_keywords_path}")

    topic_df = pd.read_csv(topic_keywords_path)

    required_cols = ["Topic_ID", "English_Label", "Doc_Count", "Top10_Keywords"]
    missing = [c for c in required_cols if c not in topic_df.columns]
    if missing:
        raise ValueError(f"topic_keywords.csv 缺少欄位: {missing}")

    topic_df = topic_df[topic_df["Topic_ID"] != -1].copy()
    topic_df["Topic_ID"] = topic_df["Topic_ID"].astype(int)

    api_key = load_openai_key()

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    rows = []
    total = len(topic_df)

    for idx, row in topic_df.iterrows():
        topic_row = row.to_dict()
        tid = int(topic_row["Topic_ID"])

        print(f"[{len(rows) + 1}/{total}] Classifying Topic {tid} ...")

        result = classify_one_topic_with_llm(
            client=client,
            topic_row=topic_row,
            model=model,
        )
        rows.append(result)

        print(
            f"  → {result['Macro_Group']} | "
            f"{result['Confidence']} | review={result['Needs_Review']}"
        )

        time.sleep(sleep_seconds)

    macro_df = pd.DataFrame(rows)
    macro_df = macro_df.sort_values("Topic_ID").reset_index(drop=True)

    macro_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    macro_df.to_csv(review_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ 已輸出 LLM 初步分類: {output_path}")
    print(f"✅ 已輸出人工檢查檔案: {review_path}")
    print("\n請人工檢查 topic_macro_groups_review.csv。")
    print("若有修正，建議另存為 topic_macro_groups_reviewed.csv。")
    print("接著執行：python stage8_macro_group_evolution.py --mode evolve")


# =============================================================================
# Step 5: 讀取人工修正檔案
# =============================================================================

def load_reviewed_macro_groups() -> pd.DataFrame:
    """
    優先讀取 topic_macro_groups_reviewed.csv。
    若不存在，則使用 topic_macro_groups_review.csv。
    若也不存在，使用 topic_macro_groups.csv。
    """

    if TOPIC_MACRO_GROUPS_REVIEWED_CSV.exists():
        path = TOPIC_MACRO_GROUPS_REVIEWED_CSV
        print(f"使用人工修正後檔案: {path}")
    elif TOPIC_MACRO_GROUPS_REVIEW_CSV.exists():
        path = TOPIC_MACRO_GROUPS_REVIEW_CSV
        print(f"找不到 reviewed 檔案，改用 review 檔案: {path}")
    elif TOPIC_MACRO_GROUPS_CSV.exists():
        path = TOPIC_MACRO_GROUPS_CSV
        print(f"找不到 review 檔案，改用 LLM 初步分類檔案: {path}")
    else:
        raise FileNotFoundError(
            "找不到 topic_macro_groups_reviewed.csv、"
            "topic_macro_groups_review.csv 或 topic_macro_groups.csv。"
            "請先執行 --mode assign。"
        )

    macro_df = pd.read_csv(path)
    required_cols = ["Topic_ID", "Macro_Group"]
    missing = [c for c in required_cols if c not in macro_df.columns]
    if missing:
        raise ValueError(f"{path.name} 缺少欄位: {missing}")

    macro_df["Topic_ID"] = macro_df["Topic_ID"].astype(int)
    macro_df["Macro_Group"] = macro_df["Macro_Group"].astype(str).str.strip()

    invalid = sorted(set(macro_df["Macro_Group"]) - VALID_GROUPS)
    if invalid:
        raise ValueError(
            f"人工檢查檔中有不合法 Macro_Group: {invalid}\n"
            f"合法值為: {sorted(VALID_GROUPS)}"
        )

    if "Macro_Group_ZH" not in macro_df.columns:
        macro_df["Macro_Group_ZH"] = macro_df["Macro_Group"].map(
            lambda x: MACRO_GROUPS[x]["zh"]
        )
    else:
        macro_df["Macro_Group_ZH"] = macro_df["Macro_Group"].map(
            lambda x: MACRO_GROUPS[x]["zh"]
        )

    return macro_df


# =============================================================================
# 時間區段與演化分類
# =============================================================================

def ensure_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    確保 patent dataframe 有 priority_year 和 time_segment。
    """

    df = df.copy()

    if "priority_date" not in df.columns:
        raise ValueError("patent_with_topics.parquet 缺少 priority_date 欄位")

    if not np.issubdtype(df["priority_date"].dtype, np.datetime64):
        df["priority_date"] = pd.to_datetime(
            df["priority_date"].astype(str),
            errors="coerce",
        )

    df = df.dropna(subset=["priority_date"])
    df["priority_year"] = df["priority_date"].dt.year

    if "time_segment" not in df.columns:
        def assign_segment(y):
            if 2006 <= y <= 2010:
                return "SEG_A_2006_2010"
            if 2011 <= y <= 2015:
                return "SEG_B_2011_2015"
            if 2016 <= y <= 2020:
                return "SEG_C_2016_2020"
            if 2021 <= y <= 2025:
                return "SEG_D_2021_2025"
            return "OUT_OF_RANGE"

        df["time_segment"] = df["priority_year"].apply(assign_segment)

    df = df[df["time_segment"].isin(SEGMENTS)].copy()

    return df


def classify_macro_trajectory(freq, shares):
    """
    與 topic_evolution_summary 類似的軌跡分類邏輯。
    """
    first, second, third, latest = freq
    early = first + second
    late = third + latest
    growth = (late - early) / (early + 1)
    share_change = shares[-1] - shares[0]
    peak_index = int(np.argmax(freq))

    if latest >= max(freq[:3]) and growth > 0.3:
        return "新興上升"
    if peak_index <= 1 and growth < -0.3:
        return "早期高峰後衰退"
    if peak_index == 2 and latest < third:
        return "中期高峰後回落"
    if share_change > 0.03:
        return "占比提升"
    if share_change < -0.03:
        return "占比下降"
    return "相對穩定"


# =============================================================================
# Step 6–7: 合併回 patent data + 計算 Macro Group 時間演化
# =============================================================================

def merge_and_compute_macro_evolution():
    """
    合併 Macro Group 至 patent_with_topics.parquet，
    輸出 patent_with_macro_groups.parquet，
    並計算 macro group 的時間演化。
    """

    print("=" * 80)
    print("STEP 6–7: 合併 Macro Group 並計算時間演化")
    print("=" * 80)

    if not PATENT_WITH_TOPICS_PARQUET.exists():
        raise FileNotFoundError(f"找不到 {PATENT_WITH_TOPICS_PARQUET}")

    macro_df = load_reviewed_macro_groups()

    patent_df = pd.read_parquet(PATENT_WITH_TOPICS_PARQUET)

    if "topic_id" not in patent_df.columns:
        raise ValueError("patent_with_topics.parquet 缺少 topic_id 欄位")

    patent_df = ensure_time_fields(patent_df)
    patent_df["topic_id"] = patent_df["topic_id"].astype(int)

    merge_cols = ["Topic_ID", "Macro_Group", "Macro_Group_ZH"]
    extra_cols = [c for c in ["Confidence", "Needs_Review", "Reason"] if c in macro_df.columns]
    merge_df = macro_df[merge_cols + extra_cols].copy()

    merged = patent_df.merge(
        merge_df,
        how="left",
        left_on="topic_id",
        right_on="Topic_ID",
    )

    # 處理 noise 或未匹配 topic
    merged["Macro_Group"] = merged["Macro_Group"].fillna("NOISE_OR_UNASSIGNED")
    merged["Macro_Group_ZH"] = merged["Macro_Group_ZH"].fillna("雜訊或未分配主題")

    # 不覆蓋原始 patent_with_topics.parquet
    merged.to_parquet(PATENT_WITH_MACRO_GROUPS_PARQUET, index=False)
    print(f"✅ 已輸出新檔案，不覆蓋原始 parquet: {PATENT_WITH_MACRO_GROUPS_PARQUET}")

    # topic 層級對照表
    topic_members = (
        macro_df
        .sort_values(["Macro_Group", "Topic_ID"])
        .reset_index(drop=True)
    )
    topic_members.to_csv(MACRO_TOPIC_MEMBERS_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 已輸出 Macro Group topic 對照表: {MACRO_TOPIC_MEMBERS_CSV}")

    # 排除 NOISE_OR_UNASSIGNED 後進行主要演化分析
    analysis_df = merged[merged["Macro_Group"] != "NOISE_OR_UNASSIGNED"].copy()

    # ── segment counts ─────────────────────────────────────────────
    segment_counts = pd.crosstab(
        analysis_df["Macro_Group"],
        analysis_df["time_segment"],
    )
    segment_counts = segment_counts.reindex(columns=SEGMENTS, fill_value=0)
    segment_counts.to_csv(MACRO_SEGMENT_COUNTS_CSV, encoding="utf-8-sig")

    # ── segment shares ─────────────────────────────────────────────
    segment_totals = (
        analysis_df["time_segment"]
        .value_counts()
        .reindex(SEGMENTS, fill_value=0)
    )
    segment_shares = segment_counts.div(
        segment_totals.replace(0, np.nan),
        axis=1,
    ).fillna(0)
    segment_shares.to_csv(MACRO_SEGMENT_SHARES_CSV, encoding="utf-8-sig")

    print(f"✅ 已輸出 Macro Group segment counts: {MACRO_SEGMENT_COUNTS_CSV}")
    print(f"✅ 已輸出 Macro Group segment shares: {MACRO_SEGMENT_SHARES_CSV}")

    # ── yearly counts / shares ─────────────────────────────────────
    yearly_counts = pd.crosstab(
        analysis_df["priority_year"],
        analysis_df["Macro_Group"],
    )
    yearly_counts = yearly_counts.reindex(range(2006, 2026), fill_value=0)
    yearly_counts.to_csv(MACRO_YEARLY_COUNTS_CSV, encoding="utf-8-sig")

    yearly_totals = yearly_counts.sum(axis=1).replace(0, np.nan)
    yearly_shares = yearly_counts.div(yearly_totals, axis=0).fillna(0)
    yearly_shares.to_csv(MACRO_YEARLY_SHARES_CSV, encoding="utf-8-sig")

    print(f"✅ 已輸出 Macro Group yearly counts: {MACRO_YEARLY_COUNTS_CSV}")
    print(f"✅ 已輸出 Macro Group yearly shares: {MACRO_YEARLY_SHARES_CSV}")

    # ── evolution summary ──────────────────────────────────────────
    rows = []

    for macro_group in sorted(segment_counts.index):
        freq = [int(segment_counts.loc[macro_group, s]) for s in SEGMENTS]
        shares = [float(segment_shares.loc[macro_group, s]) for s in SEGMENTS]

        slope = float(np.polyfit(SEGMENT_MIDPOINTS, shares, 1)[0])
        peak_idx = int(np.argmax(freq))

        rows.append({
            "Macro_Group": macro_group,
            "Macro_Group_ZH": MACRO_GROUPS.get(
                macro_group, {"zh": "未知"}
            )["zh"],
            "Topic_Count": int(
                macro_df[macro_df["Macro_Group"] == macro_group]["Topic_ID"].nunique()
            ),
            "Doc_Count": int(sum(freq)),
            "Freq_2006_2010": freq[0],
            "Freq_2011_2015": freq[1],
            "Freq_2016_2020": freq[2],
            "Freq_2021_2025": freq[3],
            "Share_2006_2010": round(shares[0], 4),
            "Share_2011_2015": round(shares[1], 4),
            "Share_2016_2020": round(shares[2], 4),
            "Share_2021_2025": round(shares[3], 4),
            "Share_Change_2006_to_2025": round(shares[-1] - shares[0], 4),
            "Share_Slope_Per_Year": round(slope, 6),
            "Peak_Segment": SEGMENTS[peak_idx],
            "Trajectory": classify_macro_trajectory(freq, shares),
        })

    evolution_df = pd.DataFrame(rows)
    evolution_df = evolution_df.sort_values(
        ["Doc_Count", "Share_2021_2025"],
        ascending=False,
    ).reset_index(drop=True)

    evolution_df.to_csv(
        MACRO_EVOLUTION_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"✅ 已輸出 Macro Group evolution summary: {MACRO_EVOLUTION_SUMMARY_CSV}")

    print("\nMacro Group 演化摘要:")
    print(
        evolution_df[
            [
                "Macro_Group_ZH",
                "Doc_Count",
                "Freq_2006_2010",
                "Freq_2011_2015",
                "Freq_2016_2020",
                "Freq_2021_2025",
                "Share_Change_2006_to_2025",
                "Trajectory",
            ]
        ].to_string(index=False)
    )


# =============================================================================
# 主程式
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 8: Assign BERTopic topics to macro groups and compute macro group evolution."
    )
    parser.add_argument(
        "--mode",
        choices=["assign", "evolve", "all"],
        default="all",
        help=(
            "assign: only call LLM and create topic_macro_groups.csv; "
            "evolve: use reviewed macro groups and compute evolution; "
            "all: run assign then evolve directly."
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model for macro group assignment.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Sleep seconds between LLM calls.",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in ["assign", "all"]:
        assign_macro_groups_with_llm(
            model=args.model,
            sleep_seconds=args.sleep,
        )

    if args.mode == "assign":
        print("\n下一步：人工檢查 topic_macro_groups_review.csv。")
        print("修正後可另存為 topic_macro_groups_reviewed.csv。")
        print("然後執行：")
        print("python stage8_macro_group_evolution.py --mode evolve")
        return

    if args.mode in ["evolve", "all"]:
        merge_and_compute_macro_evolution()

    print("\n" + "=" * 80)
    print("Stage 8 全流程完成")
    print("=" * 80)
    print(f"輸出目錄: {OUTPUT_DIR}")
    print("主要輸出:")
    print("  ├── topic_macro_groups.csv")
    print("  ├── topic_macro_groups_review.csv")
    print("  ├── patent_with_macro_groups.parquet")
    print("  ├── macro_group_topic_members.csv")
    print("  ├── macro_group_segment_counts.csv")
    print("  ├── macro_group_segment_shares.csv")
    print("  ├── macro_group_yearly_counts.csv")
    print("  ├── macro_group_yearly_shares.csv")
    print("  └── macro_group_evolution_summary.csv")


if __name__ == "__main__":
    main()