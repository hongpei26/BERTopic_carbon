# =============================================================================
# 碳捕捉專利 BERTopic 動態主題建模 Pipeline(針對碳中和領域)
# Stage 1–7：資料載入 → SciBERT → UMAP → HDBSCAN → c-TF-IDF → 主題優化 → TOT
# =============================================================================

# ── 安裝套件（首次執行時取消註解）──────────────────────────────────────────
# pip install bertopic sentence-transformers umap-learn hdbscan pandas pyarrow

import re
import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
import os
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None



from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"


def configure_huggingface_token() -> str | None:
    """
    從 .env 載入 Hugging Face token。
    支援標準名稱 HF_TOKEN / HUGGINGFACE_HUB_TOKEN，也相容既有的 HF_API_TOKEN。
    """
    if load_dotenv is not None:
        load_dotenv(PROJECT_DIR / ".env")
    else:
        env_path = PROJECT_DIR / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_API_TOKEN")
    )

    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)

    return token


# =============================================================================
# 共用停用詞設定
# =============================================================================

# 專利法律套語、英語功能詞，以及本資料集中高頻但低辨識力的說明詞。
# 保留 carbon、gas、water、catalyst、filter、membrane 等技術詞，避免削弱主題語意。
PATENT_STOPWORDS = {
    "abstract", "according", "apparatus", "background", "claim", "claims",
    "comprise", "comprises", "comprising", "configure", "configured",
    "configuring", "description", "described", "device", "disclosed",
    "disclosure", "embodiment", "embodiments", "example", "examples", "fig",
    "figure", "field", "generally", "herein", "hereinafter", "include",
    "includes", "including", "invention", "least", "method", "methods",
    "obtain", "obtained", "optionally", "particularly", "plurality",
    "preferably", "present", "process", "provide", "provided", "providing",
    "relate", "relates", "said", "second", "specifically", "step", "steps",
    "substantially", "summary", "system", "thereby", "therefrom", "thereof",
    "third", "typically", "unit", "use", "used", "using", "usually",
    "wherein",
}

CORPUS_STOPWORDS = {
    "based", "can", "containing", "contains", "each", "first", "formed",
    "forming", "having", "one", "portion", "two",
    # 專利裝置描述中常見的位置/形狀詞，容易讓 Topic 0 變成結構詞集合。
    "end", "ends", "opening", "openings", "wall", "walls", "inlet", "inlets",
    "outlet", "outlets", "upper", "lower", "inner", "outer", "side", "sides",
    "surface", "surfaces", "member", "members",
}

CUSTOM_STOPWORDS = sorted(
    set(ENGLISH_STOP_WORDS) | PATENT_STOPWORDS | CORPUS_STOPWORDS
)


# =============================================================================
# STAGE 1：數據載入與前處理 (Data Input)
# =============================================================================

def stage1_load_and_preprocess(input_path: str):
    """
    讀取 JSON 或 Parquet 格式專利資料集，提取 abstract_en 並進行文字前處理。
    確保每筆摘要與 application_number、priority_date 精準對應。
    """

    print("=" * 60)
    print("STAGE 1：數據載入與前處理")
    print("=" * 60)

    # ── 1.1 讀取資料 ──────────────────────────────────────────────────────
    data_path = Path(input_path)

    if data_path.is_file() and data_path.suffix.lower() == ".json":
        print(f"讀取 JSON 檔案：{data_path}")
        with data_path.open("r", encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))
    else:
        parquet_files = list(data_path.glob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"在 {input_path} 找不到任何 .parquet 檔案")

        print(f"找到 {len(parquet_files)} 個 Parquet 檔案，開始讀取...")

        df = pd.concat(
            [pd.read_parquet(f) for f in parquet_files],
            ignore_index=True
        )
    print(f"原始資料筆數：{len(df):,}")

    # ── 1.2 保留核心欄位 ──────────────────────────────────────────────────
    required_cols = ["application_number", "priority_date", "abstract_en"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要欄位：{missing}，請確認資料集欄位名稱")

    optional_cols = [
        "publication_number",
        "title_en",
        "matched_cpc_codes",
        "domain_matched_prefixes",
        "target_matched_prefixes",
    ]
    keep_cols = required_cols + [c for c in optional_cols if c in df.columns]
    df = df[keep_cols].copy()

    # ── 1.3 移除空值與空字串 ──────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["abstract_en"])
    df = df[df["abstract_en"].str.strip() != ""]
    print(f"移除 Abstract 空值：{before - len(df):,} 筆 → 剩餘 {len(df):,} 筆")

    # ── 1.4 文字前處理 ────────────────────────────────────────────────────
    print("執行文字前處理（轉小寫、去停用詞、去雜訊）...")

    def clean_abstract(text: str) -> str:
        """
        清洗單筆 Abstract：
        1. 轉小寫
        2. 移除數字與特殊符號（保留字母與空白）
        3. 移除多餘空白
        4. 移除專利法律套語停用詞
        """
        # 轉小寫
        text = text.lower()
        # 移除非字母字元（保留空白）
        text = re.sub(r"[^a-z\s]", " ", text)
        # 移除多餘空白
        text = re.sub(r"\s+", " ", text).strip()
        # 移除停用詞（完整單字比對）
        tokens = [t for t in text.split() if t not in CUSTOM_STOPWORDS]
        return " ".join(tokens)

    df["abstract_clean"] = df["abstract_en"].apply(clean_abstract)

    # ── 1.5 移除清洗後過短的摘要（< 30 個單詞）────────────────────────────
    before = len(df)
    df["word_count"] = df["abstract_clean"].apply(lambda x: len(x.split()))
    df = df[df["word_count"] >= 30]
    print(f"移除單詞數 < 30 的摘要：{before - len(df):,} 筆 → 剩餘 {len(df):,} 筆")

    # ── 1.6 priority_date 型別轉換 ────────────────────────────────────────
    df["priority_date"] = pd.to_datetime(
        df["priority_date"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )
    df = df.dropna(subset=["priority_date"])

    # ── 1.7 時間區段標記（依 priority_date）──────────────────────────────
    def assign_segment(date):
        year = date.year
        if 2006 <= year <= 2010:   return "SEG_A_2006_2010"
        elif 2011 <= year <= 2015: return "SEG_B_2011_2015"
        elif 2016 <= year <= 2020: return "SEG_C_2016_2020"
        elif 2021 <= year <= 2025: return "SEG_D_2021_2025"
        else:                       return "OUT_OF_RANGE"

    df["time_segment"] = df["priority_date"].apply(assign_segment)
    df = df[df["time_segment"] != "OUT_OF_RANGE"].reset_index(drop=True)

    abstracts = df["abstract_clean"].tolist()

    print("\n各時間區段樣本數：")
    print(df["time_segment"].value_counts().sort_index().to_string())
    print(f"\nSTAGE 1 完成，最終語料筆數：{len(df):,}\n")

    return df, abstracts


# =============================================================================
# STAGE 2：科學語義嵌入 (Scientific Embedding - SciBERT)
# =============================================================================

def stage2_embed(abstracts: list, batch_size: int = 64):
    """
    使用 SciBERT 將 Abstract 轉化為 768 維語義向量。
    精準捕捉冶金、化學反應（還原、電解）與工業術語。
    """

    print("=" * 60)
    print("STAGE 2：SciBERT 語義嵌入")
    print("=" * 60)

    model_name = "allenai/scibert_scivocab_uncased"
    print(f"載入模型：{model_name}")
    print("（首次執行需下載模型約 440MB，請耐心等候）\n")

    hf_token = configure_huggingface_token()
    if hf_token:
        print("已從 .env / 環境變數載入 Hugging Face token")

    embedding_model = SentenceTransformer(model_name, token=hf_token)

    print(f"開始嵌入 {len(abstracts):,} 筆摘要（batch_size={batch_size}）...")
    embeddings = embedding_model.encode(
        abstracts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"\nSTAGE 2 完成，嵌入矩陣維度：{embeddings.shape}")
    print(f"（{embeddings.shape[0]:,} 筆 × {embeddings.shape[1]} 維）\n")

    return embedding_model, embeddings


# =============================================================================
# STAGE 3：非線性降維 (Dimensionality Reduction - UMAP)
# =============================================================================

def stage3_umap() -> UMAP:
    """
    UMAP 將 768 維語義向量壓縮至 5 維。
    使用餘弦相似度確保技術語義相近的專利在低維空間仍保持鄰近關係。
    """

    print("=" * 60)
    print("STAGE 3：UMAP 非線性降維設定")
    print("=" * 60)

    umap_model = UMAP(
        n_neighbors=15,     # 平衡局部技術路線與全域語義結構
        n_components=5,     # 壓縮至 5 維
        min_dist=0.0,       # 允許緊密聚類
        metric="cosine",    # 以餘弦相似度保留語義鄰近關係
        random_state=42
    )

    print("UMAP 設定：n_neighbors=15 | n_components=5 | metric=cosine\n")
    return umap_model


# =============================================================================
# STAGE 4：自動主題聚類 (Clustering - HDBSCAN)
# =============================================================================

def stage4_hdbscan() -> HDBSCAN:
    """
    HDBSCAN 密度聚類，自動識別主題數量。
    語義不明確的零散專利歸類為雜訊（Topic = -1）。
    """

    print("=" * 60)
    print("STAGE 4：HDBSCAN 自動主題聚類設定")
    print("=" * 60)

    hdbscan_model = HDBSCAN(
        min_cluster_size=15,             # 平衡門檻：比 20 更容易捕捉氫冶金/CCS 小群，比 12 更不易過度纔分
        min_samples=5,                   # 減少雜訊，使更多語意相近專利被納入主題
        metric="euclidean",
        cluster_selection_method="leaf", # leaf 在此資料集能保留細分技術路線，eom 會過度合併
        prediction_data=True             # 允許後續對新資料預測主題
    )

    print("HDBSCAN 設定：min_cluster_size=15 | min_samples=5 | cluster_selection_method=leaf\n")
    return hdbscan_model


# =============================================================================
# STAGE 5：技術特徵提取 (Keyword Extraction - c-TF-IDF)
# =============================================================================

def stage5_vectorizer():
    """
    CountVectorizer + ClassTfidfTransformer：
    - trigram 支援 electric arc furnace、direct reduced iron 等鋼鐵技術詞
    - 自定義停用詞排除法律套語
    """

    print("=" * 60)
    print("STAGE 5：c-TF-IDF 技術特徵提取設定")
    print("=" * 60)

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 3),
        stop_words=CUSTOM_STOPWORDS,
        min_df=1,
        max_df=1.0
    )

    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=True
    )

    print(f"Vectorizer 設定：unigram/bigram/trigram | min_df=1 | max_df=1.0")
    print(f"停用詞數量：{len(CUSTOM_STOPWORDS)}\n")

    return vectorizer_model, ctfidf_model


# =============================================================================
# STAGE 6：主題建模訓練 + 優化與標籤生成
# =============================================================================

def stage6_train_and_refine(
    abstracts, df, embedding_model, embeddings,
    umap_model, hdbscan_model, vectorizer_model, ctfidf_model,
    target_topics: int = 20
):
    """
    BERTopic 完整訓練流程：
    1. 初始建模，保留 HDBSCAN 找到的原始主題
    2. 若主題數過多，再收斂至 15–20 個
    3. 主題關鍵詞輸出
    4. Topic ID 與標籤回填至原始 DataFrame
    """

    print("=" * 60)
    print("STAGE 6：BERTopic 主題建模訓練與優化")
    print("=" * 60)

    # ── 6.1 建立模型 ──────────────────────────────────────────────────────
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        nr_topics=None,
        top_n_words=10,
        calculate_probabilities=True,
        verbose=True
    )

    # ── 6.2 訓練（傳入預計算的 embeddings 加速）──────────────────────────
    print("\n開始訓練 BERTopic 模型...")
    topics, probs = topic_model.fit_transform(abstracts, embeddings)

    initial_count = len(set(topics)) - (1 if -1 in topics else 0)
    noise_count   = list(topics).count(-1)
    print(f"\n初始建模結果：識別主題數 = {initial_count} | 雜訊專利 = {noise_count:,}")

    # ── 6.3 主題收斂 ──────────────────────────────────────────────────────
    if initial_count > target_topics:
        print(f"主題數 {initial_count} > 目標 {target_topics}，執行 reduce_topics...")
        topic_model.reduce_topics(abstracts, nr_topics=target_topics)
        topics = topic_model.topics_
        final_count = len(set(topics)) - (1 if -1 in topics else 0)
        print(f"收斂後主題數：{final_count}")

    final_topics = topic_model.topics_
    final_count = len(set(final_topics)) - (1 if -1 in final_topics else 0)
    final_noise_count = final_topics.count(-1)
    final_noise_ratio = final_noise_count / len(final_topics) if final_topics else 0
    print(
        f"最終建模結果：主題數 = {final_count} | "
        f"雜訊專利 = {final_noise_count:,} ({final_noise_ratio:.1%})"
    )

    # ── 6.4 輸出主題關鍵詞 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("主題關鍵詞摘要（Top-10 技術詞彙）")
    print("=" * 60)

    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info["Topic"] != -1]

    keyword_rows = []
    for _, row in topic_info.iterrows():
        tid      = row["Topic"]
        words    = topic_model.get_topic(tid)
        keywords = " | ".join([w for w, _ in words[:10]])
        keyword_rows.append({
            "Topic_ID"      : tid,
            "Doc_Count"     : row["Count"],
            "Top10_Keywords": keywords
        })
        print(f"Topic {tid:>3} ({row['Count']:>5} 筆)：{keywords}")

    keywords_df = pd.DataFrame(keyword_rows)

    # ── 6.5 回填 Topic ID 與標籤至 DataFrame ─────────────────────────────
    df = df.copy()
    df["topic_id"]    = topic_model.topics_
    df["topic_label"] = df["topic_id"].apply(
        lambda t: f"Topic_{t}" if t != -1 else "NOISE"
    )

    # ── 6.6 儲存結果 ──────────────────────────────────────────────────────
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_dir / "patent_with_topics.parquet", index=False)
    keywords_df.to_csv(
        output_dir / "topic_keywords.csv", index=False, encoding="utf-8-sig"
    )

    print(f"\n結果已儲存至 {output_dir}")
    print("  ├── patent_with_topics.parquet（含 topic_id、topic_label）")
    print("  └── topic_keywords.csv（主題關鍵詞摘要表）")
    print(f"\nSTAGE 6 完成\n")

    return topic_model, df, keywords_df


# =============================================================================
# STAGE 7：Topics over Time 動態主題分析
# =============================================================================

def stage7_topics_over_time(topic_model, df, abstracts):
    """
    以 priority_date 切分的 4 個時間區段作為時間標籤，
    執行 Topics over Time 分析，輸出技術演進趨勢與生命週期狀態標記。
    """

    print("=" * 60)
    print("STAGE 7：Topics over Time 動態主題分析")
    print("=" * 60)

    SEGMENTS = [
        "SEG_A_2006_2010",
        "SEG_B_2011_2015",
        "SEG_C_2016_2020",
        "SEG_D_2021_2025"
    ]
    segment_to_code = {segment: idx for idx, segment in enumerate(SEGMENTS)}
    code_to_segment = {idx: segment for segment, idx in segment_to_code.items()}

    # BERTopic 0.17.4 對字串 timestamps 會呼叫 pandas 3 已移除的參數。
    # 先用數字代碼避開自動日期解析，再映射回原本的時間區段標籤。
    timestamps = df["time_segment"].map(segment_to_code).tolist()

    tot = topic_model.topics_over_time(
        abstracts,
        timestamps,
        nr_bins=None,
        evolution_tuning=True,   # 相鄰段語義連貫性
        global_tuning=True       # 全域一致性，確保跨段可比
    )

    tot["Timestamp"] = tot["Timestamp"].map(code_to_segment)

    output_dir = OUTPUT_DIR
    tot.to_csv(output_dir / "topics_over_time.csv", index=False, encoding="utf-8-sig")
    print("TOT 結果已儲存：topics_over_time.csv")

    # ── 技術生命週期狀態標記 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("技術生命週期狀態標記")
    print("=" * 60)

    lifecycle_rows = []

    for tid in sorted(tot["Topic"].unique()):
        if tid == -1:
            continue

        subset   = tot[tot["Topic"] == tid]
        freq_map = dict(zip(subset["Timestamp"], subset["Frequency"]))
        freq     = [freq_map.get(s, 0) for s in SEGMENTS]

        early  = sum(freq[:2])
        late   = sum(freq[2:])
        growth = (late - early) / (early + 1)

        if growth > 0.5 and freq[0] < freq[-1]:
            status = "🌱 新興"
        elif growth < -0.3 and freq[0] > freq[-1]:
            status = "📉 衰退"
        else:
            status = "✅ 成熟"

        lifecycle_rows.append({
            "Topic_ID"   : tid,
            "Freq_SEG_A" : freq[0],
            "Freq_SEG_B" : freq[1],
            "Freq_SEG_C" : freq[2],
            "Freq_SEG_D" : freq[3],
            "Growth_Rate": round(growth, 3),
            "Status"     : status
        })

        print(
            f"Topic {tid:>3} | "
            f"A:{freq[0]:>4}  B:{freq[1]:>4}  C:{freq[2]:>4}  D:{freq[3]:>4} | "
            f"成長率:{growth:>+6.2f} | {status}"
        )

    lifecycle_df = pd.DataFrame(lifecycle_rows)
    lifecycle_df.to_csv(
        output_dir / "topic_lifecycle.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n生命週期標記已儲存：topic_lifecycle.csv")
    print(f"\nSTAGE 7 完成\n")

    return tot, lifecycle_df


# =============================================================================
# 主程式入口
# =============================================================================

if __name__ == "__main__":

    INPUT_PATH    = str(PROJECT_DIR / "data" / "part-000000000000_domain_target_intersection.json")
    TARGET_TOPICS = 20

    # STAGE 1：載入與前處理
    df, abstracts = stage1_load_and_preprocess(INPUT_PATH)

    # STAGE 2：SciBERT 嵌入
    embedding_model, embeddings = stage2_embed(abstracts, batch_size=64)

    # STAGE 3：UMAP 降維設定
    umap_model = stage3_umap()

    # STAGE 4：HDBSCAN 聚類設定
    hdbscan_model = stage4_hdbscan()

    # STAGE 5：c-TF-IDF 特徵提取設定
    vectorizer_model, ctfidf_model = stage5_vectorizer()

    # STAGE 6：訓練、優化、標籤生成
    topic_model, df, keywords_df = stage6_train_and_refine(
        abstracts, df,
        embedding_model, embeddings,
        umap_model, hdbscan_model,
        vectorizer_model, ctfidf_model,
        target_topics=TARGET_TOPICS
    )

    # STAGE 7：Topics over Time 動態分析
    tot_df, lifecycle_df = stage7_topics_over_time(topic_model, df, abstracts)

    print("=" * 60)
    print("全流程執行完畢")
    print(f"最終 DataFrame 欄位：{list(df.columns)}")
    print("輸出目錄：/home/carbon/carbon/output/")
    print("  ├── patent_with_topics.parquet")
    print("  ├── topic_keywords.csv")
    print("  ├── topics_over_time.csv")
    print("  └── topic_lifecycle.csv")
    print("=" * 60)
