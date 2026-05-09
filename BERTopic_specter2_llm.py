# =============================================================================
# 碳捕捉專利 BERTopic 動態主題建模 Pipeline（針對碳中和領域）
# Stage 1–8：資料載入 → SPECTER2 → UMAP → HDBSCAN → c-TF-IDF → 主題優化
#            → TOT → 階層式技術群歸併 + LLM 兩階段（分群 + 命名）+ 群層級演進分析
# =============================================================================
#
# 本版主要更新（vs. 前一版）:
#   1. STAGE 8 新增「群心關鍵詞畫像」（centroid keyword profiles）：
#      在 Ward 階層分群後，對每個初始群聚合成員 topic 的 c-TF-IDF (word, score)，
#      提供有語義的群層線索給 LLM，而不只是中性編號。
#   2. STAGE 8 新增「代表性 title 取樣」：
#      利用 BERTopic 的 representative documents 機制反查 title_en，
#      為每個 topic 提供 3 個高密度語義訊號。
#   3. STAGE 8 將 LLM 呼叫拆成兩階段：
#        Step 4a: 分群決策（中性 ClusterA、ClusterB...）
#        Step 4b: cluster 命名（基於固定分群結果）
#      避免「為了好命名而調整分群」的相互汙染，並讓兩階段可獨立檢查與重跑。
#   4. 命名階段不再列出鋼鐵業 8 條技術路徑的「Possible names」範例，
#      改為要求命名直接源於 cluster 內容；衝突解決規則只保留在分群階段。
# =============================================================================

# ── 安裝套件（首次執行時取消註解）──────────────────────────────────────────
# pip install bertopic sentence-transformers umap-learn hdbscan pandas pyarrow adapters openai

import re
import json
import warnings
from html import unescape
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import os
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None



from transformers import AutoTokenizer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from scipy.cluster.hierarchy import linkage, fcluster

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output_specter2"


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
    "substantially", "summary", "system", "thereby", "therefrom", "therein",
    "thereof", "third", "typically", "unit", "use", "used", "using",
    "usually", "wherein",
}

CORPUS_STOPWORDS = {
    "based", "can", "containing", "contains", "each", "first", "formed",
    "forming", "having", "one", "portion", "two",
    # 專利裝置描述中常見的位置/形狀詞，容易讓 Topic 0 變成結構詞集合。
    "end", "ends", "opening", "openings", "wall", "walls", "inlet", "inlets",
    "outlet", "outlets", "upper", "lower", "inner", "outer", "side", "sides",
    "surface", "surfaces", "member", "members",
    # 本資料集中常見但主題辨識力偏弱的泛用描述詞。
    "high", "low", "equal", "material", "materials", "space", "spaces",
    "content", "percent", "chamber", "chambers", "line", "lines", "main",
    "level", "region", "regions", "image", "images", "imaging", "lens",
    "lenses", "video", "videos", "extruder", "extruders", "lt", "gt",
    "sub", "sup", "amp", "nbsp", "quot", "dwg", "cl", "cm", "kg", "hm",
    "partially", "uninterruptedly", "directly", "indirectly", "deals",
    "substance", "substances", "matter", "average", "layer", "visual",
    "information", "body", "export", "batch", "ls", "al", "vtd", "pgm", "pcd",
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
        1. HTML entity 解碼並移除 tag
        2. 轉小寫
        3. 移除數字與特殊符號（保留字母與空白）
        4. 移除多餘空白與 HTML 殘留 token
        5. 移除專利法律套語停用詞
        """
        text = unescape(text)
        text = re.sub(r"<[^>]+>", " ", text)
        # 轉小寫
        text = text.lower()
        # 常見 html/xml 殘留字串先清掉，避免產生 lt / gt / sub 之類雜訊 token
        text = re.sub(r"&[a-z]+;", " ", text)
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
    if "title_en" in df.columns:
        embedding_texts = (
            df["title_en"].fillna("").astype(str).str.strip()
            + " [SEP] "
            + df["abstract_en"].fillna("").astype(str).str.strip()
        ).tolist()
    else:
        embedding_texts = df["abstract_en"].fillna("").astype(str).tolist()

    print("\n各時間區段樣本數：")
    print(df["time_segment"].value_counts().sort_index().to_string())
    print(f"\nSTAGE 1 完成，最終語料筆數：{len(df):,}\n")

    return df, abstracts, embedding_texts


# =============================================================================
# STAGE 2：科學語義嵌入 (Scientific Embedding - SPECTER2)
# =============================================================================

class Specter2Embedder:
    """
    使用 allenai/specter2_base 搭配 allenai/specter2 proximity adapter。
    SPECTER2 官方建議用 title + abstract 產生科學文獻語義向量。
    """

    def __init__(self, token: str | None = None):
        try:
            from adapters import AutoAdapterModel
        except ImportError as exc:
            raise ImportError(
                "使用 SPECTER2 需要先安裝 adapters："
                "/home/carbon/carbon/.venv/bin/python -m pip install -U adapters"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            "allenai/specter2_base",
            token=token,
        )
        self.model = AutoAdapterModel.from_pretrained(
            "allenai/specter2_base",
            token=token,
        )
        self.model.load_adapter(
            "allenai/specter2",
            source="hf",
            load_as="specter2",
            set_active=True,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        total = len(texts)

        for start in range(0, total, batch_size):
            batch = texts[start:start + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=512,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                output = self.model(**inputs)
                batch_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()

            embeddings.append(batch_embeddings)
            print(f"  embedded {min(start + batch_size, total):,}/{total:,}")

        return np.vstack(embeddings)


def stage2_embed(embedding_texts: list, batch_size: int = 32):
    """
    使用 SPECTER2 將 title + abstract 轉化為 768 維語義向量。
    """

    print("=" * 60)
    print("STAGE 2：SPECTER2 語義嵌入")
    print("=" * 60)

    base_model_name = "allenai/specter2_base"
    adapter_name = "allenai/specter2"
    print(f"載入 base model：{base_model_name}")
    print(f"載入 adapter：{adapter_name}")
    print("（首次執行需下載模型與 adapter，請耐心等候）\n")

    hf_token = configure_huggingface_token()
    if hf_token:
        print("已從 .env / 環境變數載入 Hugging Face token")

    embedding_model = Specter2Embedder(token=hf_token)

    print(f"開始嵌入 {len(embedding_texts):,} 筆 title + abstract（batch_size={batch_size}）...")
    embeddings = embedding_model.encode(embedding_texts, batch_size=batch_size)

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
        n_neighbors=20,     # 拉高鄰居數，讓相近技術文本更容易形成穩定群
        n_components=5,     # 壓縮至 5 維
        min_dist=0.0,       # 允許緊密聚類
        metric="cosine",    # 以餘弦相似度保留語義鄰近關係
        random_state=42
    )

    print("UMAP 設定：n_neighbors=20 | n_components=5 | metric=cosine\n")
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
        min_cluster_size=10,             # 降低成群門檻，讓中小型技術群不易被打成雜訊
        min_samples=3,                   # 放寬核心點條件，降低 -1 比例
        metric="euclidean",
        cluster_selection_method="eom",  # eom 較能降低 leaf 帶來的高噪音比例
        prediction_data=True             # 允許後續對新資料預測主題
    )

    print("HDBSCAN 設定：min_cluster_size=10 | min_samples=3 | cluster_selection_method=eom\n")
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
        ngram_range=(1, 4),
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
    target_topics: int = 20,
    apply_outlier_reduction: bool = False,
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
        embedding_model=None,  # 使用 stage2 預先計算的 SPECTER2 embeddings
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

    # ── 6.3b 視需要將部分雜訊文件重新分配回最相近主題 ───────────────────
    noise_before_reduction = list(topic_model.topics_).count(-1)
    if apply_outlier_reduction and noise_before_reduction:
        print(
            f"偵測到 {noise_before_reduction:,} 筆雜訊，"
            "嘗試以 embeddings 策略執行 reduce_outliers..."
        )
        reassigned_topics = topic_model.reduce_outliers(
            abstracts,
            topic_model.topics_,
            embeddings=embeddings,
            strategy="embeddings",
        )
        topic_model.topics_ = reassigned_topics
        df = df.copy()
        df["topic_id"] = reassigned_topics
        print(
            "reduce_outliers 完成，剩餘雜訊："
            f"{list(reassigned_topics).count(-1):,}"
        )
    elif noise_before_reduction:
        print(
            f"保留 {noise_before_reduction:,} 筆雜訊，不執行 reduce_outliers，"
            "以避免大主題過度膨脹。"
        )

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
# STAGE 8 helpers：群心畫像 + 代表性 title + 兩階段 LLM
# =============================================================================

def _load_openai_key() -> str:
    """從 .env 載入 OPENAI_API_KEY。"""
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
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY 未設定，請確認 .env 檔案")
    return key


def _compute_initial_group_profiles(
    topic_model,
    initial_mapping_df,
    top_n_words: int = 15,
):
    """
    為每個 Ward 初始群計算「群心關鍵詞畫像」。

    做法（做法 2 加權聚合）：
      對群內每個 topic 取 topic_model.get_topic(tid) 的 (word, score)，
      將同群所有 topic 的 word score 加總後排序，取 Top-N。

    回傳:
      dict[int, list[tuple[str, float]]]
      key = Group_ID, value = [(word, weighted_score), ...]
    """
    group_profiles = {}

    for group_id in sorted(initial_mapping_df["Group_ID"].unique()):
        member_topics = initial_mapping_df[
            initial_mapping_df["Group_ID"] == group_id
        ]["Topic_ID"].tolist()

        word_scores: dict[str, float] = {}
        for tid in member_topics:
            for word, score in topic_model.get_topic(tid):
                word_scores[word] = word_scores.get(word, 0.0) + float(score)

        sorted_words = sorted(
            word_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n_words]

        group_profiles[int(group_id)] = sorted_words

    return group_profiles


def _get_representative_titles(
    topic_model,
    df,
    n_per_topic: int = 3,
):
    """
    為每個 topic 挑出最具代表性的 N 個 title_en。

    流程：
      1. 取 BERTopic 內建 representative_docs (回傳的是 abstract_clean 文字)
      2. 反查 df 中對應的 application_number / title_en
      3. 過濾掉空字串、太短、或法律套語式 title

    若 df 沒有 title_en 欄位，回傳空 dict。

    回傳:
      dict[int, list[str]]
      key = Topic_ID, value = [title_1, title_2, ...]
    """
    titles_map: dict[int, list[str]] = {}

    if "title_en" not in df.columns:
        print("  [警告] df 沒有 title_en 欄位，跳過代表性 title 取樣")
        return titles_map

    try:
        rep_docs = topic_model.get_representative_docs()
    except Exception as exc:
        print(f"  [警告] get_representative_docs 失敗：{exc}")
        return titles_map

    # 建立 abstract_clean → 第一筆 row index 的反查表（用 Series 加速）
    abstract_to_idx = (
        df.reset_index(drop=True)
          .reset_index()
          .drop_duplicates(subset="abstract_clean", keep="first")
          .set_index("abstract_clean")["index"]
    )

    def _is_meaningful(t: str) -> bool:
        """濾掉專利法律套語 title。"""
        if not t or len(t) < 10:
            return False
        t_low = t.lower().strip()
        bad_starts = (
            "apparatus and method",
            "method and apparatus",
            "system and method",
            "method and system",
            "device and method",
        )
        if t_low.startswith(bad_starts):
            # 太通用，但若加上具體技術詞仍可能有資訊量；
            # 這裡只在過短時才剔除。
            if len(t.split()) < 8:
                return False
        return True

    for tid, docs in rep_docs.items():
        if tid == -1:
            continue

        title_list: list[str] = []
        for doc_text in docs:
            row_idx = abstract_to_idx.get(doc_text)
            if row_idx is None:
                continue
            title = df.iloc[row_idx]["title_en"]
            if pd.isna(title):
                continue
            title = str(title).strip()
            if _is_meaningful(title):
                title_list.append(title)
            if len(title_list) >= n_per_topic:
                break

        titles_map[int(tid)] = title_list

    return titles_map


def _llm_assign_clusters(
    initial_mapping_df,
    group_profiles,
    model: str = "gpt-4o-mini",
):
    """
    LLM 第一階段：分群決策（不命名）。

    輸入給 LLM 的訊號（每個 topic）：
      - Topic_ID
      - Doc_Count
      - Top-10 keywords (c-TF-IDF)
      - 初始 Ward 群編號 + 群心關鍵詞畫像（語義線索）

    注意：分群階段刻意「不」給代表性 title。
    title 屬於專利層的具體訊號，會把 LLM 推向過度細分（每個 topic 自成一群）。
    title 只在命名階段（_llm_name_clusters）使用，因為命名需要的正是
    「這個 cluster 實際在做什麼」的人類語言描述。

    輸出：每個 Topic_ID 的 Cluster_ID（中性編號 ClusterA、ClusterB、...）
    這個階段保留鋼鐵業技術路徑的衝突解決規則，
    但刻意不給「Possible names」範例，避免命名偏誤反推分群。

    回傳 DataFrame：Topic_ID, Cluster_ID, Reason
    """
    from openai import OpenAI
    api_key = _load_openai_key()
    client = OpenAI(api_key=api_key)

    # ── 構造每個 topic 的訊號行 ───────────────────────────────────────
    topic_lines = []
    for _, row in initial_mapping_df.sort_values("Topic_ID").iterrows():
        tid = int(row["Topic_ID"])
        topic_lines.append(
            f"Topic {tid} | "
            f"Initial_Group_ID: {int(row['Group_ID'])} | "
            f"Doc_Count: {int(row['Doc_Count'])} | "
            f"Top10_Keywords: {row['Top10_Keywords']}"
        )
    topics_text = "\n".join(topic_lines)

    # ── 構造初始群心畫像區塊 ─────────────────────────────────────────
    profile_lines = []
    for group_id in sorted(group_profiles.keys()):
        member_topics = initial_mapping_df[
            initial_mapping_df["Group_ID"] == group_id
        ]["Topic_ID"].tolist()
        total_docs = int(
            initial_mapping_df[
                initial_mapping_df["Group_ID"] == group_id
            ]["Doc_Count"].sum()
        )
        words = group_profiles[group_id]
        word_str = ", ".join([w for w, _ in words])

        profile_lines.append(
            f"InitialGroup_{group_id} | "
            f"Members: Topic {member_topics} | "
            f"Total_Docs: {total_docs} | "
            f"Centroid_Keywords: {word_str}"
        )
    profiles_text = "\n".join(profile_lines)

    prompt = (
        "You are an expert in steel industry decarbonization technologies and "
        "patent topic analysis.\n\n"

        "STAGE OBJECTIVE:\n"
        "Your task at this stage is ONLY to assign each fine-grained topic to a "
        "higher-level cluster. Do NOT name the clusters. Use neutral identifiers "
        "(ClusterA, ClusterB, ClusterC, ...). Naming will be performed in a "
        "separate stage based on your assignment.\n\n"

        "INPUT YOU RECEIVE FOR EACH TOPIC:\n"
        "1. Top-10 keywords extracted via c-TF-IDF.\n"
        "2. An initial group ID from Ward hierarchical clustering on normalized "
        "topic embeddings. This reflects mathematical proximity in the SPECTER2 "
        "embedding space.\n"
        "3. The centroid keyword profile of that initial group, aggregated from "
        "the c-TF-IDF scores of all member topics. This is a moderate-strength "
        "semantic hint about what unifies the initial group.\n\n"

        "ASSIGNMENT PRINCIPLES:\n"
        "1. HARD CONSTRAINT: You MUST produce between 6 and 10 clusters total. "
        "Aim for 7 to 8 clusters of comparable granularity. If you find yourself "
        "wanting more than 10 clusters, you are operating at too fine a "
        "granularity — merge related topics into broader technology pathways. "
        "Single-topic clusters are allowed ONLY for genuinely isolated topics "
        "(e.g., molten carbonate fuel cells).\n"
        "2. The initial Ward groups are a reference, not a constraint. You may "
        "follow them when they align with technical pathways, or override them "
        "when topic-level keywords indicate a better grouping.\n"
        "3. Group at the technology pathway level, not the equipment or "
        "device level. Topics that share a broader pathway (e.g., direct "
        "reduction with hydrogen, regardless of whether they emphasize "
        "shaft furnace, fluidized bed, or pellet preparation) belong to the "
        "same cluster.\n"
        "4. Topics that share surface keywords but describe different "
        "technical processes must be separated.\n"
        "5. Document count is informational only — do not bias assignment "
        "toward balanced cluster sizes. If a real technology pathway has "
        "many topics, let the cluster be large.\n\n"

        "STEEL INDUSTRY CONFLICT-RESOLUTION RULES (use these to disambiguate):\n"
        "1. If a topic contains both slag and electric arc furnace terms:\n"
        "   - slag foaming for EAF operation belongs with EAF-related topics.\n"
        "   - steel slag, blast furnace slag, red mud, sludge, aqueous extraction, "
        "calcium recovery, and by-product treatment belong with slag/by-product "
        "topics.\n"
        "2. If a topic contains gas-related terms, decide based on gas context:\n"
        "   - blast furnace gas, hot blast, tuyere, stove → BF gas / energy.\n"
        "   - reducing gas, hydrogen gas, shaft furnace, DRI, metal oxide "
        "reduction → direct reduction / hydrogen metallurgy.\n"
        "   - exhaust gas, waste gas, CO2, CO, flue gas, converter gas → "
        "CCUS / process gas / off-gas utilization.\n"
        "3. If a topic contains hydrogen gas, metal oxide, water vapour, reduced "
        "metal, or reduction metal, group it with direct reduction / hydrogen "
        "metallurgy unless CO2 capture, exhaust gas, waste gas, or CO terms are "
        "clearly dominant.\n"
        "4. If a topic is dominated by flue gas, dust removal, zinc, ash, "
        "converter gas, converter flue, or waste heat, do NOT place it with EAF "
        "topics unless electric arc furnace, electrode, arc furnace, or scrap "
        "terms are dominant.\n"
        "5. Pig iron / molten iron production topics (pig iron, liquid pig iron, "
        "DRI product melting, smelting reduction) form their own pathway "
        "distinct from BF gas management.\n"
        "6. Metallurgical treatment / composition control topics (ladle "
        "treatment, denitrification, deoxidation, refining chemistry) form "
        "their own pathway distinct from primary iron-making.\n\n"

        "OUTPUT FORMAT:\n"
        "Return valid JSON only. No markdown, no comments, no text outside JSON. "
        "Return a JSON array. Each item must contain exactly:\n"
        "  - Topic_ID: integer\n"
        "  - Cluster_ID: string (must be one of ClusterA, ClusterB, ClusterC, "
        "..., using consecutive letters starting from A)\n"
        "  - Reason: string (brief technical rationale, in your own words, "
        "explaining what unifies this topic with others in the same cluster)\n\n"
        "Every Topic_ID must appear exactly once. The number of distinct "
        "Cluster_IDs must be between 6 and 10 inclusive.\n\n"

        "INITIAL GROUP CENTROID PROFILES (Ward clustering on SPECTER2 "
        "embeddings):\n"
        f"{profiles_text}\n\n"

        "TOPICS TO ASSIGN:\n\n"
        f"{topics_text}"
    )

    print("\n[LLM Stage 4a] Calling LLM for cluster assignment...")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=4000,
    )

    raw = response.choices[0].message.content.strip()
    print(f"LLM cluster-assignment response (preview):\n{raw[:600]}\n...\n")

    raw_clean = re.sub(r"^```json\s*", "", raw)
    raw_clean = re.sub(r"^```\s*", "", raw_clean)
    raw_clean = re.sub(r"\s*```$", "", raw_clean)

    try:
        data = json.loads(raw_clean)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse LLM cluster-assignment JSON: {e}\nRaw:\n{raw}"
        )

    assign_df = pd.DataFrame(data)

    required_cols = {"Topic_ID", "Cluster_ID", "Reason"}
    missing = required_cols - set(assign_df.columns)
    if missing:
        raise ValueError(f"LLM assignment output missing columns: {missing}")

    assign_df["Topic_ID"] = assign_df["Topic_ID"].astype(int)
    assign_df["Cluster_ID"] = assign_df["Cluster_ID"].astype(str).str.strip()

    expected_topics = set(initial_mapping_df["Topic_ID"].astype(int))
    returned_topics = set(assign_df["Topic_ID"].astype(int))

    missing_topics = expected_topics - returned_topics
    extra_topics = returned_topics - expected_topics
    if missing_topics:
        raise ValueError(f"LLM missed Topic_IDs: {sorted(missing_topics)}")
    if extra_topics:
        raise ValueError(f"LLM returned unknown Topic_IDs: {sorted(extra_topics)}")
    if assign_df["Topic_ID"].duplicated().any():
        dup = assign_df.loc[assign_df["Topic_ID"].duplicated(), "Topic_ID"].tolist()
        raise ValueError(f"LLM duplicated Topic_IDs: {dup}")

    n_clusters = assign_df["Cluster_ID"].nunique()
    if not (6 <= n_clusters <= 10):
        raise ValueError(
            f"LLM produced {n_clusters} clusters, expected 6 to 10. "
            "Re-run or revise the prompt."
        )

    return assign_df


def _llm_name_clusters(
    cluster_assign_df,
    initial_mapping_df,
    representative_titles,
    model: str = "gpt-4o-mini",
):
    """
    LLM 第二階段：cluster 命名。

    輸入給 LLM 的訊號（每個 cluster）：
      - Cluster_ID（中性編號）
      - 該 cluster 內所有成員 topic 的 Top-10 keywords + 代表性 title
      - 該 cluster 的總文件數

    Prompt 設計刻意避開：
      - 不列出鋼鐵業 8 條技術路徑的「Possible names」範例
      - 不提供任何預設的命名候選

    這是為了讓命名直接源於 cluster 實際內容，而非套用先驗標籤。
    若 cluster 內容混雜，要求 LLM 用 "MIXED:" 前綴標記，作為
    分群階段是否需要重做的診斷訊號。

    回傳 DataFrame：Cluster_ID, Cluster_Name, Reason
    """
    from openai import OpenAI
    api_key = _load_openai_key()
    client = OpenAI(api_key=api_key)

    # ── 為每個 cluster 構造完整成員資訊區塊 ───────────────────────────
    # merge keywords / doc_count
    merged = cluster_assign_df.merge(
        initial_mapping_df[["Topic_ID", "Top10_Keywords", "Doc_Count"]],
        on="Topic_ID",
        how="left",
    )

    cluster_blocks = []
    for cid in sorted(merged["Cluster_ID"].unique()):
        sub = merged[merged["Cluster_ID"] == cid].sort_values("Topic_ID")
        total_docs = int(sub["Doc_Count"].sum())
        n_topics = len(sub)

        member_lines = []
        for _, r in sub.iterrows():
            tid = int(r["Topic_ID"])
            titles = representative_titles.get(tid, [])
            if titles:
                title_str = " || ".join([f'"{t}"' for t in titles[:2]])
            else:
                title_str = "(no titles)"
            member_lines.append(
                f"  - Topic {tid} ({int(r['Doc_Count'])} docs) | "
                f"Keywords: {r['Top10_Keywords']} | "
                f"Titles: {title_str}"
            )

        cluster_blocks.append(
            f"=== {cid} ===\n"
            f"Total_Docs: {total_docs} | N_Topics: {n_topics}\n"
            f"Members:\n" + "\n".join(member_lines)
        )

    clusters_text = "\n\n".join(cluster_blocks)

    prompt = (
        "You are naming technology clusters from a corpus of patents related to "
        "industrial decarbonization (with steel industry coverage). The clustering "
        "decision is already FIXED — your task is ONLY to generate a concise, "
        "technically accurate English name for each cluster, derived directly "
        "from the cluster's actual content.\n\n"

        "FOR EACH CLUSTER, YOU SEE:\n"
        "- A neutral Cluster_ID (ClusterA, ClusterB, ...)\n"
        "- All member topics with their Top-10 c-TF-IDF keywords\n"
        "- 1–2 representative document titles per topic (written by patent authors)\n"
        "- Total document count and number of member topics\n\n"

        "NAMING REQUIREMENTS:\n"
        "1. The name MUST be derived from the cluster's own keywords and titles. "
        "Do NOT apply names from any external taxonomy or industry classification. "
        "Do NOT invent technologies that the keywords and titles do not support.\n"
        "2. Length: 3 to 8 words. Prefer concrete technical descriptors over "
        "abstract category labels.\n"
        "3. The name must reflect what UNIFIES all member topics, not the dominant "
        "topic alone. If the cluster spans multiple sub-themes, the name should "
        "capture the shared technology pathway.\n"
        "4. If multiple clusters share the same theme, give them slightly different "
        "names that reflect their internal differences. Never assign identical "
        "Cluster_Name strings to two different Cluster_IDs.\n"
        "5. AVOID OVERLY GENERIC NAMES: 'Steel Production', 'Carbon Neutral "
        "Technologies', 'Furnace Technologies', 'Gas Utilization', 'Green Steel "
        "Technologies', 'Metallurgical Processes', 'Industrial Decarbonization', "
        "or any name where 'Technology / System / Process / Method' is the only "
        "noun (e.g., 'Gas Recovery Technology' is too generic).\n"
        "6. AVOID OVERLY NARROW NAMES: do not use a single member topic's content "
        "as the cluster name if the cluster contains multiple topics covering a "
        "broader pathway.\n"
        "7. If you cannot find a coherent unifying theme, prefix the name with "
        "'MIXED:' and use the Reason field to describe the heterogeneity. This "
        "signals that the previous clustering may need revision.\n\n"

        "OUTPUT FORMAT:\n"
        "Return valid JSON only. No markdown, no comments. Return a JSON array. "
        "Each item must contain exactly:\n"
        "  - Cluster_ID: string (must match input)\n"
        "  - Cluster_Name: string (3 to 8 words)\n"
        "  - Reason: string (brief explanation of what unifies the cluster, "
        "in your own words, citing specific keywords or titles)\n\n"
        "Every input Cluster_ID must appear exactly once.\n\n"

        "CLUSTERS TO NAME:\n\n"
        f"{clusters_text}"
    )

    print("\n[LLM Stage 4b] Calling LLM for cluster naming...")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )

    raw = response.choices[0].message.content.strip()
    print(f"LLM cluster-naming response (preview):\n{raw[:600]}\n...\n")

    raw_clean = re.sub(r"^```json\s*", "", raw)
    raw_clean = re.sub(r"^```\s*", "", raw_clean)
    raw_clean = re.sub(r"\s*```$", "", raw_clean)

    try:
        data = json.loads(raw_clean)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse LLM naming JSON: {e}\nRaw:\n{raw}"
        )

    name_df = pd.DataFrame(data)

    required_cols = {"Cluster_ID", "Cluster_Name", "Reason"}
    missing = required_cols - set(name_df.columns)
    if missing:
        raise ValueError(f"LLM naming output missing columns: {missing}")

    name_df["Cluster_ID"] = name_df["Cluster_ID"].astype(str).str.strip()

    expected_clusters = set(cluster_assign_df["Cluster_ID"].unique())
    returned_clusters = set(name_df["Cluster_ID"].unique())

    missing_clusters = expected_clusters - returned_clusters
    extra_clusters = returned_clusters - expected_clusters
    if missing_clusters:
        raise ValueError(f"LLM missed Cluster_IDs: {sorted(missing_clusters)}")
    if extra_clusters:
        raise ValueError(f"LLM returned unknown Cluster_IDs: {sorted(extra_clusters)}")

    if name_df["Cluster_Name"].duplicated().any():
        dup = name_df.loc[name_df["Cluster_Name"].duplicated(), "Cluster_Name"].tolist()
        raise ValueError(
            f"LLM produced duplicated Cluster_Name across different Cluster_IDs: {dup}. "
            "Each cluster must have a unique name."
        )

    # 警示但不阻擋：MIXED: 前綴的 cluster 表示分群可能有問題
    mixed_clusters = name_df[name_df["Cluster_Name"].str.startswith("MIXED:")]
    if not mixed_clusters.empty:
        print(
            f"\n[警示] {len(mixed_clusters)} 個 cluster 被標記為 MIXED，"
            "建議檢查分群階段或考慮重跑：\n"
            + mixed_clusters[["Cluster_ID", "Cluster_Name"]].to_string(index=False)
        )

    return name_df


def build_final_group_mapping_v2(
    initial_mapping_df,
    cluster_assign_df,
    cluster_name_df,
    output_dir,
):
    """
    將兩階段 LLM 結果合併為最終 topic-to-group 映射，並指派 Final_Group_ID。

    回傳的 final_mapping_df 欄位:
      Topic_ID, Doc_Count, Top10_Keywords,
      Group_ID (Ward 初始群),
      Group_Name (InitialGroup_X 中性編號),
      Cluster_ID (LLM 分群中性編號),
      Final_Group_ID (依 Doc_Count 排序),
      Final_Group_Name (LLM 命名),
      Assignment_Reason (LLM 分群理由),
      Naming_Reason (LLM 命名理由)
    """
    # 合併 cluster assignment 與 naming
    cluster_full = cluster_assign_df.merge(
        cluster_name_df,
        on="Cluster_ID",
        how="left",
        suffixes=("_assign", "_naming"),
    )

    # 重新命名 Reason 欄位以避免歧義
    cluster_full = cluster_full.rename(columns={
        "Reason_assign": "Assignment_Reason",
        "Reason_naming": "Naming_Reason",
        "Cluster_Name": "Final_Group_Name",
    })

    final_mapping_df = initial_mapping_df.merge(
        cluster_full,
        on="Topic_ID",
        how="left",
    )

    if final_mapping_df["Final_Group_Name"].isna().any():
        missing_topics = final_mapping_df[
            final_mapping_df["Final_Group_Name"].isna()
        ]["Topic_ID"].tolist()
        raise ValueError(f"Missing Final_Group_Name for topics: {missing_topics}")

    # 依 Final_Group_Name 內 Doc_Count 總和排序，賦予 Final_Group_ID
    group_order = (
        final_mapping_df
        .groupby("Final_Group_Name")["Doc_Count"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    group_name_to_id = {name: idx for idx, name in enumerate(group_order)}

    final_mapping_df["Final_Group_ID"] = final_mapping_df["Final_Group_Name"].map(
        group_name_to_id
    )

    final_mapping_df = final_mapping_df[
        [
            "Topic_ID",
            "Doc_Count",
            "Top10_Keywords",
            "Group_ID",
            "Group_Name",
            "Cluster_ID",
            "Final_Group_ID",
            "Final_Group_Name",
            "Assignment_Reason",
            "Naming_Reason",
        ]
    ].sort_values(["Final_Group_ID", "Topic_ID"])

    final_mapping_df.to_csv(
        output_dir / "final_group_topic_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Final topic-to-group mapping saved:")
    print("  └── final_group_topic_mapping.csv")

    return final_mapping_df


def apply_final_groups_to_patents(
    df,
    final_mapping_df,
    output_dir,
):
    """
    將 LLM 校正後的 final 技術群套用到 patent-level dataframe。
    """

    topic_to_final_group_id = dict(zip(
        final_mapping_df["Topic_ID"],
        final_mapping_df["Final_Group_ID"]
    ))

    topic_to_final_group_name = dict(zip(
        final_mapping_df["Topic_ID"],
        final_mapping_df["Final_Group_Name"]
    ))

    df = df.copy()

    df["final_group_id"] = df["topic_id"].map(
        lambda t: topic_to_final_group_id.get(t, -1)
    )

    df["final_group_name"] = df["topic_id"].map(
        lambda t: topic_to_final_group_name.get(t, "NOISE") if t != -1 else "NOISE"
    )

    df.to_parquet(
        output_dir / "patent_with_final_topic_groups.parquet",
        index=False
    )

    df.to_csv(
        output_dir / "patent_with_final_topic_groups.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("Patent-level final group file saved:")
    print("  ├── patent_with_final_topic_groups.parquet")
    print("  └── patent_with_final_topic_groups.csv")

    return df


def stage8_final_group_evolution(
    df,
    output_dir,
):
    """
    Run technology-group-level evolution analysis using LLM-refined final groups.
    """
    print("=" * 60)
    print("STAGE 8E: Final technology-group-level evolution analysis")
    print("=" * 60)

    SEGMENTS = [
        "SEG_A_2006_2010",
        "SEG_B_2011_2015",
        "SEG_C_2016_2020",
        "SEG_D_2021_2025",
    ]

    group_df = df[df["final_group_id"] != -1].copy()
    group_df["priority_year"] = group_df["priority_date"].dt.year

    g_seg_counts = pd.crosstab(
        group_df["final_group_name"],
        group_df["time_segment"]
    )
    g_seg_counts = g_seg_counts.reindex(columns=SEGMENTS, fill_value=0)

    g_seg_counts.to_csv(
        output_dir / "final_group_segment_counts.csv",
        encoding="utf-8-sig"
    )

    seg_totals = group_df["time_segment"].value_counts().reindex(SEGMENTS, fill_value=0)
    g_seg_shares = g_seg_counts.div(
        seg_totals.replace(0, np.nan),
        axis=1
    ).fillna(0)

    g_seg_shares.to_csv(
        output_dir / "final_group_segment_shares.csv",
        encoding="utf-8-sig"
    )

    g_yearly_counts = pd.crosstab(
        group_df["priority_year"],
        group_df["final_group_name"]
    )
    g_yearly_counts = g_yearly_counts.reindex(range(2006, 2026), fill_value=0)

    g_yearly_counts.to_csv(
        output_dir / "final_group_yearly_counts.csv",
        encoding="utf-8-sig"
    )

    g_yearly_totals = g_yearly_counts.sum(axis=1).replace(0, np.nan)
    g_yearly_shares = g_yearly_counts.div(g_yearly_totals, axis=0).fillna(0)

    g_yearly_shares.to_csv(
        output_dir / "final_group_yearly_shares.csv",
        encoding="utf-8-sig"
    )

    segment_midpoints = np.array([2008, 2013, 2018, 2023], dtype=float)

    rows = []

    for group_name in g_seg_counts.index:
        freq = [int(g_seg_counts.loc[group_name, s]) for s in SEGMENTS]
        shares = [float(g_seg_shares.loc[group_name, s]) for s in SEGMENTS]

        slope = float(np.polyfit(segment_midpoints, shares, 1)[0])
        peak_idx = int(np.argmax(freq))

        early = sum(freq[:2])
        late = sum(freq[2:])
        growth = (late - early) / (early + 1)

        trajectory = classify_trajectory(freq, shares)

        rows.append({
            "Final_Group_Name": group_name,
            "Doc_Count": sum(freq),
            "Freq_2006_2010": freq[0],
            "Freq_2011_2015": freq[1],
            "Freq_2016_2020": freq[2],
            "Freq_2021_2025": freq[3],
            "Share_2006_2010": round(shares[0], 4),
            "Share_2011_2015": round(shares[1], 4),
            "Share_2016_2020": round(shares[2], 4),
            "Share_2021_2025": round(shares[3], 4),
            "Share_Change": round(shares[-1] - shares[0], 4),
            "Share_Slope_Per_Year": round(slope, 6),
            "Growth_Rate": round(growth, 3),
            "Peak_Segment": SEGMENTS[peak_idx],
            "Trajectory": trajectory,
        })

    final_group_evo_df = pd.DataFrame(rows).sort_values(
        "Doc_Count",
        ascending=False
    )

    final_group_evo_df.to_csv(
        output_dir / "final_group_evolution_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("Final group evolution files saved:")
    print("  ├── final_group_segment_counts.csv")
    print("  ├── final_group_segment_shares.csv")
    print("  ├── final_group_yearly_counts.csv")
    print("  ├── final_group_yearly_shares.csv")
    print("  └── final_group_evolution_summary.csv")

    return g_seg_counts, g_seg_shares, final_group_evo_df


def stage8_initial_grouping_and_llm_regrouping(
    topic_model,
    df,
    abstracts,
    keywords_df,
    n_groups: int = 12,
    interactive: bool = True,
):
    """
    STAGE 8（v2 新版本）：
      Step 1   : 提取 topic embeddings
      Step 2   : Ward 階層分群（n_groups 個初始群）
      Step 3   : 構造 initial_mapping_df（中性 InitialGroup_X 命名）
      Step 3.5 : 計算群心關鍵詞畫像（centroid keyword profiles）
      Step 3.6 : 為每個 topic 取 3 個代表性 title
      Step 4a  : LLM 第一階段——分群決策（中性 ClusterA、ClusterB...）
      Step 4b  : LLM 第二階段——cluster 命名（基於固定分群結果）
      Step 5   : 合併兩階段結果為 final mapping
      Step 6   : 套用到 patent-level dataframe
      Step 7   : 群層級演進分析
    """
    from sklearn.preprocessing import normalize

    print("=" * 60)
    print("STAGE 8 (v2): Ward + 兩階段 LLM (assignment → naming)")
    print("=" * 60)

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1：Extract topic embeddings ─────────────────────────────
    topic_ids = sorted([t for t in set(topic_model.topics_) if t != -1])
    n_topics = len(topic_ids)
    print(f"\nFine-grained topics: {n_topics}")

    all_embs = topic_model.topic_embeddings_
    outlier_offset = getattr(topic_model, "_outliers", 0)

    topic_embs = np.array([
        all_embs[tid + outlier_offset]
        for tid in topic_ids
    ])

    print(f"Raw topic embedding shape: {topic_embs.shape}")

    # ── Step 2：Initial Ward clustering ──────────────────────────────
    topic_embs_norm = normalize(topic_embs, norm="l2")
    Z = linkage(topic_embs_norm, method="ward", metric="euclidean")
    cluster_labels = fcluster(Z, t=n_groups, criterion="maxclust")

    unique_labels = sorted(set(cluster_labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    cluster_labels = np.array([label_map[x] for x in cluster_labels])

    n_groups = len(unique_labels)
    topic_to_group = dict(zip(topic_ids, cluster_labels))

    print(f"\nWard 初始分群產生 {n_groups} 個群：")
    for gid in range(n_groups):
        members = [t for t, g in topic_to_group.items() if g == gid]
        print(f"  InitialGroup_{gid}: Topic {members}")

    # 儲存 linkage matrix（供論文 dendrogram 使用）
    linkage_df = pd.DataFrame(
        Z,
        columns=["Cluster_1", "Cluster_2", "Distance", "Sample_Count"]
    )
    linkage_df.to_csv(
        output_dir / "topic_hierarchical_linkage.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ── Step 3：Prepare initial mapping ──────────────────────────────
    keyword_map = dict(zip(keywords_df["Topic_ID"], keywords_df["Top10_Keywords"]))
    count_map = dict(zip(keywords_df["Topic_ID"], keywords_df["Doc_Count"]))

    initial_group_names = {gid: f"InitialGroup_{gid}" for gid in range(n_groups)}

    initial_rows = []
    for tid in topic_ids:
        gid = topic_to_group[tid]
        initial_rows.append({
            "Topic_ID": tid,
            "Group_ID": gid,
            "Group_Name": initial_group_names[gid],
            "Doc_Count": int(count_map.get(tid, 0)),
            "Top10_Keywords": keyword_map.get(tid, ""),
        })

    initial_mapping_df = pd.DataFrame(initial_rows).sort_values("Topic_ID")
    initial_mapping_df.to_csv(
        output_dir / "initial_group_topic_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\nInitial group mapping saved → initial_group_topic_mapping.csv")

    # ── Step 3.5：Compute centroid keyword profiles ──────────────────
    print("\n計算各初始群的群心關鍵詞畫像 (centroid keyword profiles)...")
    group_profiles = _compute_initial_group_profiles(
        topic_model,
        initial_mapping_df,
        top_n_words=15,
    )
    for gid, words in group_profiles.items():
        preview = ", ".join([w for w, _ in words[:8]])
        print(f"  InitialGroup_{gid}: {preview}")

    profile_rows = []
    for gid, words in group_profiles.items():
        member_topics = initial_mapping_df[
            initial_mapping_df["Group_ID"] == gid
        ]["Topic_ID"].tolist()
        profile_rows.append({
            "Group_ID": gid,
            "Member_Topic_IDs": str(member_topics),
            "Centroid_Keywords": ", ".join([w for w, _ in words]),
        })
    pd.DataFrame(profile_rows).to_csv(
        output_dir / "initial_group_profiles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("Centroid profiles saved → initial_group_profiles.csv")

    # ── Step 3.6：Sample representative titles for each topic ───────
    print("\n取每個 topic 的 3 個代表性 title...")
    representative_titles = _get_representative_titles(
        topic_model,
        df,
        n_per_topic=3,
    )
    coverage = sum(1 for v in representative_titles.values() if v)
    print(f"  成功取得代表性 title 的 topic 數: {coverage}/{n_topics}")

    title_rows = []
    for tid in sorted(representative_titles.keys()):
        title_rows.append({
            "Topic_ID": tid,
            "Representative_Titles": " || ".join(representative_titles[tid]),
        })
    pd.DataFrame(title_rows).to_csv(
        output_dir / "topic_representative_titles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("Representative titles saved → topic_representative_titles.csv")

    # ── Step 4a：LLM 分群決策 ────────────────────────────────────────
    # 注意：分群階段刻意不傳 representative_titles，
    # title 是專利層具體訊號，會把 LLM 推向過度細分。
    # title 只在 Step 4b 命名階段使用。
    cluster_assign_df = _llm_assign_clusters(
        initial_mapping_df,
        group_profiles,
    )
    cluster_assign_df.to_csv(
        output_dir / "llm_cluster_assignment_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\n分群結果：產生 {cluster_assign_df['Cluster_ID'].nunique()} 個 clusters")
    for cid in sorted(cluster_assign_df["Cluster_ID"].unique()):
        members = cluster_assign_df[
            cluster_assign_df["Cluster_ID"] == cid
        ]["Topic_ID"].tolist()
        print(f"  {cid}: Topic {members}")
    print("LLM cluster assignment saved → llm_cluster_assignment_raw.csv")

    # ── Step 4b：LLM cluster 命名 ────────────────────────────────────
    cluster_name_df = _llm_name_clusters(
        cluster_assign_df,
        initial_mapping_df,
        representative_titles,
    )
    cluster_name_df.to_csv(
        output_dir / "llm_cluster_naming_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\n命名結果：")
    for _, r in cluster_name_df.iterrows():
        print(f"  {r['Cluster_ID']} → {r['Cluster_Name']}")
    print("LLM cluster naming saved → llm_cluster_naming_raw.csv")

    # ── Step 5：Build final mapping ─────────────────────────────────
    final_mapping_df = build_final_group_mapping_v2(
        initial_mapping_df,
        cluster_assign_df,
        cluster_name_df,
        output_dir,
    )

    print("\nFinal topic-to-group 結果：")
    for group_name, sub_df in final_mapping_df.groupby("Final_Group_Name"):
        topics = sub_df["Topic_ID"].tolist()
        docs = sub_df["Doc_Count"].sum()
        print(f"  {group_name} ({docs} docs): Topic {topics}")

    # ── Step 6：Apply final groups to patent-level dataframe ─────────
    df_final = apply_final_groups_to_patents(
        df,
        final_mapping_df,
        output_dir,
    )

    # ── Step 7：Final group-level evolution analysis ────────────────
    g_seg_counts, g_seg_shares, final_group_evo_df = stage8_final_group_evolution(
        df_final,
        output_dir,
    )

    print("\nSTAGE 8 (v2) completed\n")

    return df_final, initial_mapping_df, final_mapping_df, final_group_evo_df


# =============================================================================
# STAGE 7：Topics over Time 動態主題分析
# =============================================================================

def classify_trajectory(freq, shares):
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


def stage7_topics_over_time(topic_model, df, abstracts, keywords_df):
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
            status = " 新興"
        elif growth < -0.3 and freq[0] > freq[-1]:
            status = " 衰退"
        else:
            status = " 成熟"

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

    # ── 近 20 年演進軌跡輸出 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("近 20 年技術演進軌跡分析")
    print("=" * 60)

    topic_df = df[df["topic_id"] != -1].copy()
    topic_df["priority_year"] = topic_df["priority_date"].dt.year

    yearly_counts = pd.crosstab(topic_df["priority_year"], topic_df["topic_id"])
    yearly_counts = yearly_counts.reindex(range(2006, 2026), fill_value=0)
    yearly_counts.to_csv(output_dir / "topic_yearly_counts.csv", encoding="utf-8-sig")

    yearly_totals = yearly_counts.sum(axis=1).replace(0, np.nan)
    yearly_shares = yearly_counts.div(yearly_totals, axis=0).fillna(0)
    yearly_shares.to_csv(output_dir / "topic_yearly_shares.csv", encoding="utf-8-sig")

    segment_counts = pd.crosstab(topic_df["topic_id"], topic_df["time_segment"])
    segment_counts = segment_counts.reindex(columns=SEGMENTS, fill_value=0)
    segment_counts.to_csv(output_dir / "topic_segment_counts.csv", encoding="utf-8-sig")

    segment_totals = topic_df["time_segment"].value_counts().reindex(SEGMENTS, fill_value=0)
    segment_shares = segment_counts.div(segment_totals.replace(0, np.nan), axis=1).fillna(0)
    segment_shares.to_csv(output_dir / "topic_segment_shares.csv", encoding="utf-8-sig")

    keyword_map = dict(zip(keywords_df["Topic_ID"], keywords_df["Top10_Keywords"]))
    count_map = dict(zip(keywords_df["Topic_ID"], keywords_df["Doc_Count"]))

    evolution_rows = []
    segment_midpoints = np.array([2008, 2013, 2018, 2023], dtype=float)

    for topic_id in sorted(segment_counts.index):
        freq = [int(segment_counts.loc[topic_id, segment]) for segment in SEGMENTS]
        shares = [float(segment_shares.loc[topic_id, segment]) for segment in SEGMENTS]
        slope = float(np.polyfit(segment_midpoints, shares, 1)[0])
        peak_idx = int(np.argmax(freq))
        keywords = keyword_map.get(topic_id, "")

        evolution_rows.append({
            "Topic_ID": topic_id,
            "Topic_Label": f"Topic_{topic_id}",
            "Top10_Keywords": keywords,
            "Doc_Count": int(count_map.get(topic_id, sum(freq))),
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
            "Trajectory": classify_trajectory(freq, shares),
        })

    evolution_df = pd.DataFrame(evolution_rows)
    evolution_df.to_csv(
        output_dir / "topic_evolution_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("演進軌跡檔案已儲存：")
    print("  ├── topic_yearly_counts.csv")
    print("  ├── topic_yearly_shares.csv")
    print("  ├── topic_segment_counts.csv")
    print("  ├── topic_segment_shares.csv")
    print("  └── topic_evolution_summary.csv")
    print(f"\nSTAGE 7 完成\n")

    return tot, lifecycle_df, evolution_df


# =============================================================================
# 主程式入口
# =============================================================================

if __name__ == "__main__":
    # INPUT_PATH    = str(PROJECT_DIR / "data" / "part-000000000000_carbon_neutral_keywords.json")
    INPUT_PATH    = str(PROJECT_DIR / "dataa" /"global_allonlycpc"/ "global_allonlycpc_carbon_neutral_keywords.json")
    TARGET_TOPICS = 30


    # STAGE 1：載入與前處理
    df, abstracts, embedding_texts = stage1_load_and_preprocess(INPUT_PATH)

    # STAGE 2：SPECTER2 嵌入
    embedding_model, embeddings = stage2_embed(embedding_texts, batch_size=32)

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
    tot_df, lifecycle_df, evolution_df = stage7_topics_over_time(
        topic_model, df, abstracts, keywords_df
    )

    # STAGE 8: Ward + 兩階段 LLM (assignment → naming)
    df_final, initial_mapping_df, final_mapping_df, final_group_evo_df = (
        stage8_initial_grouping_and_llm_regrouping(
            topic_model,
            df,
            abstracts,
            keywords_df,
            n_groups=12,
            interactive=True,
        )
    )

    print("=" * 60)
    print("全流程執行完畢")
    print(f"最終 DataFrame 欄位：{list(df_final.columns)}")
    print(f"輸出目錄：{OUTPUT_DIR}/")
    print("  ├── topic_keywords.csv")
    print("  ├── patent_with_topics.parquet")
    print("  ├── topics_over_time.csv")
    print("  ├── topic_lifecycle.csv")
    print("  ├── topic_evolution_summary.csv")
    print("  ├── initial_group_topic_mapping.csv")
    print("  ├── initial_group_profiles.csv          [v2 新增]")
    print("  ├── topic_representative_titles.csv     [v2 新增]")
    print("  ├── llm_cluster_assignment_raw.csv      [v2 新增]")
    print("  ├── llm_cluster_naming_raw.csv          [v2 新增]")
    print("  ├── final_group_topic_mapping.csv")
    print("  ├── patent_with_final_topic_groups.parquet")
    print("  ├── patent_with_final_topic_groups.csv")
    print("  ├── final_group_segment_counts.csv")
    print("  ├── final_group_segment_shares.csv")
    print("  ├── final_group_yearly_counts.csv")
    print("  ├── final_group_yearly_shares.csv")
    print("  ├── final_group_evolution_summary.csv")
    print("  └── topic_hierarchical_linkage.csv")
    print("=" * 60)