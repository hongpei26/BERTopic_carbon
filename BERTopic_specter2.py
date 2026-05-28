# =============================================================================
# 碳捕捉專利 BERTopic 動態主題建模 Pipeline v3(軟分配版)
# Stage 1–7:資料載入 → SPECTER2 → UMAP → HDBSCAN → c-TF-IDF
#           → KeyBERT+MMR+LLM 表示法微調 → 軟分配 → TOT
# -----------------------------------------------------------------------------
# v3 新增(軟分配鏈路):
#  Stage 6:calculate_probabilities=True(已啟用,沿用)
#  Stage 6.5:approximate_distribution() 推算 token 級主題分佈矩陣
#            → 套用 PROBABILITY_THRESHOLD=0.20 門檻 → 產生 multi_topics 欄位
#  Stage 7:以 df.explode("multi_topics") 展開,讓跨界專利同時貢獻多個技術軸線
# =============================================================================

# ── 安裝套件(首次執行時取消註解)──────────────────────────────────────────
# pip install bertopic sentence-transformers umap-learn hdbscan pandas pyarrow
# pip install adapters openai>=1.0 tiktoken

import os
import re
import json
import warnings
from html import unescape
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from transformers import AutoTokenizer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
    OpenAI as OpenAIRepresentation,
)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output_specter2_3000"
EMBEDDING_CACHE = OUTPUT_DIR / "specter2_embeddings.npy"
EMBEDDING_INDEX = OUTPUT_DIR / "specter2_embeddings_index.parquet"
EMBEDDING_INPUT_MODE = "title_title_abstract"

# === v3 新增:軟分配參數 ======================================================
PROBABILITY_THRESHOLD = 0.20   # 門檻:文件對某主題機率 > 此值即視為具備該主題屬性
APPROX_DIST_WINDOW = 4         # approximate_distribution 的 token window 大小
APPROX_DIST_STRIDE = 1         # token window 滑動步長
APPROX_DIST_BATCH_SIZE = 500   # 批次大小,依 RAM 調整
# =============================================================================


# =============================================================================
# 環境變數設定(HuggingFace + OpenAI)
# =============================================================================

def configure_env_tokens() -> tuple[str | None, str | None]:
    """從 .env 載入 HF_TOKEN 與 OPENAI_API_KEY。"""
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

    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_API_TOKEN")
    )
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    openai_key = os.getenv("OPENAI_API_KEY")
    return hf_token, openai_key


# =============================================================================
# Stage 5 專用停用詞
# =============================================================================

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
    "end", "ends", "opening", "openings", "wall", "walls", "inlet", "inlets",
    "outlet", "outlets", "upper", "lower", "inner", "outer", "side", "sides",
    "surface", "surfaces", "member", "members",
    "high", "low", "equal", "material", "materials", "space", "spaces",
    "content", "percent", "chamber", "chambers", "line", "lines", "main",
    "level", "region", "regions", "image", "images", "imaging", "lens",
    "lenses", "video", "videos", "extruder", "extruders", "lt", "gt",
    "sub", "sup", "amp", "nbsp", "quot", "dwg", "cl", "cm", "kg", "hm",
    "partially", "uninterruptedly", "directly", "indirectly", "deals",
    "substance", "substances", "matter", "average", "layer", "visual",
    "information", "body", "export", "batch", "ls", "al", "vtd", "pgm", "pcd",
    "wt", "weight", "mass", "temperature", "degree", "celsius", "pressure",
    "mm", "vol", "lb", "btu",
}

CUSTOM_STOPWORDS = sorted(
    set(ENGLISH_STOP_WORDS) | PATENT_STOPWORDS | CORPUS_STOPWORDS
)


# =============================================================================
# STAGE 1:資料載入與物理清洗
# =============================================================================

def stage1_load_and_preprocess(input_path: str):
    print("=" * 60)
    print("STAGE 1:資料載入與物理清洗")
    print("=" * 60)

    data_path = Path(input_path)
    if data_path.is_file() and data_path.suffix.lower() == ".json":
        print(f"讀取 JSON 檔案:{data_path}")
        with data_path.open("r", encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))
    else:
        parquet_files = list(data_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"在 {input_path} 找不到任何 .parquet 檔案")
        print(f"找到 {len(parquet_files)} 個 Parquet 檔案,開始讀取...")
        df = pd.concat(
            [pd.read_parquet(f) for f in parquet_files], ignore_index=True
        )
    print(f"原始資料筆數:{len(df):,}")

    required_cols = ["application_number", "priority_date", "abstract_en"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要欄位:{missing}")

    optional_cols = [
        "publication_number", "title_en",
        "matched_cpc_codes", "domain_matched_prefixes", "target_matched_prefixes",
    ]
    keep_cols = required_cols + [c for c in optional_cols if c in df.columns]
    df = df[keep_cols].copy()

    before = len(df)
    df = df.dropna(subset=["abstract_en"])
    df = df[df["abstract_en"].astype(str).str.strip() != ""]
    print(f"移除 Abstract 空值:{before - len(df):,} 筆 → 剩餘 {len(df):,} 筆")

    print("執行物理清洗:HTML 解碼 + 移除 tag + 壓縮空白")

    def physical_clean(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = unescape(text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"&#\d+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df["abstract_clean"] = df["abstract_en"].apply(physical_clean)
    if "title_en" in df.columns:
        df["title_clean"] = df["title_en"].apply(physical_clean)
    else:
        df["title_clean"] = ""

    before = len(df)
    df["word_count"] = df["abstract_clean"].apply(lambda x: len(x.split()))
    df = df[df["word_count"] >= 30]
    print(f"移除 word_count < 30 的摘要:{before - len(df):,} 筆 → 剩餘 {len(df):,} 筆")

    df["priority_date"] = pd.to_datetime(
        df["priority_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df = df.dropna(subset=["priority_date"])

    def assign_segment(date):
        y = date.year
        if 2006 <= y <= 2010: return "SEG_A_2006_2010"
        if 2011 <= y <= 2015: return "SEG_B_2011_2015"
        if 2016 <= y <= 2020: return "SEG_C_2016_2020"
        if 2021 <= y <= 2025: return "SEG_D_2021_2025"
        return "OUT_OF_RANGE"

    df["time_segment"] = df["priority_date"].apply(assign_segment)
    df = df[df["time_segment"] != "OUT_OF_RANGE"].reset_index(drop=True)

    df["model_text"] = (
        df["title_clean"].fillna("").astype(str).str.strip()
        + ". "
        + df["title_clean"].fillna("").astype(str).str.strip()
        + ". "
        + df["abstract_clean"].fillna("").astype(str).str.strip()
    ).str.strip()

    abstracts = df["model_text"].tolist()
    embedding_texts = df["model_text"].tolist()

    print("\n各時間區段樣本數:")
    print(df["time_segment"].value_counts().sort_index().to_string())
    print(f"\nSTAGE 1 完成,最終語料筆數:{len(df):,}\n")
    return df, abstracts, embedding_texts


# =============================================================================
# STAGE 2:SPECTER2 嵌入 + 快取
# =============================================================================

class Specter2Embedder:
    def __init__(self, token: str | None = None):
        try:
            from adapters import AutoAdapterModel
        except ImportError as exc:
            raise ImportError("pip install -U adapters") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            "allenai/specter2_base", token=token
        )
        self.model = AutoAdapterModel.from_pretrained(
            "allenai/specter2_base", token=token
        )
        self.model.load_adapter(
            "allenai/specter2", source="hf",
            load_as="specter2", set_active=True,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def encode(
        self, texts, batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True, **kwargs,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        total = len(texts)
        for start in range(0, total, batch_size):
            batch = texts[start:start + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                return_tensors="pt", return_token_type_ids=False,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
                vec = out.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(vec)
            if show_progress_bar or total > 200:
                print(f"  embedded {min(start + batch_size, total):,}/{total:,}")
        return np.vstack(embeddings)


def stage2_embed(embedding_texts, df, batch_size: int = 32, use_cache: bool = True):
    print("=" * 60)
    print("STAGE 2:SPECTER2 語義嵌入")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hf_token, _ = configure_env_tokens()
    if hf_token:
        print("已從 .env 載入 HF token")

    embedding_model = Specter2Embedder(token=hf_token)

    if use_cache and EMBEDDING_CACHE.exists() and EMBEDDING_INDEX.exists():
        cached_idx = pd.read_parquet(EMBEDDING_INDEX)
        if (
            len(cached_idx) == len(df)
            and "embedding_input_mode" in cached_idx.columns
            and (cached_idx["embedding_input_mode"] == EMBEDDING_INPUT_MODE).all()
            and (cached_idx["application_number"].values
                 == df["application_number"].values).all()
        ):
            print(f"從快取載入:{EMBEDDING_CACHE.name}")
            embeddings = np.load(EMBEDDING_CACHE)
            print(f"快取 shape:{embeddings.shape}\n")
            return embedding_model, embeddings
        else:
            print("快取與當前資料不一致,重新計算 embedding")

    print(f"嵌入 {len(embedding_texts):,} 筆 abstract(batch={batch_size})...")
    embeddings = embedding_model.encode(embedding_texts, batch_size=batch_size)
    print(f"嵌入矩陣 shape:{embeddings.shape}")

    np.save(EMBEDDING_CACHE, embeddings)
    embedding_index = df[["application_number"]].copy()
    embedding_index["embedding_input_mode"] = EMBEDDING_INPUT_MODE
    embedding_index.to_parquet(EMBEDDING_INDEX, index=False)
    print(f"已寫入快取:{EMBEDDING_CACHE.name}\n")
    return embedding_model, embeddings


# =============================================================================
# STAGE 3:UMAP
# =============================================================================

def stage3_umap() -> UMAP:
    print("=" * 60)
    print("STAGE 3:UMAP 降維設定")
    print("=" * 60)
    umap_model = UMAP(
        n_neighbors=20, n_components=5,
        min_dist=0.0, metric="cosine", random_state=42,
    )
    print("UMAP:n_neighbors=20 | n_components=5 | metric=cosine\n")
    return umap_model


# =============================================================================
# STAGE 4:HDBSCAN
# =============================================================================

def stage4_hdbscan() -> HDBSCAN:
    print("=" * 60)
    print("STAGE 4:HDBSCAN 聚類設定")
    print("=" * 60)
    hdbscan_model = HDBSCAN(
        min_cluster_size=15, min_samples=8,
        metric="euclidean", cluster_selection_method="eom",
        prediction_data=True,
    )
    print("HDBSCAN:min_cluster_size=15 | min_samples=8 | eom\n")
    return hdbscan_model


# =============================================================================
# STAGE 5:c-TF-IDF
# =============================================================================

def stage5_vectorizer():
    print("=" * 60)
    print("STAGE 5:c-TF-IDF 特徵設定")
    print("=" * 60)

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 3),
        stop_words=CUSTOM_STOPWORDS,
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]+\b",
        min_df=2, max_df=0.95,
    )
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    print(f"Vectorizer:unigram/bigram/trigram | min_df=2 | max_df=0.95")
    print(f"停用詞數量:{len(CUSTOM_STOPWORDS)}\n")
    return vectorizer_model, ctfidf_model


# =============================================================================
# STAGE 6:BERTopic 訓練 + 表示法微調
# =============================================================================

LLM_LABEL_PROMPT = """You are an expert in patent technology classification, specifically in the domains of carbon capture, carbon neutrality, ironmaking, steelmaking, and metallurgy.

I have a topic that contains the following representative patent abstracts:
[DOCUMENTS]

The topic is described by the following keywords: [KEYWORDS]

Based on the documents and keywords above, generate ONE concise and specific topic label.

STRICT REQUIREMENTS:
1. The label MUST be written in English using ASCII letters only.
2. The label MUST be under 8 words.
3. Use precise technical terminology from the carbon-neutral metallurgy / ironmaking / steelmaking domain.
4. Reflect the dominant shared technical scope across the documents and keywords.
5. If the representative documents cover multiple closely related sub-processes,
choose a broader label that captures the common technical denominator.
6. Do NOT include numbering, quotation marks, punctuation, or any explanation.
7. Output ONLY the English label on a single line.

Topic label:"""


def build_representation_model(embedding_model, openai_key, include_llm=True):
    keybert = KeyBERTInspired(
        top_n_words=20, nr_repr_docs=5,
        nr_samples=500, nr_candidate_words=100,
    )
    mmr = MaximalMarginalRelevance(diversity=0.4)
    rep_dict = {"Main": [keybert, mmr]}

    if include_llm and openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            english_label = OpenAIRepresentation(
                client=client, model="gpt-4o-mini",
                chat=True, prompt=LLM_LABEL_PROMPT,
                nr_docs=6, doc_length=300,
                tokenizer="char", delay_in_seconds=1,
            )
            rep_dict["English_Label"] = english_label
            print("  ✓ OpenAI GPT-4o-mini 已掛載")
        except Exception as e:
            print(f"  ✗ OpenAI 初始化失敗:{e}")
    elif include_llm:
        print("  ⚠ 未提供 OPENAI_API_KEY,跳過 LLM 英文標籤")

    return rep_dict


def stage6_train_and_refine(
    abstracts, df, embedding_model, embeddings,
    umap_model, hdbscan_model, vectorizer_model, ctfidf_model,
    target_topics=None, apply_outlier_reduction=False,
):
    print("=" * 60)
    print("STAGE 6:BERTopic 訓練 + 表示法微調")
    print("=" * 60)

    _, openai_key = configure_env_tokens()
    keyword_representation_model = build_representation_model(
        embedding_model, openai_key=None, include_llm=False
    )
    final_representation_model = build_representation_model(
        embedding_model, openai_key, include_llm=True
    )

    # ── 6.1 建模(步驟①:啟用機率計算)─────────────────────────────────
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=keyword_representation_model,
        nr_topics=target_topics,
        top_n_words=10,
        calculate_probabilities=True,   # ★ 步驟①:開啟機率計算
        verbose=True,
    )

    # ── 6.2 訓練 ──────────────────────────────────────────────────────────
    print("\n開始訓練 BERTopic(calculate_probabilities=True)...")
    topics, probs = topic_model.fit_transform(abstracts, embeddings)

    initial_count = len(set(topics)) - (1 if -1 in topics else 0)
    noise_count = list(topics).count(-1)
    print(f"主題收斂後主題數:{initial_count} | 雜訊:{noise_count:,}")

    # ── 6.3 雜訊重新分配(可選)──────────────────────────────────────
    if apply_outlier_reduction and list(topic_model.topics_).count(-1):
        noise_before = list(topic_model.topics_).count(-1)
        print(f"執行 reduce_outliers,目前雜訊:{noise_before:,}")
        new_topics = topic_model.reduce_outliers(
            abstracts, topic_model.topics_,
            strategy="embeddings", embeddings=embeddings, threshold=0.10,
        )
        topic_model.topics_ = new_topics

    # ── 主題收斂後才掛 OpenAI,只對最終主題生成標籤 ──────────────────
    topic_model.update_topics(
        abstracts,
        topics=topic_model.topics_,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=final_representation_model,
    )

    final_topics = topic_model.topics_
    final_count = len(set(final_topics)) - (1 if -1 in final_topics else 0)
    final_noise = list(final_topics).count(-1)
    print(f"最終主題數:{final_count} | 雜訊:{final_noise:,} "
          f"({final_noise / len(final_topics):.1%})")

    # ── 6.4 輸出主題關鍵字 + 英文標籤 ──────────────────────────────────
    print("\n" + "=" * 60)
    print("主題關鍵字 + 英文標籤")
    print("=" * 60)

    topic_info = topic_model.get_topic_info()
    topic_info_no_noise = topic_info[topic_info["Topic"] != -1].copy()

    keyword_rows = []
    for _, row in topic_info_no_noise.iterrows():
        tid = int(row["Topic"])
        words = topic_model.get_topic(tid) or []
        keywords = " | ".join([w for w, _ in words[:10]])

        english_label = ""
        if "English_Label" in topic_info_no_noise.columns:
            raw = row.get("English_Label", "")
            if isinstance(raw, list) and raw:
                english_label = str(raw[0]).strip()
            elif isinstance(raw, str):
                english_label = raw.strip()

        keyword_rows.append({
            "Topic_ID": tid,
            "Doc_Count": int(row["Count"]),
            "English_Label": english_label,
            "Top10_Keywords": keywords,
        })
        print(f"Topic {tid:>3} ({int(row['Count']):>4}) "
              f"[{english_label}] | {keywords}")

    keywords_df = pd.DataFrame(keyword_rows)
    keywords_df.to_csv(OUTPUT_DIR / "topic_keywords.csv",
                       index=False, encoding="utf-8-sig")

    # ── 6.5 軟分配:approximate_distribution + threshold ─────────────────
    # ★ 步驟②:推算分佈矩陣  ★ 步驟③:設定篩選門檻
    print("\n" + "=" * 60)
    print(f"STAGE 6.5:軟分配 (Soft-Assignment)")
    print(f"  - approximate_distribution(window={APPROX_DIST_WINDOW}, "
          f"stride={APPROX_DIST_STRIDE})")
    print(f"  - 門檻 Threshold = {PROBABILITY_THRESHOLD}")
    print("=" * 60)

    # approximate_distribution 回傳 (n_docs, n_topics_without_noise) 機率矩陣
    # 注意:此矩陣的欄位順序對應「去除 -1 之後」的主題 ID,需透過 topic 列表查表
    topic_distr, _ = topic_model.approximate_distribution(
        abstracts,
        window=APPROX_DIST_WINDOW,
        stride=APPROX_DIST_STRIDE,
        use_embedding_model=False,
        calculate_tokens=False,
        separator=" ",
        batch_size=APPROX_DIST_BATCH_SIZE,
        padding=False,
    )
    print(f"分佈矩陣 shape:{topic_distr.shape}")

    # 建立「矩陣欄索引 → 真實 Topic_ID」的對應
    # BERTopic.approximate_distribution 的欄位順序為:不含 -1 的所有主題,按 ID 升序
    sorted_topic_ids = sorted([t for t in set(topic_model.topics_) if t != -1])
    if topic_distr.shape[1] != len(sorted_topic_ids):
        print(f"  ⚠ 警告:分佈矩陣欄數({topic_distr.shape[1]}) "
              f"與主題數({len(sorted_topic_ids)})不一致,以矩陣欄數為準")
        sorted_topic_ids = sorted_topic_ids[:topic_distr.shape[1]]

    # 套用門檻:對每一篇文件取出所有機率 > THRESHOLD 的主題
    multi_topics_col = []
    for i in range(topic_distr.shape[0]):
        probs_i = topic_distr[i]
        selected = [
            sorted_topic_ids[j]
            for j in range(len(probs_i))
            if probs_i[j] > PROBABILITY_THRESHOLD
        ]
        # 若無任何主題達標,回退到 hard assignment(若非雜訊則用該主題)
        if not selected:
            hard = topic_model.topics_[i]
            selected = [hard] if hard != -1 else []
        multi_topics_col.append(selected)

    # 統計多標籤覆蓋率
    n_multi = sum(1 for x in multi_topics_col if len(x) >= 2)
    n_single = sum(1 for x in multi_topics_col if len(x) == 1)
    n_empty = sum(1 for x in multi_topics_col if len(x) == 0)
    print(f"  ✓ 多標籤(≥2 主題)文件數:{n_multi:,} "
          f"({n_multi / len(multi_topics_col):.1%})")
    print(f"  ✓ 單標籤文件數:{n_single:,}")
    print(f"  ✓ 無標籤文件數(全部低於門檻且為雜訊):{n_empty:,}")

    # ── 6.6 回填 hard / soft 兩種主題欄位至 df ──────────────────────────
    df = df.copy()
    df["topic_id"] = topic_model.topics_                  # 硬分配
    df["multi_topics"] = multi_topics_col                 # 軟分配(list)

    # 對映英文標籤(便於後續閱讀)
    label_map = dict(zip(keywords_df["Topic_ID"], keywords_df["English_Label"]))
    df["topic_label"] = df["topic_id"].apply(
        lambda t: label_map.get(t, "NOISE" if t == -1 else "")
    )
    df["multi_topic_labels"] = df["multi_topics"].apply(
        lambda lst: [label_map.get(t, "") for t in lst]
    )

    # 儲存(parquet 支援 list 欄位)
    df.to_parquet(OUTPUT_DIR / "patent_with_topics.parquet", index=False)
    print(f"\n已儲存:patent_with_topics.parquet"
          f"(含 topic_id, multi_topics, multi_topic_labels)")

    # 額外儲存原始分佈矩陣供進階分析
    np.save(OUTPUT_DIR / "topic_distribution_matrix.npy", topic_distr)
    print(f"已儲存:topic_distribution_matrix.npy(shape={topic_distr.shape})")

    # 模型本體
    topic_model.save(
        str(OUTPUT_DIR / "bertopic_model"),
        serialization="safetensors",
        save_embedding_model=False,
        save_ctfidf=True,
    )
    print(f"已儲存:bertopic_model/\n")

    return topic_model, df, keywords_df


# =============================================================================
# STAGE 7:Topics over Time(含 .explode 多主題展開)
# =============================================================================

def classify_trajectory(freq, shares):
    """根據 4 個區段的頻次與佔比判定主題演進軌跡。"""
    early = sum(freq[:2])
    late = sum(freq[2:])
    if early + late == 0:
        return "無資料"
    growth = (late - early) / (early + 1)
    peak_idx = int(np.argmax(freq))
    if growth > 1.0 and peak_idx >= 2:
        return "新興爆發"
    if growth > 0.3 and peak_idx >= 2:
        return "持續成長"
    if growth < -0.3:
        return "明顯衰退"
    if peak_idx == 1 or peak_idx == 2:
        return "中期高峰"
    return "穩定成熟"


def stage7_topics_over_time(topic_model, df, abstracts, keywords_df):
    print("=" * 60)
    print("STAGE 7:Topics over Time(含 .explode 多主題展開)")
    print("=" * 60)

    SEGMENTS = [
        "SEG_A_2006_2010", "SEG_B_2011_2015",
        "SEG_C_2016_2020", "SEG_D_2021_2025",
    ]
    segment_to_code = {s: i for i, s in enumerate(SEGMENTS)}
    code_to_segment = {i: s for s, i in segment_to_code.items()}
    label_map = dict(zip(keywords_df["Topic_ID"], keywords_df["English_Label"]))

    # ── 7.0 步驟④:數據展開 (.explode) ──────────────────────────────────
    print("步驟④:執行 .explode('multi_topics') 數據展開")
    print("  → 讓跨界專利能同時對多個技術軸線做出貢獻")

    # 若有文件 multi_topics 為空 list,explode 後會變成 NaN,後續轉成 -1 過濾
    df_exploded = df.explode("multi_topics").reset_index(drop=True)
    df_exploded["multi_topics"] = (
        df_exploded["multi_topics"].fillna(-1).astype(int)
    )
    df_exploded["multi_topic_label"] = df_exploded["multi_topics"].apply(
        lambda t: label_map.get(t, "") if t != -1 else "NOISE"
    )

    print(f"  原始文件數:{len(df):,}")
    print(f"  展開後紀錄數:{len(df_exploded):,} "
          f"(平均每篇 {len(df_exploded) / len(df):.2f} 個主題)")

    df_exploded.to_parquet(
        OUTPUT_DIR / "patent_with_topics_exploded.parquet", index=False
    )
    print(f"  已儲存:patent_with_topics_exploded.parquet\n")

    # ── 7.1 用展開後資料計算 Topics over Time ────────────────────────────
    # 過濾掉雜訊(-1)再餵入 topics_over_time
    df_tot_input = df_exploded[df_exploded["multi_topics"] != -1].copy()
    exploded_abstracts = df_tot_input["model_text"].tolist()
    timestamps = df_tot_input["time_segment"].map(segment_to_code).tolist()
    exploded_topics = df_tot_input["multi_topics"].tolist()

    print(f"餵入 topics_over_time 的展開紀錄數:{len(exploded_abstracts):,}")
    tot = topic_model.topics_over_time(
        exploded_abstracts, timestamps,
        nr_bins=None,
        topics=exploded_topics,
        evolution_tuning=True,
        global_tuning=True,
    )
    tot["Timestamp"] = tot["Timestamp"].map(code_to_segment)
    tot.to_csv(OUTPUT_DIR / "topics_over_time.csv",
               index=False, encoding="utf-8-sig")
    print("TOT 已儲存:topics_over_time.csv\n")

    # ── 7.2 生命週期判定 ──────────────────────────────────────────────────
    lifecycle_rows = []
    for tid in sorted(tot["Topic"].unique()):
        if tid == -1:
            continue
        subset = tot[tot["Topic"] == tid]
        freq_map = dict(zip(subset["Timestamp"], subset["Frequency"]))
        freq = [freq_map.get(s, 0) for s in SEGMENTS]
        early, late = sum(freq[:2]), sum(freq[2:])
        growth = (late - early) / (early + 1)
        if growth > 0.5 and freq[0] < freq[-1]:
            status = "新興"
        elif growth < -0.3 and freq[0] > freq[-1]:
            status = "衰退"
        else:
            status = "成熟"
        lifecycle_rows.append({
            "Topic_ID": tid,
            "English_Label": label_map.get(tid, ""),
            "Freq_SEG_A": freq[0], "Freq_SEG_B": freq[1],
            "Freq_SEG_C": freq[2], "Freq_SEG_D": freq[3],
            "Growth_Rate": round(growth, 3),
            "Status": status,
        })
        print(f"Topic {tid:>3} | A:{freq[0]:>4} B:{freq[1]:>4} "
              f"C:{freq[2]:>4} D:{freq[3]:>4} | 成長:{growth:>+6.2f} | {status}")
    lifecycle_df = pd.DataFrame(lifecycle_rows)
    lifecycle_df.to_csv(OUTPUT_DIR / "topic_lifecycle.csv",
                        index=False, encoding="utf-8-sig")

    # ── 7.3 演進軌跡(年度 / 區段)──────────────────────────────────────
    # 這裡也用展開後的資料,確保跨界專利同時計入多個技術軸線
    topic_df = df_exploded[df_exploded["multi_topics"] != -1].copy()
    topic_df["priority_year"] = topic_df["priority_date"].dt.year

    yearly_counts = pd.crosstab(topic_df["priority_year"], topic_df["multi_topics"])
    yearly_counts = yearly_counts.reindex(range(2006, 2026), fill_value=0)
    yearly_counts.to_csv(OUTPUT_DIR / "topic_yearly_counts.csv",
                         encoding="utf-8-sig")

    yearly_totals = yearly_counts.sum(axis=1).replace(0, np.nan)
    yearly_shares = yearly_counts.div(yearly_totals, axis=0).fillna(0)
    yearly_shares.to_csv(OUTPUT_DIR / "topic_yearly_shares.csv",
                         encoding="utf-8-sig")

    segment_counts = pd.crosstab(topic_df["multi_topics"], topic_df["time_segment"])
    segment_counts = segment_counts.reindex(columns=SEGMENTS, fill_value=0)
    segment_counts.to_csv(OUTPUT_DIR / "topic_segment_counts.csv",
                          encoding="utf-8-sig")

    segment_totals = topic_df["time_segment"].value_counts().reindex(SEGMENTS, fill_value=0)
    segment_shares = segment_counts.div(
        segment_totals.replace(0, np.nan), axis=1
    ).fillna(0)
    segment_shares.to_csv(OUTPUT_DIR / "topic_segment_shares.csv",
                          encoding="utf-8-sig")

    keyword_map = dict(zip(keywords_df["Topic_ID"], keywords_df["Top10_Keywords"]))
    midpoints = np.array([2008, 2013, 2018, 2023], dtype=float)

    evolution_rows = []
    for tid in sorted(segment_counts.index):
        freq = [int(segment_counts.loc[tid, s]) for s in SEGMENTS]
        shares = [float(segment_shares.loc[tid, s]) for s in SEGMENTS]
        slope = float(np.polyfit(midpoints, shares, 1)[0])
        peak_idx = int(np.argmax(freq))
        evolution_rows.append({
            "Topic_ID": tid,
            "English_Label": label_map.get(tid, ""),
            "Top10_Keywords": keyword_map.get(tid, ""),
            "Doc_Count_Exploded": int(sum(freq)),   # 展開後的次數(可大於文件數)
            "Freq_2006_2010": freq[0], "Freq_2011_2015": freq[1],
            "Freq_2016_2020": freq[2], "Freq_2021_2025": freq[3],
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
    evolution_df.to_csv(OUTPUT_DIR / "topic_evolution_summary.csv",
                        index=False, encoding="utf-8-sig")

    print(f"\nSTAGE 7 完成\n")
    return tot, lifecycle_df, evolution_df


# =============================================================================
# 主程式入口
# =============================================================================

if __name__ == "__main__":
    INPUT_PATH = str(
        PROJECT_DIR / "data_globalmorecpc" / "global_onlycpc_carbon_neutral_v2.json"
    )
    TARGET_TOPICS = None

    # STAGE 1
    df, abstracts, embedding_texts = stage1_load_and_preprocess(INPUT_PATH)

    # STAGE 2
    embedding_model, embeddings = stage2_embed(
        embedding_texts, df, batch_size=32, use_cache=False
    )

    # STAGE 3
    umap_model = stage3_umap()

    # STAGE 4
    hdbscan_model = stage4_hdbscan()

    # STAGE 5
    vectorizer_model, ctfidf_model = stage5_vectorizer()

    # STAGE 6 + 6.5(軟分配)
    topic_model, df, keywords_df = stage6_train_and_refine(
        abstracts, df,
        embedding_model, embeddings,
        umap_model, hdbscan_model,
        vectorizer_model, ctfidf_model,
        target_topics=TARGET_TOPICS,
        apply_outlier_reduction=False,
    )

    # STAGE 7(.explode 展開)
    tot_df, lifecycle_df, evolution_df = stage7_topics_over_time(
        topic_model, df, abstracts, keywords_df
    )

    print("=" * 60)
    print("全流程執行完畢")
    print(f"輸出目錄:{OUTPUT_DIR}")
    print("  ├── specter2_embeddings.npy")
    print("  ├── topic_distribution_matrix.npy   ← 新增(軟分配機率矩陣)")
    print("  ├── patent_with_topics.parquet      ← 含 multi_topics 欄")
    print("  ├── patent_with_topics_exploded.parquet  ← 新增(展開後)")
    print("  ├── topic_keywords.csv")
    print("  ├── topics_over_time.csv")
    print("  ├── topic_lifecycle.csv")
    print("  ├── topic_evolution_summary.csv")
    print("  ├── topic_yearly_counts.csv / topic_yearly_shares.csv")
    print("  ├── topic_segment_counts.csv / topic_segment_shares.csv")
    print("  └── bertopic_model/")
    print("=" * 60)