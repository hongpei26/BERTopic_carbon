# =============================================================================
# 碳捕捉專利 BERTopic 動態主題建模 Pipeline v2(對齊方法論)
# Stage 1–7:資料載入 → SPECTER2 → UMAP → HDBSCAN → c-TF-IDF
#           → KeyBERT+MMR+LLM 表示法微調 → TOT
# -----------------------------------------------------------------------------
# 主要調整(對齊方法論):
#  Stage 1:只做物理清洗,保留標點、停用詞、大小寫,供 Transformer 完整理解語意
#  Stage 2:加入 embedding 快取(.npy),避免反覆計算
#  Stage 5:停用詞與小寫化的「真正落腳點」,只影響 c-TF-IDF 關鍵字展示
#  Stage 6:加入 KeyBERTInspired → MMR → OpenAI GPT-4o-mini 英文標籤微調流水線
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

PROJECT_DIR = Path(__file__).resolve().parent.parent

# 使用 weighted_related_patents.json 後的新輸出資料夾
# OUTPUT_DIR = PROJECT_DIR / "output_specter2_weighted_related_robust"
OUTPUT_DIR = PROJECT_DIR / "output_specter2_weighted_related_main"

EMBEDDING_CACHE = OUTPUT_DIR / "specter2_embeddings.npy"
EMBEDDING_INDEX = OUTPUT_DIR / "specter2_embeddings_index.parquet"
EMBEDDING_INPUT_MODE = "title_title_abstract"


# =============================================================================
# 環境變數設定(HuggingFace + OpenAI)
# =============================================================================

def configure_env_tokens() -> tuple[str | None, str | None]:
    """
    從 /home/carbon/carbon/.env 載入 HF_TOKEN 與 OPENAI_API_KEY。
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
# Stage 5 專用停用詞(注意:Stage 1 不再使用)
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
# STAGE 1:資料載入與「物理清洗」(對齊方法論:不做語意閹割)
# =============================================================================

def stage1_load_and_preprocess(input_path: str, sample_mode: str = "robust"):
    """
    讀取 JSON / Parquet,僅做物理清洗:
      - HTML entity 解碼
      - 移除 HTML / XML 標籤與殘留 token
      - 壓縮多餘空白
    保留:大小寫、標點符號、停用詞、完整句型結構
    (因為 SPECTER2 需要完整上下文才能精準理解技術語意)
    """

    print("=" * 60)
    print("STAGE 1:資料載入與物理清洗(保留完整句型供 Transformer 理解)")
    print("=" * 60)

    # ── 1.1 讀取資料 ──────────────────────────────────────────────────────
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

    # ── 1.1.1 依 final_label 選擇樣本口徑 ─────────────────────────────
    if "final_label" in df.columns:
        before = len(df)

        if sample_mode == "main":
            df = df[df["final_label"] == "related"].copy()
            print(
                f"樣本模式 main：只保留 related，"
                f"移除 {before - len(df):,} 筆 → 剩餘 {len(df):,} 筆"
            )

        elif sample_mode == "robust":
            df = df[df["final_label"].isin([
                "related",
                "weak_related",
                "weak_related_audit",
            ])].copy()
            print(
                f"樣本模式 robust：保留 related + weak_related + weak_related_audit，"
                f"移除 {before - len(df):,} 筆 → 剩餘 {len(df):,} 筆"
            )

        else:
            raise ValueError("sample_mode must be 'main' or 'robust'")
    else:
        print("未偵測到 final_label 欄位，將使用全部資料。")

    # ── 1.2 保留核心欄位 ──────────────────────────────────────────────────
    required_cols = ["application_number", "priority_date", "abstract_en"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要欄位:{missing}")

    optional_cols = [
        "publication_number",
        "title_en",

        # 前面加權篩選結果
        "final_label",
        "relevance_label",
        "relevance_rule",
        "total_score",
        "cpc_score",
        "high_score",
        "decarbon_score",
        "steel_score",
        "action_score",
        "noise_penalty",
        "adjusted_noise_penalty",
        "noise_adjusted",
        "has_decarbon_context",
        "cpc_strength",

        # CPC 與關鍵字命中
        "matched_cpc_codes",
        "core_cpc_hits",
        "support_cpc_hits",
        "peripheral_cpc_hits",
        "high_hits",
        "decarbon_hits",
        "steel_hits",
        "action_hits",
        "noise_hits",
        "audit_high_confidence",
        "audit_categories",
        "audit_hits",

        # 原本 CPC 篩選來源欄位
        "domain_matched_prefixes",
        "target_matched_prefixes",
        "domain_source",
        "all_cpc_codes",
        "all_ipc_codes",
        "priority_year",
    ]
    keep_cols = required_cols + [c for c in optional_cols if c in df.columns]
    df = df[keep_cols].copy()

    # ── 1.3 移除空值 ──────────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["abstract_en"])
    df = df[df["abstract_en"].astype(str).str.strip() != ""]
    print(f"移除 Abstract 空值:{before - len(df):,} 筆 → 剩餘 {len(df):,} 筆")

    # ── 1.4 物理清洗(只做這幾件事)────────────────────────────────────
    print("執行物理清洗:HTML 解碼 + 移除 tag + 壓縮空白")
    print("(保留大小寫、標點符號、停用詞 → 供 SPECTER2 完整理解語意)")

    def physical_clean(text: str) -> str:
        if not isinstance(text, str):
            return ""
        # 1) HTML entity 解碼(&amp; → &, &lt; → <, etc.)
        text = unescape(text)
        # 2) 移除 HTML / XML 標籤
        text = re.sub(r"<[^>]+>", " ", text)
        # 3) 移除殘留的 HTML entity 文字形式(若 unescape 後仍存在)
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"&#\d+;", " ", text)
        # 4) 壓縮多餘空白(包含換行、tab)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df["abstract_clean"] = df["abstract_en"].apply(physical_clean)
    if "title_en" in df.columns:
        df["title_clean"] = df["title_en"].apply(physical_clean)
    else:
        df["title_clean"] = ""

    # ── 1.5 篩除過短摘要(< 30 個 token,以空白切)──────────────────────
    before = len(df)
    df["word_count"] = df["abstract_clean"].apply(lambda x: len(x.split()))
    df = df[df["word_count"] >= 30]
    print(f"移除 word_count < 30 的摘要:{before - len(df):,} 筆 → 剩餘 {len(df):,} 筆")

    # ── 1.6 priority_date 型別轉換與時間區段標記 ──────────────────────────
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

    # ── 1.7 建立 BERTopic 與 SPECTER2 的輸入文本 ──────────────────────────
    # Title 重複一次以提高短標題中技術詞的權重,同時保留 abstract 完整上下文。
    # CountVectorizer 會在 Stage 5 自行做小寫化 + 停用詞過濾。
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
# STAGE 2:SPECTER2 科學語義嵌入 + 快取
# =============================================================================

class Specter2Embedder:
    """
    SPECTER2 base + proximity adapter,以 [CLS] token 作為文獻向量。
    亦提供 encode() 介面,可作為 BERTopic representation_model 的 embedding_model。
    """

    def __init__(self, token: str | None = None):
        try:
            from adapters import AutoAdapterModel
        except ImportError as exc:
            raise ImportError(
                "使用 SPECTER2 需先安裝 adapters:pip install -U adapters"
            ) from exc

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
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,  # 相容 sentence-transformers 介面
        convert_to_numpy: bool = True,
        **kwargs,
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


def stage2_embed(
    embedding_texts: list,
    df: pd.DataFrame,
    batch_size: int = 32,
    use_cache: bool = True,
):
    """
    SPECTER2 嵌入 + .npy 快取機制
      - 若快取存在且筆數一致,直接讀取
      - 否則重新計算並寫入快取
    """

    print("=" * 60)
    print("STAGE 2:SPECTER2 語義嵌入(含快取)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hf_token, _ = configure_env_tokens()
    if hf_token:
        print("已從 .env 載入 HF token")

    embedding_model = Specter2Embedder(token=hf_token)

    # ── 快取讀取 ──────────────────────────────────────────────────────────
    if use_cache and EMBEDDING_CACHE.exists() and EMBEDDING_INDEX.exists():
        cached_idx = pd.read_parquet(EMBEDDING_INDEX)
        if (
            len(cached_idx) == len(df)
            and "embedding_input_mode" in cached_idx.columns
            and (cached_idx["embedding_input_mode"] == EMBEDDING_INPUT_MODE).all()
            and (
                cached_idx["application_number"].values
                == df["application_number"].values
            ).all()
        ):
            print(f"從快取載入:{EMBEDDING_CACHE.name}")
            embeddings = np.load(EMBEDDING_CACHE)
            print(f"快取 shape:{embeddings.shape}\n")
            return embedding_model, embeddings
        else:
            print("快取與當前資料不一致,重新計算 embedding")

    # ── 重新計算 ──────────────────────────────────────────────────────────
    print(f"嵌入 {len(embedding_texts):,} 筆 abstract(batch={batch_size})...")
    embeddings = embedding_model.encode(embedding_texts, batch_size=batch_size)
    print(f"嵌入矩陣 shape:{embeddings.shape}")

    # ── 寫入快取 ──────────────────────────────────────────────────────────
    np.save(EMBEDDING_CACHE, embeddings)
    embedding_index = df[["application_number"]].copy()
    embedding_index["embedding_input_mode"] = EMBEDDING_INPUT_MODE
    embedding_index.to_parquet(EMBEDDING_INDEX, index=False)
    print(f"已寫入快取:{EMBEDDING_CACHE.name}\n")

    return embedding_model, embeddings


# =============================================================================
# STAGE 3:UMAP 降維
# =============================================================================

def stage3_umap() -> UMAP:
    print("=" * 60)
    print("STAGE 3:UMAP 降維設定")
    print("=" * 60)
    umap_model = UMAP(
        n_neighbors=20,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    print("UMAP:n_neighbors=20 | n_components=5 | metric=cosine\n")
    return umap_model


# =============================================================================
# STAGE 4:HDBSCAN 密度聚類
# =============================================================================

def stage4_hdbscan() -> HDBSCAN:
    print("=" * 60)
    print("STAGE 4:HDBSCAN 聚類設定")
    print("=" * 60)
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        min_samples=8,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    print("HDBSCAN:min_cluster_size=15 | min_samples=8 | eom\n")
    return hdbscan_model


# =============================================================================
# STAGE 5:c-TF-IDF(停用詞與小寫化的「真正落腳點」)
# =============================================================================

def stage5_vectorizer():
    """
    CountVectorizer 在此階段:
      - lowercase=True(預設):此處才轉小寫,不影響 Stage 2 嵌入
      - stop_words=CUSTOM_STOPWORDS:此處才去停用詞
      - ngram_range=(1, 3):支援 electric arc furnace 等技術詞
    """
    print("=" * 60)
    print("STAGE 5:c-TF-IDF 特徵設定(停用詞與小寫化在此落腳)")
    print("=" * 60)

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 3),
        stop_words=CUSTOM_STOPWORDS,
        lowercase=True,  # 明確標示:小寫化在這裡才發生
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]+\b",
        min_df=2,        # 略提高 min_df,過濾極稀有雜訊詞
        max_df=0.95,     # 過濾跨主題高頻泛用詞
    )
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    print(f"Vectorizer:unigram/bigram/trigram | min_df=2 | max_df=0.95")
    print(f"停用詞數量:{len(CUSTOM_STOPWORDS)}\n")
    return vectorizer_model, ctfidf_model


# =============================================================================
# STAGE 6:BERTopic 訓練 + 表示法多層級微調(KeyBERT → MMR → LLM)
# =============================================================================

LLM_LABEL_PROMPT = """You are an expert in patent technology classification, specifically in the domains of carbon capture, carbon neutrality, ironmaking, steelmaking, and metallurgy.

I have a topic that contains the following representative patent abstracts:
[DOCUMENTS]

The topic is described by the following keywords: [KEYWORDS]

Based on the documents and keywords above, generate ONE concise and specific topic label.

STRICT REQUIREMENTS:
1. The label MUST be written in English using ASCII letters only.
2. The label MUST be under 8 words.
3. Use precise technical terminology from the carbon-neutral metallurgy / ironmaking / steelmaking domain (e.g., hydrogen shaft furnace, direct reduced iron, electric arc furnace, converter gas recovery, pulverized coal injection, submerged arc furnace, waste heat recovery, red mud valorization).
4. Reflect the dominant shared technical scope across the documents and keywords.
5. If the representative documents cover multiple closely related sub-processes,
choose a broader label that captures the common technical denominator.
Do not over-focus on only one representative document.
6. Do NOT include numbering, quotation marks, punctuation, or any explanation.
7. Output ONLY the English label on a single line.

Example outputs (format reference only, do not copy):
Blast Furnace Energy Optimization
Blast Furnace Waste Heat Recovery
Hydrogen Shaft Furnace DRI
Hydrogen Metallurgy Ironmaking
Low Carbon Reductant Injection
Electric Arc Furnace Scrap Recycling
Electric Arc Furnace Energy Efficiency
Converter Gas Heat Recovery
Steelmaking Off Gas CCUS
Smart Energy Monitoring
Slag Byproduct Valorization
Red Mud Iron Recovery

Topic label:"""


def build_representation_model(
    embedding_model,
    openai_key: str | None,
    include_llm: bool = True,
):
    """
    建立三層串聯的表示法微調流水線:
      Main 關鍵字:[KeyBERTInspired → MMR]
      English_Label:OpenAI GPT-4o-mini(英文標籤,可選)
    """
    # ── 第一層:KeyBERTInspired(語意過濾)──────────────────────────────
    keybert = KeyBERTInspired(
        top_n_words=20,        # 候選詞池
        nr_repr_docs=5,        # 代表文件數
        nr_samples=500,        # 取樣文件數
        nr_candidate_words=100,
    )

    # ── 第二層:MMR(多樣性去重)───────────────────────────────────────
    mmr = MaximalMarginalRelevance(diversity=0.4)  # 0.4 兼顧多樣性與相關性

    # ── 第三層:OpenAI GPT-4o-mini(英文標籤)─────────────────────
    rep_dict = {
        "Main": [keybert, mmr],
    }

    if include_llm and openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            english_label = OpenAIRepresentation(
                client=client,
                model="gpt-4o-mini",      
                chat=True,
                prompt=LLM_LABEL_PROMPT,
                nr_docs=6,                # 餵 6 篇最具代表性的摘要
                doc_length=300,           # 每篇摘要截斷至 300 token
                tokenizer="char",
                delay_in_seconds=1,
            )
            rep_dict["English_Label"] = english_label
            print("  ✓ OpenAI GPT-4o-mini 已掛載,將產生英文主題標籤")
        except Exception as e:
            print(f"  ✗ OpenAI 初始化失敗(略過 LLM 標籤):{e}")
    elif include_llm:
        print("  ⚠ 未提供 OPENAI_API_KEY,跳過 LLM 英文標籤生成")

    return rep_dict


def stage6_train_and_refine(
    abstracts, df, embedding_model, embeddings,
    umap_model, hdbscan_model, vectorizer_model, ctfidf_model,
    target_topics: int | None = None,
    apply_outlier_reduction: bool = False,
):
    """
    BERTopic 訓練 + 表示法多層級微調 + Topic ID 回填
    """

    print("=" * 60)
    print("STAGE 6:BERTopic 訓練 + KeyBERT → MMR → LLM 表示法微調")
    print("=" * 60)

    _, openai_key = configure_env_tokens()
    keyword_representation_model = build_representation_model(
        embedding_model, openai_key=None, include_llm=False
    )
    final_representation_model = build_representation_model(
        embedding_model, openai_key, include_llm=True
    )

    # ── 6.1 建模 ──────────────────────────────────────────────────────────
    # embedding_model 傳入 SPECTER2 是為了讓 KeyBERTInspired 能對候選詞重新嵌入
    # 訓練/收斂階段先不掛 OpenAI,避免對會被合併的暫時主題產生標籤成本
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=keyword_representation_model,
        nr_topics=target_topics,
        top_n_words=10,
        calculate_probabilities=False,  # 提速;若需 probs 再開啟
        verbose=True,
    )

    # ── 6.2 訓練(傳入預計算 embeddings)─────────────────────────────────
    print("\n開始訓練 BERTopic...")
    topics, probs = topic_model.fit_transform(abstracts, embeddings)

    initial_count = len(set(topics)) - (1 if -1 in topics else 0)
    noise_count = list(topics).count(-1)
    print(f"主題收斂後主題數:{initial_count} | 雜訊:{noise_count:,}")

    # ── 6.3 雜訊重新分配(可選)──────────────────────────────────────
    noise_before = list(topic_model.topics_).count(-1)
    if apply_outlier_reduction and noise_before:
        print(f"執行 reduce_outliers,目前雜訊:{noise_before:,}")
        new_topics = topic_model.reduce_outliers(
            abstracts,
            topic_model.topics_,
            strategy="embeddings",
            embeddings=embeddings,
            threshold=0.10,
        )
        topic_model.topics_ = new_topics

    # 重要:主題收斂與雜訊重分配後才掛 OpenAI,只為最終主題產生標籤
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

    # ── 6.4 輸出主題關鍵字 + 英文標籤 ────────────────────────────────────
    print("\n" + "=" * 60)
    print("主題關鍵字 + 英文標籤")
    print("=" * 60)

    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info["Topic"] != -1]

    keyword_rows = []
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        words = topic_model.get_topic(tid)
        keywords = " | ".join([w for w, _ in words[:10]])

        # 取 LLM 標籤(若有)
        english_label = ""
        if "English_Label" in topic_info.columns:
            raw = row.get("English_Label", "")
            if isinstance(raw, list) and raw:
                english_label = str(raw[0]).strip()
            elif isinstance(raw, str):
                english_label = raw.strip()

        keyword_rows.append({
            "Topic_ID": tid,
            "English_Label": english_label,
            "Doc_Count": row["Count"],
            "Top10_Keywords": keywords,
        })

        label_show = english_label if english_label else "(無 LLM 標籤)"
        print(f"Topic {tid:>3} ({row['Count']:>5} 筆) {label_show}")
        print(f"          {keywords}")

    keywords_df = pd.DataFrame(keyword_rows)

    # ── 6.5 回填 Topic ID + English_Label 至 DataFrame ───────────────────
    df = df.copy()
    df["topic_id"] = topic_model.topics_

    label_map = dict(zip(keywords_df["Topic_ID"], keywords_df["English_Label"]))
    df["topic_label"] = df["topic_id"].apply(
        lambda t: label_map.get(t, "") if t != -1 else "NOISE"
    )

    # ── 6.6 儲存 ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_DIR / "patent_with_topics.parquet", index=False)
    keywords_df.to_csv(
        OUTPUT_DIR / "topic_keywords.csv", index=False, encoding="utf-8-sig"
    )
    # 儲存 BERTopic 模型本身(可重新載入做 TOT)
    try:
        topic_model.save(
            OUTPUT_DIR / "bertopic_model",
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=False,
        )
        print(f"\nBERTopic 模型已儲存:{OUTPUT_DIR / 'bertopic_model'}")
    except Exception as e:
        print(f"\nBERTopic 模型儲存失敗(略過):{e}")

    print(f"結果已儲存至:{OUTPUT_DIR}")
    print(f"\nSTAGE 6 完成\n")

    return topic_model, df, keywords_df


# =============================================================================
# STAGE 7:Topics over Time(與 v1 相同邏輯)
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
    print("=" * 60)
    print("STAGE 7:Topics over Time 動態分析")
    print("=" * 60)

    SEGMENTS = [
        "SEG_A_2006_2010", "SEG_B_2011_2015",
        "SEG_C_2016_2020", "SEG_D_2021_2025",
    ]
    segment_to_code = {s: i for i, s in enumerate(SEGMENTS)}
    code_to_segment = {i: s for s, i in segment_to_code.items()}

    timestamps = df["time_segment"].map(segment_to_code).tolist()

    tot = topic_model.topics_over_time(
        abstracts, timestamps,
        nr_bins=None,
        evolution_tuning=True,
        global_tuning=True,
    )
    tot["Timestamp"] = tot["Timestamp"].map(code_to_segment)
    tot.to_csv(OUTPUT_DIR / "topics_over_time.csv",
               index=False, encoding="utf-8-sig")
    print("TOT 已儲存:topics_over_time.csv")

    # ── 生命週期 ──────────────────────────────────────────────────────────
    label_map = dict(zip(keywords_df["Topic_ID"], keywords_df["English_Label"]))
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

    # ── 演進軌跡 ──────────────────────────────────────────────────────────
    topic_df = df[df["topic_id"] != -1].copy()
    topic_df["priority_year"] = topic_df["priority_date"].dt.year

    yearly_counts = pd.crosstab(topic_df["priority_year"], topic_df["topic_id"])
    yearly_counts = yearly_counts.reindex(range(2006, 2026), fill_value=0)
    yearly_counts.to_csv(OUTPUT_DIR / "topic_yearly_counts.csv",
                         encoding="utf-8-sig")

    yearly_totals = yearly_counts.sum(axis=1).replace(0, np.nan)
    yearly_shares = yearly_counts.div(yearly_totals, axis=0).fillna(0)
    yearly_shares.to_csv(OUTPUT_DIR / "topic_yearly_shares.csv",
                         encoding="utf-8-sig")

    segment_counts = pd.crosstab(topic_df["topic_id"], topic_df["time_segment"])
    segment_counts = segment_counts.reindex(columns=SEGMENTS, fill_value=0)
    segment_counts.to_csv(OUTPUT_DIR / "topic_segment_counts.csv",
                          encoding="utf-8-sig")

    segment_totals = topic_df["time_segment"].value_counts().reindex(SEGMENTS, fill_value=0)
    segment_shares = segment_counts.div(segment_totals.replace(0, np.nan), axis=1).fillna(0)
    segment_shares.to_csv(OUTPUT_DIR / "topic_segment_shares.csv",
                          encoding="utf-8-sig")

    keyword_map = dict(zip(keywords_df["Topic_ID"], keywords_df["Top10_Keywords"]))
    count_map = dict(zip(keywords_df["Topic_ID"], keywords_df["Doc_Count"]))
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
            "Doc_Count": int(count_map.get(tid, sum(freq))),
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
    # main：只使用 final_label == related
    # robust：使用 related + weak_related + weak_related_audit
    # SAMPLE_MODE = "robust"
    SAMPLE_MODE = "main"

    INPUT_PATH = (
        "/home/carbon/carbon/data_global_v2/"
        "Carbon_onlycpc_global_morecpc_v2/"
        "weighted_relevance_output/"
        "weighted_related_patents.json"
    )

    TARGET_TOPICS = None

    # STAGE 1:資料載入 + 物理清洗
    df, abstracts, embedding_texts = stage1_load_and_preprocess(
        INPUT_PATH,
        sample_mode=SAMPLE_MODE,
    )

    # STAGE 2:SPECTER2 嵌入(含快取)
    embedding_model, embeddings = stage2_embed(
        embedding_texts, df, batch_size=32, use_cache=True
    )

    # STAGE 3:UMAP
    umap_model = stage3_umap()

    # STAGE 4:HDBSCAN
    hdbscan_model = stage4_hdbscan()

    # STAGE 5:c-TF-IDF
    vectorizer_model, ctfidf_model = stage5_vectorizer()

    # STAGE 6:訓練 + 表示法多層級微調
    topic_model, df, keywords_df = stage6_train_and_refine(
        abstracts, df,
        embedding_model, embeddings,
        umap_model, hdbscan_model,
        vectorizer_model, ctfidf_model,
        target_topics=TARGET_TOPICS,
        apply_outlier_reduction=False,
    )

    # STAGE 7:Topics over Time
    tot_df, lifecycle_df, evolution_df = stage7_topics_over_time(
        topic_model, df, abstracts, keywords_df
    )

    print("=" * 60)
    print("全流程執行完畢")
    print(f"輸出目錄:{OUTPUT_DIR}")
    print("  ├── specter2_embeddings.npy(嵌入快取)")
    print("  ├── patent_with_topics.parquet")
    print("  ├── topic_keywords.csv(含 English_Label 欄)")
    print("  ├── topics_over_time.csv")
    print("  ├── topic_lifecycle.csv")
    print("  ├── topic_evolution_summary.csv")
    print("  ├── topic_yearly_counts.csv / topic_yearly_shares.csv")
    print("  └── bertopic_model/(模型 safetensors)")
    print("=" * 60)
