#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鋼鐵業碳中和專利摘要抽取 — 雙層弱監督機器學習分類管線
=============================================================================
本程式旨在自海量專利數據庫中，精準提取出「鋼鐵業碳中和相關技術」專利。
採用了「規則弱監督 (Rule-based Weak Supervision) + 統計機器學習 (Logistic Regression)」的雙層管線：

1. **第一層：弱監督種子標註 (Weak Supervision)**
   - 利用預設的高精準度「場域詞 (FIELD)」與「正向低碳訊號 (POS_STRONG)」規則，自動篩選出強正例種子 (y=1)。
   - 同時利用擴展後的「負向干擾詞 (NEG_RULES)」排除「鋼水精煉品質脫碳」、「有色冶金」與「下游環保吸附」等雜訊，標記為強負例種子 (y=0)。
   - **防污染機制**：正向種子絕對不能包含任何負向特徵，確保機器學習訓練集的純淨度。

2. **第二層：TF-IDF 特徵提取與邏輯迴歸 (Logistic Regression)**
   - 將「標題 + 摘要」轉化為 TF-IDF 詞頻矩陣（支援 Unigram 與 Bigram，擷取雙字組技術特徵）。
   - 以第一層產出的自動標籤訓練一個帶有類別平衡 (class_weight='balanced') 的邏輯迴歸分類器。
   - 對全部專利計算相關度機率 (`relevance_prob`)，凡機率 >= 0.50 或命中第一層強正向訊號者皆予以收入。

3. **第三層：信心分層與結果導出 (Output & Tiering)**
   - 根據預測機率將收錄結果劃分為 A (高信心)、B (中高信心)、C (中信心) 等三個信心區段。
   - 導出精準的 CSV 數據集，供後續主題建模 (BERTopic) 使用。
=============================================================================
"""

import json
import re
import html
import csv
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ============================================================================
# 1. 路徑、參數與全局設定
# ============================================================================
# 輸入專利庫的 JSON 檔案路徑（交集篩選後的數據）
INPUT_PATH = Path(
    "/home/carbon/carbon/data_global_v2/Carbon_onlycpc_global_morecpc_v2/"
    "global_onlycpc_domain_target_intersection.json"
)
OUTPUT_DIR = INPUT_PATH.parent
# 最終篩選並純化後的 CSV 輸出路徑
OUTPUT_CSV = OUTPUT_DIR / "steel_carbonneutral_extracted.csv"
# 另外輸出 JSON 檔案路徑（格式與原始輸入完全相同）
OUTPUT_JSON = OUTPUT_DIR / "steel_carbonneutral_extracted.json"

ABSTRACT_FIELD = "abstract_en"   # 主要判定欄位（英文摘要）
TITLE_FIELD = "title_en"         # 標題欄位（標題會併入摘要文本一起進行特徵分析）
THRESHOLD = 0.50                 # 第二層分類器的收入機率門檻
RANDOM_SEED = 42                 # 固定隨機種子以確保實驗可重複性

# 可選設定：是否啟用句向量 (Embedding) 重排序（預設關閉以維持輕量與快速執行）
# 啟用時需要本地具備 sentence-transformers 套件與網路連接
USE_EMBEDDING_RERANK = False
ENV_PATH = Path("/home/carbon/carbon/.env")
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================================
# 2. 關鍵字與正規表示式規則庫 (Regex Rule Base)
# ============================================================================
# ----------------------------------------------------------------------------
# 2.1 應用場域詞 (FIELD)
# 說明：正例專利必須命中以下至少一個鋼鐵/鐵水/煉鋼/冶金現場的應用場域。
# ----------------------------------------------------------------------------
FIELD = re.compile(
    r"\b(blast furnace|converter|basic oxygen furnace|electric arc furnace|eaf|"
    r"steelmaking|ironmaking|molten iron|molten steel|direct reduced iron|dri|"
    r"sponge iron|shaft furnace|coke oven|steel slag|steelworks|steel mill|sinter|"
    r"pellet|ferrous metallurg|iron and steel|metallurgical)\b",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------------
# 2.2 強正向低碳/碳中和技術訊號 (POS_STRONG)
# 說明：採用高精準度的技術特徵片語。部分規則使用 `.{0,N}` 限制關鍵詞在鄰近距離內出現，
#      以兼顧長句型中的語意關聯性（例如：廢熱與回收在 40 個字元內鄰近出現）。
# ----------------------------------------------------------------------------
POS_STRONG = [
    re.compile(p, re.IGNORECASE) for p in [
        # (1) 二氧化碳捕捉、利用與封存 (CCUS) 類
        r"\b(co2 capture|carbon capture|ccus|\bccs\b|carbon sequestration|"
        r"carbon neutral|carbon-neutral|green steel|decarboni[sz])",
        
        # (2) 氫能冶金與前沿非碳還原技術
        r"\b(hydrogen reduction|hydrogen metallurg|hydrogen-rich|hydrogen-based|"
        r"hydrogen rich|molten oxide electrolysis|h2 reduction)",
        
        # (3) 近零碳/超低排放之宣示性語意
        r"\b(near[\s-]?zero emission|zero[\s-]?carbon|ultra[\s-]?low emission|"
        r"low[\s-]carbon (metallurg|process|production|technolog|develop|"
        r"ironmaking|steelmaking))",
        
        # (4) 鋼廠高溫廢熱與顯熱回收 (Waste Heat & Sensible Heat Recovery)
        r"\b(waste heat|sensible heat)\b.{0,40}\b(recover|recovery|utiliz|generation|power)",
        r"\b(recover|recovery|utiliz|recycl)\b.{0,30}\b(waste heat|sensible heat)",
        
        # (5) 製程副產燃氣回收與發電 (Converter/Blast Furnace Gas Utilization)
        r"\b(converter gas|blast furnace gas|coke oven gas|top gas|tail gas|"
        r"by[\s-]?product gas|flue gas|off[\s-]?gas)\b.{0,40}"
        r"\b(recover|recovery|recycl|utiliz|reuse|calorific|power generation)",
        
        # (6) 直接減碳/降低焦炭比 (Carbon & Coke Ratio Reduction)
        r"\b(reduce|reducing|reduction of|lower(ing)?)\b.{0,30}"
        r"\b(co2|carbon dioxide|carbon emission|coke (rate|ratio|consumption)|greenhouse)",
        r"\b(emission reduction|carbon emission reduction|greenhouse gas emission)",
        
        # (7) 生物碳/生物質低碳替代還原劑 (Biomass/Biochar Injection)
        r"\b(biomass|biochar)\b.{0,40}"
        r"\b(reduc|reductant|reducing agent|fuel|inject|metallurg|ironmaking|blast furnace)",
        
        # (8) 電弧爐廢鋼預熱與循環利用 (Scrap Recycling)
        r"\bscrap\b.{0,20}\bpreheat",
        
        # (9) 鋼鐵冶金爐渣、粉塵、污泥之循環利用 (Metallurgical Slag & Dust Recycling)
        r"\b(slag|dust|sludge|gas ash|metallurgical solid waste)\b.{0,40}"
        r"\b(recycl|resource utiliz|comprehensive utiliz|reclaim|reuse)",
        
        # (10) 其他低碳高爐或冶金新製程 (如 FINEX, COREX 等)
        r"\b(top gas recycling|oxygen blast furnace|hisarna|finex|corex|carbon recycling)",
    ]
]

# ----------------------------------------------------------------------------
# 2.3 負向排除規則 (NEG_RULES) — 防污染核心規則庫
# 說明：排除冶金學中特有的「同形雙關語」與「非目標領域」，包含：
#      - 鋼水冶煉品質控制中的「脫碳 (Decarburization)」過程（同樣拼為 decarbonization）；
#      - 不鏽鋼、矽鋼等純鋼種品質與合金添加；
#      - 銅、鎳、鉛、鋅等有色金屬冶煉；
#      - 下游的廢棄物處置（危廢焚燒）或土壤/重金屬吸附劑（非鋼鐵製程本體技術）。
# ----------------------------------------------------------------------------
NEG_RULES = [
    re.compile(p, re.IGNORECASE) for p in [
        # (1) 鋼材機械性能與常規性能測試
        r"\b(tensile strength|yield strength|wear resist|hardness|fatigue|"
        r"elongation|toughness|corrosion resist|impact energy)\b",
        
        # (2) 純特鋼、不鏽鋼、矽鋼等特定鋼種之化學配方與成分控制
        r"\b(stainless steel|tool steel|die steel|spring steel|bearing steel|"
        r"pipeline steel|gear steel|mould steel|weathering steel|silicon steel)\b",
        r"\b(nodular cast iron|ductile (cast )?iron|gray cast iron|grey cast iron|vermicular)\b",
        
        # (3) 常規高爐/轉爐爐體機械構造與連鑄設備
        r"\b(taphole|tap hole|tuyere|oxygen lance|cooling wall|furnace mouth|"
        r"sealing device|charging device|slag dart|"
        r"continuous casting (machine|mold|mould)|crystallizer)\b",
        
        # (4) 鋼水常規精煉控制（排除轉爐/精煉爐中鋼液的品質「脫碳」雙關語）
        r"\b(deoxidation|desulfuriz|desulphuriz|dephosphoriz|inclusion (control|removal)|"
        r"argon blowing|refining slag|vacuum refining|rh refining|kr treatment|decarburiz)\b",
        
        # (5) 有色金屬冶煉與礦石（非鐵/非鋼場域）
        r"\b(copper|zinc|lead|nickel|alumina|titanium|magnesium|gold|silver|"
        r"vanadium|rare earth|cobalt|tungsten|laterite nickel|nickel ore)\b",
        
        # (6) 下游環保吸附劑、危險廢物熱裂解（非鋼鐵內部製程低碳化）
        r"\b(hazardous waste|heavy metal adsorption|adsorbent|soil remediation|municipal solid waste)\b",
    ]
]


# ============================================================================
# 3. 輔助處理函數
# ============================================================================
def clean_text(t: str) -> str:
    """
    執行摘要與標題的物理清洗：
    1. 解碼 HTML 轉義字元（如 &amp; 轉為 &）；
    2. 移除殘留的 XML/HTML 標籤；
    3. 壓縮多餘的空白字元。
    """
    if not t:
        return ""
    t = html.unescape(t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def has_field(d: str) -> bool:
    """是否命中鋼鐵業應用場域詞"""
    return FIELD.search(d) is not None


def has_pos(d: str) -> bool:
    """是否命中任何一個強正向低碳技術訊號"""
    return any(p.search(d) for p in POS_STRONG)


def has_neg(d: str) -> bool:
    """是否命中任何一個負向干擾/排除規則"""
    return any(p.search(d) for p in NEG_RULES)


def confidence_tier(prob: float) -> str:
    """根據邏輯迴歸預測機率，進行預測信心的硬性分層"""
    if prob >= 0.80:
        return "A 高信心"
    if prob >= 0.65:
        return "B 中高信心"
    if prob >= 0.50:
        return "C 中信心"
    return "D 強訊號保底"


# ============================================================================
# 4. 可選：基於句向量 (Embedding) 的語意相似度重排序 (Reranking)
# ============================================================================
def load_hf_token(env_path: Path) -> str:
    """自本地 .env 檔案讀取 HuggingFace API Token"""
    token = ""
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_API_TOKEN"):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
    return token


def embedding_scores(texts):
    """
    使用 SentenceTransformer 模型計算專利文本與「鋼鐵碳中和核心定義」錨標句的語意餘弦相似度。
    這可用於輔助邏輯迴歸，提升對邊緣模糊摘要的召回率（Recall）。
    """
    import os
    os.environ.setdefault("HF_TOKEN", load_hf_token(ENV_PATH))
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 載入輕量級高效雙向句向量模型
    model = SentenceTransformer(HF_MODEL)
    
    # 定義代表鋼鐵碳中和四大技術範疇的錨點句子 (Anchor Sentences)
    anchors = [
        "reducing CO2 emission and carbon use in ironmaking and steelmaking",
        "waste heat and by-product gas recovery in blast furnace and converter",
        "steel slag, dust and scrap recycling and resource utilization",
        "hydrogen reduction, direct reduced iron and low-carbon metallurgy, CCUS",
    ]
    # 計算錨點句子的 Embedding
    a = model.encode(anchors, normalize_embeddings=True)
    # 計算輸入專利的 Embedding
    emb = model.encode(list(texts), batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True)
    # 返回專利文本與任一錨標句的最大餘弦相似度
    return cosine_similarity(emb, a).max(axis=1)


# ============================================================================
# 5. 分類管線主程式 (Main Execution Pipeline)
# ============================================================================
def main():
    print(f"讀取:{INPUT_PATH}")
    with open(INPUT_PATH, encoding="utf-8") as f:
        records = json.load(f)
    if isinstance(records, dict):                 # 容錯處理：若外層被包裹為字典
        records = records.get("data", list(records.values()))
    print(f"原始紀錄數:{len(records)}")

    # 進行文本拼接（標題 + 摘要）並保留原始 record 以便後續導出其他 metadata
    docs, keep = [], []
    for r in records:
        abs_ = clean_text(r.get(ABSTRACT_FIELD, "") or "")
        if not abs_:
            continue
        ttl = clean_text(r.get(TITLE_FIELD, "") or "")
        docs.append((ttl + ". " + abs_).strip())
        keep.append(r)
    N = len(docs)
    low = [d.lower() for d in docs]
    print(f"有摘要可分析:{N}")

    # ------------------------------------------------------------------------
    # STAGE 1: 弱監督標籤初始化 (Labeling with Seed Anti-Pollution)
    # ------------------------------------------------------------------------
    y = np.full(N, -1, dtype=int)                 # 初始化標籤數組：-1 代表未標記 (Unlabeled)
    strong = np.zeros(N, dtype=bool)              # 記錄命中強正向規則且未受雜訊干擾的專利
    
    for i, d in enumerate(low):
        fld = has_field(d)
        pos = has_pos(d)
        neg = has_neg(d)
        
        # 強正向種子條件：必須同時命中應用場域 (FIELD) 與低碳特徵 (POS_STRONG)，
        #                且絕對不能含有任何負向排除特徵 (NEG_RULES)，以防止雜訊污染。
        strong[i] = fld and pos and not neg
        
        if fld and pos and not neg:
            y[i] = 1                              # 標記為正例種子
        elif neg:
            y[i] = 0                              # 只要含有排除特徵，即標記為負例種子
            
    print(f"第一層種子 → 正例 {int((y==1).sum())}  負例 {int((y==0).sum())}  未標 {int((y==-1).sum())}")

    # ------------------------------------------------------------------------
    # STAGE 2: TF-IDF 特徵空間構建與邏輯迴歸訓練 (Logistic Regression)
    # ------------------------------------------------------------------------
    # 配置 TF-IDF 向量化器：支援 Unigram 與 Bigram，忽略跨文檔高頻詞與低頻稀疏詞
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_df=0.6,
                          sublinear_tf=True, stop_words="english", max_features=60000)
    X_all = vec.fit_transform(docs)               # 轉換全部文本
    
    # 提取已被自動標記為種子（y=1 或 y=0）的專利子集
    lab = np.where(y != -1)[0]
    X_lab, y_lab = X_all[lab], y[lab]

    # 初始化邏輯迴歸分類器：採用類別權重平衡 (class_weight='balanced') 以處理正負種子極度不平衡問題
    clf = LogisticRegression(max_iter=2000, C=4, class_weight="balanced",
                             random_state=RANDOM_SEED)
    
    # 執行 5 折交叉驗證，評估弱監督訓練集的一致性與模型泛化能力
    try:
        cv = cross_val_score(clf, X_lab, y_lab, cv=5, scoring="f1")
        print(f"種子 5-fold F1:{cv.mean():.3f} ± {cv.std():.3f}")
    except Exception as e:
        print(f"(略過交叉驗證:{e})")

    # 擬合分類器並預測全部摘要的碳中和相關度機率 (Relevance Probability)
    clf.fit(X_lab, y_lab)
    proba = clf.predict_proba(X_all)[:, 1]

    # [可選步驟]：基於 Embedding 語意相似度對機率進行重排序
    if USE_EMBEDDING_RERANK:
        try:
            emb = embedding_scores(docs)
            emb = (emb - emb.min()) / (emb.max() - emb.min() + 1e-9)
            proba = np.maximum(proba, emb)       # 取邏輯迴歸機率與語意相似度的最大值，防範漏失
            print("已套用 embedding 重排序")
        except Exception as e:
            print(f"(embedding 重排序失敗,改用純 TF-IDF 分數:{e})")

    # 收入最終決策：預測機率大於門檻，或者直接命中了強正向弱監督規則（雙重保障）
    relevant = (proba >= THRESHOLD) | strong
    print(f"\n最終相關(門檻 {THRESHOLD} OR 強訊號):{int(relevant.sum())} 篇")

    # [可解釋性分析]：輸出模型學到的特徵單詞權重，檢驗物理合理性
    fn = np.array(vec.get_feature_names_out())
    coef = clf.coef_[0]
    print("正向權重前20:", ", ".join(fn[np.argsort(coef)[-20:]][::-1]))
    print("負向權重前15:", ", ".join(fn[np.argsort(coef)[:15]]))

    # ------------------------------------------------------------------------
    # STAGE 3: 信心分層、排序與結果數據集導出 (CSV & JSON Exporting)
    # ------------------------------------------------------------------------
    tier_counter = Counter()
    rows = []
    # 用於儲存要輸出為 JSON 格式的原始紀錄清單
    extracted_records = []
    
    # 按照相關度機率由高到低，獲取排序索引
    sorted_indices = np.argsort(-proba)
    
    for i in sorted_indices:
        if not relevant[i]:
            continue
        tier = confidence_tier(proba[i])         # 預估機率信心分層
        tier_counter[tier] += 1
        r = keep[i]
        extracted_records.append(r)              # 保留與輸入格式完全相同的原始 JSON 紀錄字典
        
        # 封裝即將輸出的專利對象
        rows.append({
            "publication_number": r.get("publication_number", ""),
            "application_number": r.get("application_number", ""),
            "country_code": r.get("country_code", ""),
            "publication_date": r.get("publication_date", ""),
            "priority_year": r.get("priority_year", ""),
            "relevance_prob": round(float(proba[i]), 3),
            "confidence_tier": tier,
            "matched_cpc_codes": "|".join(r.get("matched_cpc_codes", []) or []),
            "title_en": r.get("title_en", ""),
            "abstract_en": r.get("abstract_en", ""),
        })

    # 定義 CSV 對應的標頭欄位
    fields = ["publication_number", "application_number", "country_code",
              "publication_date", "priority_year", "relevance_prob",
              "confidence_tier", "matched_cpc_codes",
              "title_en", "abstract_en"]
              
    # 將數據寫入 CSV，使用 utf-8-sig 以保證在 Excel 中能正常打開而不出現亂碼
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # 將提取到的專利以完全相同的原始 JSON 格式導出
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(extracted_records, f, ensure_ascii=False, indent=2)

    # 輸出最終分類概覽與統計
    print("\n=== 信心分層 ===")
    for k in ["A 高信心", "B 中高信心", "C 中信心", "D 強訊號保底"]:
        print(f"  {k}: {tier_counter.get(k, 0)}")
    print(f"\n輸出 CSV:{OUTPUT_CSV}  共 {len(rows)} 筆")
    print(f"輸出 JSON:{OUTPUT_JSON}  共 {len(extracted_records)} 筆")


if __name__ == "__main__":
    main()