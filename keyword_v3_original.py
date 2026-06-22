#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鋼鐵業碳中和摘要抽取 — 三層管線
  第一層:關鍵字弱監督  → 造訓練標籤(正例需「場域詞 + 正例訊號」;負例需「雜訊且無正例訊號」)
  第二層:TF-IDF + 邏輯迴歸 → 對全部摘要打相關機率,機率 >= THRESHOLD(或命中強訊號)即收入
  第三層:關鍵字類別標註 → 對被收入的摘要貼八大技術類別(事後標記,不影響收不收)

不需要 HuggingFace。HF token 僅在檔尾「可選 embedding 重排序」開啟時才使用。
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

# ----------------------------------------------------------------------------
# 路徑與參數
# ----------------------------------------------------------------------------
INPUT_PATH = Path(
    "/home/carbon/carbon/data_global_v2/Carbon_onlycpc_global_morecpc_v2/"
    "global_onlycpc_domain_target_intersection.json"
)
OUTPUT_DIR = INPUT_PATH.parent
OUTPUT_CSV = OUTPUT_DIR / "steel_carbonneutral_extracted.csv"

ABSTRACT_FIELD = "abstract_en"   # 主要判定欄位
TITLE_FIELD = "title_en"         # 標題會併入文本一起判定
THRESHOLD = 0.50                 # 第二層收入門檻
RANDOM_SEED = 42

# 可選:embedding 重排序(預設關閉,需要 HF token 與網路時才設 True)
USE_EMBEDDING_RERANK = False
ENV_PATH = Path("/home/carbon/carbon/.env")
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ----------------------------------------------------------------------------
# 關鍵字規則
# ----------------------------------------------------------------------------
# 場域詞:正例必須同時命中其一
FIELD = re.compile(
    r"\b(blast furnace|converter|basic oxygen furnace|electric arc furnace|eaf|"
    r"steelmaking|ironmaking|molten iron|molten steel|direct reduced iron|dri|"
    r"sponge iron|shaft furnace|coke oven|steel slag|steelworks|steel mill|sinter|"
    r"pellet|ferrous metallurg|iron and steel|metallurgical)\b",
    re.IGNORECASE,
)

# 正例訊號(高精準度;部分以 .{0,N} 表示「鄰近出現」,N 即字元距離上限)
POS_STRONG = [
    re.compile(p, re.IGNORECASE) for p in [
        # 碳捕捉類
        r"\b(co2 capture|carbon capture|ccus|\bccs\b|carbon sequestration|"
        r"carbon neutral|carbon-neutral|green steel|decarboni[sz])",
        # 氫冶金類
        r"\b(hydrogen reduction|hydrogen metallurg|hydrogen-rich|hydrogen-based|"
        r"hydrogen rich|molten oxide electrolysis|h2 reduction)",
        # 低碳宣告
        r"\b(near[\s-]?zero emission|zero[\s-]?carbon|ultra[\s-]?low emission|"
        r"low[\s-]carbon (metallurg|process|production|technolog|develop|"
        r"ironmaking|steelmaking))",
        # 餘熱:waste/sensible heat 與 recover/utilize/power 在 ~40 字內
        r"\b(waste heat|sensible heat)\b.{0,40}\b(recover|recovery|utiliz|generation|power)",
        r"\b(recover|recovery|utiliz|recycl)\b.{0,30}\b(waste heat|sensible heat)",
        # 副產氣:各種爐氣 與 回收/利用/熱值 在 ~40 字內
        r"\b(converter gas|blast furnace gas|coke oven gas|top gas|tail gas|"
        r"by[\s-]?product gas|flue gas|off[\s-]?gas)\b.{0,40}"
        r"\b(recover|recovery|recycl|utiliz|reuse|calorific|power generation)",
        # 直接減碳:reduce/lower 與 CO2/碳排/焦比 在 ~30 字內;或直接片語
        r"\b(reduce|reducing|reduction of|lower(ing)?)\b.{0,30}"
        r"\b(co2|carbon dioxide|carbon emission|coke (rate|ratio|consumption)|greenhouse)",
        r"\b(emission reduction|carbon emission reduction|greenhouse gas emission)",
        # 低碳還原劑:biomass/biochar 與 還原/燃料/噴吹 在 ~40 字內
        r"\b(biomass|biochar)\b.{0,40}"
        r"\b(reduc|reductant|reducing agent|fuel|inject|metallurg|ironmaking|blast furnace)",
        # 廢鋼:scrap 與 preheat 在 ~20 字內
        r"\b(scrap)\b.{0,20}\b(preheat)",
        # 爐渣循環:slag/dust/sludge 與 回收/資源化 在 ~40 字內
        r"\b(slag|dust|sludge|gas ash|metallurgical solid waste)\b.{0,40}"
        r"\b(recycl|resource utiliz|comprehensive utiliz|reclaim|reuse)",
        # 其他低碳冶金路線
        r"\b(top gas recycling|oxygen blast furnace|hisarna|finex|corex|carbon recycling)",
    ]
]

# 負例訊號(無任何正例訊號時才當負例)
NEG_RULES = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(tensile strength|yield strength|wear resist|hardness|fatigue|"
        r"elongation|toughness|corrosion resist|impact energy)\b",
        r"\b(stainless steel|tool steel|die steel|spring steel|bearing steel|"
        r"pipeline steel|gear steel|mould steel|weathering steel|silicon steel)\b",
        r"\b(nodular cast iron|ductile (cast )?iron|gray cast iron|grey cast iron|vermicular)\b",
        r"\b(taphole|tap hole|tuyere|oxygen lance|cooling wall|furnace mouth|"
        r"sealing device|charging device|slag dart|"
        r"continuous casting (machine|mold|mould)|crystallizer)\b",
        r"\b(deoxidation|desulfuriz|desulphuriz|dephosphoriz|inclusion (control|removal)|"
        r"argon blowing|refining slag)\b",
        r"\b(copper|zinc|lead|nickel|alumina|titanium|magnesium|gold|silver|"
        r"vanadium|rare earth|cobalt|tungsten)\b",
    ]
]


def clean_text(t: str) -> str:
    """去 HTML 實體與標籤、壓縮空白。"""
    if not t:
        return ""
    t = html.unescape(t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def has_field(d: str) -> bool:
    return FIELD.search(d) is not None


def has_pos(d: str) -> bool:
    return any(p.search(d) for p in POS_STRONG)


def has_neg(d: str) -> bool:
    return any(p.search(d) for p in NEG_RULES)


# ----------------------------------------------------------------------------
# 第三層:八大技術類別標註
# ----------------------------------------------------------------------------
def tag_categories(d: str):
    H = lambda pat: re.search(pat, d, re.IGNORECASE) is not None
    cats = []
    if H(r"\b(co2 capture|carbon capture|ccus|\bccs\b|carbon sequestration|"
          r"carbon neutral|green steel|carbon (utiliz|fixation|mineraliz))"):
        cats.append("CCUS/碳捕捉")
    if H(r"\b(waste heat|sensible heat|heat recovery|power generation|energy[\s-]?saving)") \
       or (H(r"\bheat\b") and H(r"recover")):
        cats.append("餘熱/能效")
    if H(r"\b(converter gas|blast furnace gas|coke oven gas|top gas|tail gas|"
          r"flue gas|off[\s-]?gas|by[\s-]?product gas)\b") \
       and H(r"(recover|recovery|recycl|utiliz|reuse|calorific|purif)"):
        cats.append("副產氣回收")
    if H(r"\b(slag|dust|sludge|gas ash|solid waste|tailing)\b") \
       and H(r"(recycl|reuse|reclaim|resource utiliz|recover)"):
        cats.append("爐渣/粉塵循環")
    if H(r"\bscrap\b") and H(r"(preheat|recycl|reuse|ratio|charg)"):
        cats.append("廢鋼/電爐")
    if H(r"\b(hydrogen reduc|hydrogen[\s-]rich|hydrogen metallurg|hydrogen[\s-]based|"
          r"biochar|biomass|molten oxide electrolysis)"):
        cats.append("氫冶金/低碳還原")
    if H(r"\b(direct[\s-]*reduc|\bdri\b|sponge iron|gas[\s-]based shaft)"):
        cats.append("DRI/直接還原")
    if H(r"\b(energy (monitor|consumption|management)|process optimiz|"
          r"intelligent control|reduce (energy|consumption)|heat loss)"):
        cats.append("智能/製程優化")
    if H(r"\b(reduce co2|co2 emission|carbon emission|emission reduction|reduce emission|"
          r"decarboni[sz]|reduce coke|coke (rate|ratio)|greenhouse)"):
        cats.append("直接減碳")
    return cats or ["其他(語意判定)"]


def confidence_tier(prob: float) -> str:
    if prob >= 0.80:
        return "A 高信心"
    if prob >= 0.65:
        return "B 中高信心"
    if prob >= 0.50:
        return "C 中信心"
    return "D 強訊號保底"


# ----------------------------------------------------------------------------
# 可選:embedding 重排序(預設關閉)
# ----------------------------------------------------------------------------
def load_hf_token(env_path: Path) -> str:
    """從 .env 讀 HF_API_TOKEN。"""
    token = ""
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_API_TOKEN"):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
    return token


def embedding_scores(texts):
    """以句向量對「鋼鐵碳中和」定義錨句做餘弦相似度。需 sentence-transformers 與網路。"""
    import os
    os.environ.setdefault("HF_TOKEN", load_hf_token(ENV_PATH))
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    model = SentenceTransformer(HF_MODEL)
    anchors = [
        "reducing CO2 emission and carbon use in ironmaking and steelmaking",
        "waste heat and by-product gas recovery in blast furnace and converter",
        "steel slag, dust and scrap recycling and resource utilization",
        "hydrogen reduction, direct reduced iron and low-carbon metallurgy, CCUS",
    ]
    a = model.encode(anchors, normalize_embeddings=True)
    emb = model.encode(list(texts), batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True)
    return cosine_similarity(emb, a).max(axis=1)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    print(f"讀取:{INPUT_PATH}")
    with open(INPUT_PATH, encoding="utf-8") as f:
        records = json.load(f)
    if isinstance(records, dict):                 # 容錯:若外層是 dict
        records = records.get("data", list(records.values()))
    print(f"原始紀錄數:{len(records)}")

    # 文本 = 標題 + 摘要(清理後);保留原始 record 以便輸出其他欄位
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

    # ---------- 第一層:弱監督標籤 ----------
    y = np.full(N, -1, dtype=int)
    strong = np.zeros(N, dtype=bool)
    for i, d in enumerate(low):
        fld = has_field(d)
        pos = has_pos(d)
        strong[i] = fld and pos
        if fld and pos:
            y[i] = 1
        elif (not pos) and has_neg(d):
            y[i] = 0
    print(f"第一層種子 → 正例 {int((y==1).sum())}  負例 {int((y==0).sum())}  未標 {int((y==-1).sum())}")

    # ---------- 第二層:TF-IDF + 邏輯迴歸 ----------
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_df=0.6,
                          sublinear_tf=True, stop_words="english", max_features=60000)
    X_all = vec.fit_transform(docs)
    lab = np.where(y != -1)[0]
    X_lab, y_lab = X_all[lab], y[lab]

    clf = LogisticRegression(max_iter=2000, C=4, class_weight="balanced",
                             random_state=RANDOM_SEED)
    try:
        cv = cross_val_score(clf, X_lab, y_lab, cv=5, scoring="f1")
        print(f"種子 5-fold F1:{cv.mean():.3f} ± {cv.std():.3f}")
    except Exception as e:
        print(f"(略過交叉驗證:{e})")

    clf.fit(X_lab, y_lab)
    proba = clf.predict_proba(X_all)[:, 1]

    # 可選 embedding 重排序:取模型分數與 embedding 相似度的較大值(提升召回)
    if USE_EMBEDDING_RERANK:
        try:
            emb = embedding_scores(docs)
            emb = (emb - emb.min()) / (emb.max() - emb.min() + 1e-9)
            proba = np.maximum(proba, emb)
            print("已套用 embedding 重排序")
        except Exception as e:
            print(f"(embedding 重排序失敗,改用純 TF-IDF 分數:{e})")

    # 收入決策:機率 >= 門檻 或 命中強訊號(保底)
    relevant = (proba >= THRESHOLD) | strong
    print(f"\n最終相關(門檻 {THRESHOLD} OR 強訊號):{int(relevant.sum())} 篇")

    # 模型倚重詞(可解釋性)
    fn = np.array(vec.get_feature_names_out())
    coef = clf.coef_[0]
    print("正向權重前20:", ", ".join(fn[np.argsort(coef)[-20:]][::-1]))
    print("負向權重前15:", ", ".join(fn[np.argsort(coef)[:15]]))

    # ---------- 第三層:類別標註 + 輸出 ----------
    cat_counter, tier_counter = Counter(), Counter()
    rows = []
    for i in range(N):
        if not relevant[i]:
            continue
        cats = tag_categories(low[i])
        tier = confidence_tier(proba[i])
        for c in cats:
            cat_counter[c] += 1
        tier_counter[tier] += 1
        r = keep[i]
        rows.append({
            "publication_number": r.get("publication_number", ""),
            "application_number": r.get("application_number", ""),
            "country_code": r.get("country_code", ""),
            "publication_date": r.get("publication_date", ""),
            "priority_year": r.get("priority_year", ""),
            "relevance_prob": round(float(proba[i]), 3),
            "confidence_tier": tier,
            "tech_categories": "|".join(cats),
            "matched_cpc_codes": "|".join(r.get("matched_cpc_codes", []) or []),
            "title_en": r.get("title_en", ""),
            "abstract_en": r.get("abstract_en", ""),
        })

    rows.sort(key=lambda x: -x["relevance_prob"])
    fields = ["publication_number", "application_number", "country_code",
              "publication_date", "priority_year", "relevance_prob",
              "confidence_tier", "tech_categories", "matched_cpc_codes",
              "title_en", "abstract_en"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\n=== 信心分層 ===")
    for k in ["A 高信心", "B 中高信心", "C 中信心", "D 強訊號保底"]:
        print(f"  {k}: {tier_counter.get(k, 0)}")
    print("\n=== 八大技術類別(可重複計)===")
    for k, v in cat_counter.most_common():
        print(f"  {k}: {v}")
    print(f"\n輸出:{OUTPUT_CSV}  共 {len(rows)} 筆")


if __name__ == "__main__":
    main()