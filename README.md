# 碳捕捉與鋼鐵業減碳專利分析 Pipeline (`berttopic_carbon`)

## 📌 專案簡介 (Overview)

本專案建立了一套高精度、端到端的巨量專利資料處理與 AI 分析管線（Pipeline），專注於「**鋼鐵產業碳中和與減碳技術**」領域。整個資料科學流程從全球專利資料庫的跨國去重開始，經過 CPC 分類碼雙群組交集篩選、弱監督機器學習 (Logistic Regression) 特徵萃取，最終進入以 SPECTER2 + BERTopic 為核心的動態主題建模與趨勢分析 (Topics over Time)，全面解析 2006 至 2025 年間全球鋼鐵減碳技術的演進軌跡與發展趨勢。

---

## 🌟 核心特色 (Core Features)

1. **多重階層式專利去重機制 (Multi-Level Deduplication)**
   - 整合申請號 (`application_number`)、跨國同族專利 (`family_id`) 與英文摘要文本 (`abstract_en`) 三階段去重，結合國家優先權偏好 (US/WO/EP/CA 等) 與摘要品質評分，避免同一技術重複採樣造成的權重偏差與頻率失真。
2. **CPC 雙群組強交集篩選 (Dual-Group CPC Intersection)**
   - 精確結合「鋼鐵場域 CPC (`C21B`/`C21C`)」與「減碳目標 CPC (如 `B01D53` 氣體分離、`Y02C`/`Y02P` 溫室氣體減排系列)」，並強制進行 2006–2025 時間維度過濾，確保分析語料兼具產業專一性與技術相關性。
3. **防污染弱監督機器學習分類器 (Anti-Pollution Weak Supervision ML)**
   - 結合鋼鐵場域詞與強正向減碳訊號，並導入負向干擾規則（嚴格排除鋼水精煉「脫碳雙關語 (Decarburization)」、合金鋼成分、有色金屬冶煉與下游無關廢棄物處置），訓練具備類別平衡 (Class Balanced) 的 TF-IDF 邏輯迴歸模型，並進行 A/B/C/D 信心分層與保底機制。
4. **SPECTER2 科學語意嵌入與快取 (SPECTER2 Semantic Embedding with Caching)**
   - 前處理採用「純物理清洗」，完整保留大小寫、標點與句型結構供 Transformer 模型的 [CLS] 向量理解科學語意；內建 `.npy` 矩陣快取機制，大幅提升重複實驗與參數調校的執行速度。
5. **多層級主題表示法與 LLM 自動標籤 (Multi-Stage Representation & LLM Labeling)**
   - 結合 `KeyBERTInspired` 語意選詞、`MaximalMarginalRelevance (MMR)` 多樣性去重，並可掛載 OpenAI GPT-4o-mini，根據代表摘要生成符合冶金與減碳專業領域的英文主題標籤 (`English_Label`)。
6. **動態生命週期與趨勢演進分析 (Topics over Time & Trajectory Tracking)**
   - 自動將資料劃分為 4 個五年間隔 (2006–2010, 2011–2015, 2016–2020, 2021–2025)，計算主題成長率 (Growth Rate)、市場占比斜率與趨勢分類（如「新興上升」、「相對穩定」、「早期高峰後衰退」），提供深刻的技術前瞻洞察。

---


## 📁 目錄結構 (Directory Structure)

`berttopic_carbon/`
├── `dedup_application_number_more.py`    # 階段一 (1.1)：依申請號 (application_number) 去重
├── `dedup_family_id.py`                 # 階段一 (1.2)：依專利族 (family_id) 跨國同族去重
├── `dedup_abs.py`                       # 階段一 (1.3)：依英文摘要 (abstract_en) 文本去重
├── `filter_domain_target_cpc_nokeyword.py` # 階段二：CPC 鋼鐵場域 & 減碳目標雙群組交集與時間 (2006-2025) 篩選
├── `keyword_v3.py`                      # 階段三：雙層弱監督機器學習 (Rules + TF-IDF + Logistic Regression) 關鍵字萃取
├── `BERTopic_specter2.py`              # 階段四：7-Stage SPECTER2 + BERTopic 動態主題建模與趨勢演進分析
└── `requirements.txt`                   # 專案依賴套件清單

---

## 🔄 執行流程 (Pipeline Workflow)

完整資料處理與建模流程分為四大階段，各腳本需依序執行：

### 階段一：海量專利去重 (Deduplication)
確保同一技術發明不會因為跨國申請、多重公開或版本重複而造成後續主題建模的權重與頻率偏差。

1. **`dedup_application_number_more.py`**
   - **功能**：讀取來源 Parquet 專利資料，依據 `application_number`（申請號）進行去重。若出現同申請號多筆資料，優先保留 `priority_date` $\rightarrow$ `filing_date` $\rightarrow$ `publication_date` 日期最早者。
   - **輸出**：`global_application_dedup.json`

2. **`dedup_family_id.py`**
   - **功能**：讀取 `global_application_dedup.json`，依據 `family_id`（專利族）進行跨國同族公開案去重。優先保留英文摘要品質較佳（詞數較多）及特定優先國別代表案（排序：US $\rightarrow$ WO $\rightarrow$ EP $\rightarrow$ CA $\rightarrow$ AU $\rightarrow$ GB $\rightarrow$ CN $\rightarrow$ JP $\rightarrow$ KR ...）。
   - **輸出**：`global_family_dedup.json` 與審計日誌 `global_family_dedup_audit.json`

3. **`dedup_abs.py`**
   - **功能**：針對 `global_family_dedup.json` 進行英文摘要 (`abstract_en`) 精確字面去重，排除空摘要及不同專利族但摘要內容完全相同的極端重複案，保留最早日期紀錄。
   - **輸出**：`global_abstract_dedup.json`

---

### 階段二：CPC 領域與目標過濾 (Filtering)
4. **`filter_domain_target_cpc_nokeyword.py`**
   - **功能**：
     - **時間過濾**：優先保留 `priority_date` 年份落在 **2006–2025** 年之間的專利。
     - **CPC 雙群組交集**：要求專利必須同時命中：
       1. **鋼鐵領域 CPC (Domain)**：`C21B` (高爐/鐵製造) 或 `C21C` (煉鋼精煉)。
       2. **減碳/碳中和目標 CPC (Target)**：`B01D53` (氣體分離/CO2捕集)、`Y02C20/` (溫室氣體處置)、`Y02P10/` (金屬加工減碳)、`Y02P20/` (化工減碳)、`Y02P40/` (礦物加工減碳)、`Y02P70/` (產品減碳)、`Y02P80/` (資源循環/能效)、`Y02P90/` (智慧製造/氫能)。
   - **輸出**：`global_onlycpc_domain_target_intersection.json`

---

### 階段三：弱監督機器學習文本萃取 (Weak Supervision & ML Classification)
5. **`keyword_v3.py`**
   - **功能**：採用「規則弱監督 (Rule-based Weak Supervision) + 統計機器學習 (Logistic Regression)」雙層管線：
     - **第一層 (弱監督標註)**：結合鋼鐵場域詞 (`FIELD`) 與強正向低碳訊號 (`POS_STRONG`) 標註正例種子 ($y=1$)；同時利用負向排除規則 (`NEG_RULES`) 嚴格排除鋼水精煉脫碳雙關語 (decarburization)、合金特鋼成分、有色金屬 (銅/鎳/鋁) 與下游廢棄物處理等強負例種子 ($y=0$)。防污染機制確保訓練資料品質。
     - **第二層 (TF-IDF + 邏輯迴歸)**：將標題與摘要轉為 Unigram/Bigram TF-IDF 矩陣，訓練帶有類別權重平衡 (`class_weight='balanced'`) 的 `LogisticRegression` 模型，計算相關度機率 `relevance_prob`。
     - **第三層 (信心分層與導出)**：依據預測機率及強訊號進行 A (高信心 $\ge 0.80$)、B (中高信心 $\ge 0.65$)、C (中信心 $\ge 0.50$)、D (強訊號保底) 分層。
   - **輸出**：`steel_carbonneutral_extracted.csv` 與 `steel_carbonneutral_extracted.json`

---

### 階段四：7-Stage 動態主題建模 (Dynamic Topic Modeling)
6. **`BERTopic_specter2.py`**
   - **功能**：對純化後的專利文本進行 BERTopic 動態主題建模，並分析主題隨時間的演進軌跡 (Topics over Time)。
   - **7 個 Stage 詳細流程**：
     - **Stage 1 (資料載入與物理清洗)**：讀取 `steel_carbonneutral_extracted.json`，僅做 HTML 解碼、標籤與多餘空白清除，**保留大小寫、標點符號與完整句型**，供 Transformer 語意模型完整理解。過濾摘要詞數 $< 30$ 者，劃分 4 個時間區段 (2006–2010, 2011–2015, 2016–2020, 2021–2025)。
     - **Stage 2 (SPECTER2 語意嵌入與快取)**：採用 `allenai/specter2_base` + `allenai/specter2` adapter 生成科學語意的 `[CLS]` 向量，支援 `.npy` 快取檔 (`specter2_embeddings.npy`) 以加速重複實驗。
     - **Stage 3 (UMAP 降維)**：`n_neighbors=20, n_components=5, metric='cosine'`。
     - **Stage 4 (HDBSCAN 聚類)**：`min_cluster_size=10, min_samples=4, cluster_selection_method='eom'`。
     - **Stage 5 (c-TF-IDF)**：採用 `CountVectorizer` (Unigram/Bigram/Trigram)，真正套用小寫化與特定專利/語料停用詞過濾 (`CUSTOM_STOPWORDS`)，僅作用於主題關鍵字展示。
     - **Stage 6 (多層級表示法微調與模型儲存)**：`KeyBERTInspired` $\rightarrow$ `MaximalMarginalRelevance (MMR)` $\rightarrow$ OpenAI GPT-4o-mini 生成英文主題標籤 (`English_Label`)，輸出 `patent_with_topics.parquet`、`topic_keywords.csv` 及 BERTopic safetensors 模型。
     - **Stage 7 (Topics over Time 動態分析)**：分析主題在四個時間區段的頻率變化與占比成長率 (Growth Rate)，劃分「新興上升」、「相對穩定」、「早期高峰後衰退」等演進軌跡。
   - **輸出**：預設存於 `output_specter2_weighted_keywordtraining/` 目錄：
     - `specter2_embeddings.npy` (嵌入快取)
     - `patent_with_topics.parquet` (主題標記專利資料)
     - `topic_keywords.csv` (主題 Top10 關鍵字與 GPT 英文標籤)
     - `topics_over_time.csv` (時間維度主題頻率)
     - `topic_lifecycle.csv` (主題生命周期狀態與成長率)
     - `topic_evolution_summary.csv` (主題演進軌跡摘要與斜率)
     - `topic_yearly_counts.csv` / `topic_yearly_shares.csv`
     - `bertopic_model/` (已儲存的模型)

---

## 🛠️ 目錄與環境需求 (Environment Setup)

### 套件安裝

```bash
pip install -r berttopic_carbon/requirements.txt
```

### 必備環境變數 (`.env`)
在專案根目錄 `/home/carbon/carbon/` 配置 `.env` 檔案：

```env
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_api_key
```
- `HF_TOKEN`：用於下載 Hugging Face `allenai/specter2_base` 模型與 Adapter。
- `OPENAI_API_KEY`：用於 BERTopic Stage 6 GPT-4o-mini 生成主題英文標籤 (選用)。

### 硬體建議
- 建議於 GPU (CUDA) 環境下執行 `BERTopic_specter2.py`，以大幅縮短 SPECTER2 向量嵌入與降維聚類等運算時間。

