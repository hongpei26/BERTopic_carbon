# 碳捕捉與鋼鐵業減碳專利分析 Pipeline

本目錄 (`backup_python`) 包含一系列用於處理、清洗、篩選及分析「鋼鐵業碳中和/減碳」相關專利數據的 Python 腳本。整個資料科學流程從原始 Parquet 專利資料的去重開始，經過 CPC 分類碼篩選、機器學習文本分類萃取，最終進入進階的 BERTopic 動態主題建模。

## 執行流程 (Pipeline)

此資料處理流程可分為以下四大階段，各腳本需依序執行以產出最終結果：

### 階段一：資料去重 (Deduplication)
這三個腳本負責將海量專利資料去蕪存菁，確保同一技術發明不會因為跨國申請或多重公開而重複出現，避免後續主題建模時的權重偏差。

1. **`dedup_application_number_more.py`**
   - **功能**：依據 `application_number`（申請號）對多個來源 Parquet 檔案進行合併與去重。若有多筆相同申請號，優先保留日期較早者。
   - **輸出**：`global_application_dedup.json`
2. **`dedup_family_id.py`**
   - **功能**：依據 `family_id`（專利族）進行跨國同族去重。優先保留英文摘要品質較佳（字數多）及特定優先國家（如 US, WO, EP）的代表案。
   - **輸出**：`global_family_dedup.json` (及審計日誌 `global_family_dedup_audit.json`)
3. **`dedup_abs.py`**
   - **功能**：針對 `abstract_en`（英文摘要）進行最終的文本精確去重，處理不同專利族但摘要內容完全相同的極端情況。
   - **輸出**：`global_abstract_dedup.json`

### 階段二：領域與目標過濾 (Filtering)
4. **`filter_domain_target_cpc_nokeyword.py`**
   - **功能**：基於去重後的資料，進行**時間區間** (2006-2025) 與 **CPC 專利分類碼**過濾。專利必須同時符合「鋼鐵領域 (C21B/C21C)」且具備「減碳/碳中和目標 (如 B01D, Y02 系列)」的 CPC 碼才會被保留。
   - **輸出**：`global_onlycpc_domain_target_intersection.json`

### 階段三：機器學習關鍵字萃取 (Machine Learning Classification)
5. **`keyword_v3.py`**
   - **功能**：採用「規則弱監督 (Weak Supervision) + 邏輯迴歸 (Logistic Regression)」雙層管線。先利用設定好的強正向（減碳）與負向（雜訊如脫碳雙關語、有色金屬）規則標註種子，再以 TF-IDF 特徵訓練機器學習模型，精準萃取出鋼鐵業碳中和專利，並依據機率進行預測信心分層 (A/B/C/D 級)。
   - **輸出**：`steel_carbonneutral_extracted.csv` 及同名的 JSON 檔案。

### 階段四：動態主題建模 (Topic Modeling)
6. **`BERTopic_specter2.py`** (及備份版 `BERTopic_specter2 copy.py`)
   - **功能**：對最終萃取出的專利文本進行 BERTopic 動態主題建模，並分析主題隨時間的演進 (Topics over Time)。
   - **流程包含 7 個 Stage**：物理清洗文本 $\rightarrow$ SPECTER2 科學語意嵌入向量 (Embedding) $\rightarrow$ UMAP 降維 $\rightarrow$ HDBSCAN 密度聚類 $\rightarrow$ c-TF-IDF 特徵提取 $\rightarrow$ KeyBERT + MMR + LLM (OpenAI GPT-4o-mini) 生成英文主題標籤 $\rightarrow$ 動態趨勢分析。
   - **特色**：包含 Embedding 矩陣快取機制 (.npy 存檔) 以加速重複實驗，並設計對齊語意理解的停用詞與小寫化策略。
   - **輸出**：主題關鍵字、專利主題分群結果 (`patent_with_topics.parquet`)、生命週期演進軌跡報表 (`topics_over_time.csv` 等) 及儲存的 BERTopic 模型。

## 目錄與環境需求

### 套件安裝

```bash
pip install -r backup_python/requirements.txt
```

- **輸入/輸出資料夾配置**：腳本內主要預設讀寫資料路徑為 `/home/carbon/carbon/data_global_v2/Carbon_onlycpc_global_morecpc_v2/`，而建模輸出將存於根目錄的 `output_specter2_weighted_keywordtraining/`。
- **主要依賴套件**：
  - 資料處理：`pandas`, `numpy`, `pyarrow`
  - 機器學習與 NLP：`scikit-learn`, `bertopic`, `sentence-transformers`, `umap-learn`, `hdbscan`
  - 模型微調與 LLM：`adapters`, `openai`
  - 環境變數：`python-dotenv` (需準備 `.env` 檔案並填寫 `HF_TOKEN` 與 `OPENAI_API_KEY` 以利 HuggingFace 模型下載與 GPT-4 標籤生成)

---
*註：執行腳本前請確認檔案路徑與硬碟空間充足，部分機器學習與深度學習模型 (如 SPECTER2) 支援並建議於 GPU (CUDA) 環境下執行以大幅縮短運算時間。*
