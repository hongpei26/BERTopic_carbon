import pandas as pd
from bertopic import BERTopic

print("正在載入原始文獻與主題模型，請稍候...")

# 1. 載入您之前產出的 Parquet 檔案（裡面有清洗好的摘要）
df = pd.read_parquet("output_specter2_3000/patent_with_topics.parquet")
docs = df["abstract_clean"].astype(str).tolist()

# 2. 載入已經訓練好的 BERTopic 模型
# 因為只是要計算階層與畫圖，不需要載入笨重的 Embedding 模型
try:
    topic_model = BERTopic.load("output_specter2_3000/bertopic_model", embedding_model=None)
except Exception as e:
    print(f"模型載入失敗，請確認路徑或儲存格式: {e}")
    exit()

print("正在計算階層式主題結構 (Hierarchical Topics)...")
# 3. 呼叫 BERTopic 內建的階層計算功能
# 這會根據主題之間的語意距離，由下而上(Bottom-up)把相近的主題合併起來
hierarchical_topics = topic_model.hierarchical_topics(docs)

print("正在生成樹狀圖 (Dendrogram)...")
# 4. 將階層關係畫成樹狀圖
fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

# 5. 儲存為 HTML 互動式網頁檔案
OUTPUT_HTML = "output_specter2_3000/topic_hierarchy_dendrogram.html"
fig.write_html(OUTPUT_HTML)

print(f"\n✅ 成功！樹狀圖已儲存至: {OUTPUT_HTML}")
print("這是一個互動式的網頁檔案，您可以：")
print("1. 進入資料夾，雙擊點開這個 HTML 檔案。")
print("2. 你的瀏覽器會打開它，您可以滑鼠游標停留在節點上看它怎麼聚合。")
print("3. 可以將它截圖放進論文，作為『演算法自動分群』的學術證據！")
