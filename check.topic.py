import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_parquet("output_specter2_3000/patent_with_topics.parquet")
embeddings = np.load("output_specter2_3000/specter2_embeddings.npy")

topic_id = 0
idx = df.index[df["topic_id"] == topic_id].to_numpy()

topic_emb = embeddings[idx]
centroid = topic_emb.mean(axis=0, keepdims=True)

sims = cosine_similarity(topic_emb, centroid).ravel()

check = df.iloc[idx].copy()
check["topic_centroid_similarity"] = sims

check = check.sort_values("topic_centroid_similarity")

check[[
    "application_number",
    "priority_date",
    "topic_label",
    "topic_centroid_similarity",
    "title_clean",
    "abstract_clean"
]].head(50).to_csv(
    "output_specter2_3000/topic_0_low_similarity_check.csv",
    index=False,
    encoding="utf-8-sig"
)
