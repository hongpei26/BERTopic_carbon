import json
import re
import sys
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

file_path = "/home/carbon/carbon/data/part-000000000000_dedup_by_abstract.json"

print(f"Reading {file_path}...", flush=True)
try:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception as e:
    print(f"Error reading file {file_path}: {e}")
    sys.exit(1)

texts = []
for item in data:
    title = item.get("title_en", "") or item.get("title", "")
    abstract = item.get("abstract_en", "") or item.get("abstract", "")
    texts.append((str(title) + " " + str(abstract)).lower())

print(f"總專利篇數 (Total documents): {len(texts)}", flush=True)

# 1. 預定義碳中和相關技術詞彙的出現頻率 
target_keywords = {
    "直接還原 / 海綿鐵": ["direct reduction", "dri", "sponge iron", "hbi", "midrex", "energiron"],
    "電弧爐 / 廢鋼": ["electric arc", "eaf", "scrap", "recycling"],
    "氫冶金": ["hydrogen", "h2 "],
    "碳捕捉與封存": ["carbon capture", "co2 capture", "sequestration", "ccus", "carbon dioxide", "co2"],
    "爐渣與廢料再利用": ["slag", "waste heat", "heat recovery", "coproduct", "by-product", "red mud", "zinc dust"],
    "能源技術": ["fuel cell", "electrolysis", "renewable", "biomass", "charcoal"],
    "減碳目標用詞": ["emission", "carbon footprint", "greenhouse", "low carbon", "decarbonization", "carbon neutral"]
}

print("\n=== 預定義碳中和領域技術關鍵字 (出現在多少篇專利中) ===", flush=True)
for category, kws in target_keywords.items():
    print(f"\n【{category}】:", flush=True)
    for kw in kws:
        count = sum(1 for text in texts if re.search(r'\b' + re.escape(kw) + r'\b', text))
        if count > 0:
            print(f"  - {kw:<16}: {count:>5} 篇", flush=True)

# 2. 自動抽取 Top Bigrams & Trigrams 
print("\n=== 自動萃取最常出現的技術字串 (Top 30 Bigrams/Trigrams) ===", flush=True)
try:
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(2, 3), min_df=20, max_df=0.2)
    X = vectorizer.fit_transform(texts)
    suma = X.sum(axis=0)
    words_freq = [(word, suma[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    ignore_list = ["present invention", "relates to", "method for", "apparatus for", "weight percent", "utility model", "invention relates", "provided is", "providing a", "step of"]
    count = 0
    for word, freq in words_freq:
        if any(ig in word.lower() for ig in ignore_list):
            continue
        print(f"{word:<25}: {freq:>5} 次", flush=True)
        count += 1
        if count >= 30:
            break
except Exception as e:
    print(f"Error extracting ngrams: {e}")
