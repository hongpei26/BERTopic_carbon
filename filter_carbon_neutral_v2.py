import json
import re
from pathlib import Path
from collections import Counter

# 設定輸入與輸出路徑
src = Path("/home/carbon/carbon/data_globalmorecpc/global_onlycpc_domain_target_intersection.json")
out = Path("/home/carbon/carbon/data_globalmorecpc/global_onlycpc_carbon_neutral_v2.json")


# 彈性檢索 helper:
# - 詞形彈性: recover / recovery / recovered / recovering 等。
# - 片語彈性: waste heat recovery / recovery of waste heat 等。
RECOVER_FORMS = r"(?:recover(?:s|ed|ing)?|recovery|recoveries)"
RECYCLE_FORMS = r"(?:recycl(?:e|es|ed|ing)|recycling)"
WORD_GAP = r"(?:[-\s]+\w+){0,3}[-\s]+"
OF_FROM_GAP = r"(?:[-\s]+(?:of|from|the|a|an)){0,3}[-\s]+"


def phrase_pattern(*words):
    """建立可接受空白或 hyphen 的片語 regex。"""
    return r"[-\s]+".join(re.escape(word) for word in words)


def action_phrase_pattern(object_pattern, action_pattern):
    """建立「object + action」與「action + of/from + object」雙向片語 regex。"""
    return (
        rf"(?:{object_pattern}{WORD_GAP}{action_pattern}"
        rf"|{action_pattern}{OF_FROM_GAP}{object_pattern})"
    )


# 1. 負向排除關鍵字 (Negative Filters) - 僅精準排除下游產品材料，不誤傷製程回收
EXCLUSION_TERMS = [
    r"low[-\s]carbon steel",
    r"low[-\s]carbon equivalent",
    r"bearing steel",
    r"marine steel",
    r"high[-\s]strength steel",
    r"spring steel",
    r"gear steel",
    r"free[-\s]cutting steel",
    r"stainless steel (?:sheet|plate|material|composition|wire|bar|product)",
    r"duplex stainless steel material",
    r"austenitic stainless steel material",
    r"ferritic stainless steel material",
]

# 2. 高度相關關鍵字 (High Relevance) - 命中即保留 (僅保留「純原生綠色低碳技術」)
HIGH_RELEVANCE_TERMS = [
    r"hydrogen metallurgy", 
    r"hydrogen[-\s]based", 
    r"hydrogen reduction",
    r"hydrogen[-\s]injection",
    r"co2[-\s]injection",
    r"molten oxide electrolysis",
    r"carbon capture", 
    r"co2 capture", 
    r"ccus\b", 
    r"carbon storage", 
    r"carbon sequestration",
    r"green steel", 
    r"carbon footprint", 
    r"carbon neutral", 
    r"decarbonization",
    r"biochar", 
    r"biomass", 
    r"low[-\s]carbon fuel", 
    r"low[-\s]carbon reducing agent",
    ("waste heat recovery", action_phrase_pattern(phrase_pattern("waste", "heat"), RECOVER_FORMS)),
    ("gas recovery", action_phrase_pattern(r"(?:gas|off[-\s]+gas|waste[-\s]+gas|exhaust[-\s]+gas)", RECOVER_FORMS)),
    ("slag recycling", action_phrase_pattern(r"(?:slag|steel[-\s]+slag|converter[-\s]+slag|blast[-\s]+furnace[-\s]+slag)", RECYCLE_FORMS)),
    ("top gas recovery", action_phrase_pattern(phrase_pattern("top", "gas"), RECOVER_FORMS)),
    r"full scrap", 
    r"100% scrap", 
    r"solar thermal steelmaking",
    ("top gas recycling", action_phrase_pattern(phrase_pattern("top", "gas"), RECYCLE_FORMS)),
    r"\btgr\b"
]

# 3. 中度相關關鍵字 (Medium Relevance) - 需搭配 ACTION_TERMS
# (已將傳統核心設備與副產物詞彙完整收錄於此，降級實施動態交叉驗證)
MEDIUM_RELEVANCE_TERMS = [
    r"blast furnace", 
    r"converter", 
    r"smelting furnace", 
    r"electric furnace",
    r"electric arc furnace",
    r"arc furnace",
    r"\beaf\b",
    r"shaft furnace",
    r"direct reduction",
    r"direct reduced iron",
    r"\bdri\b",
    r"sponge iron",
    r"melting reduction",
    r"steel[-\s]slag",
    r"steelmaking[-\s]slag",
    r"converter[-\s]slag",
    r"blast[-\s]furnace[-\s]slag",
    r"molten slag",
    r"\bslag(?:s)?\b",
    r"steelmaking[-\s]dust",
    r"converter[-\s]dust",
    r"blast[-\s]furnace[-\s]dust",
    r"iron[-\s]and[-\s]steel[-\s]dust",
    r"\bdust(?:s)?\b",
    r"blast[-\s]furnace[-\s]gas",
    r"top[-\s]gas",
    r"coke[-\s]oven[-\s]gas",
    r"furnace[-\s]gas",
    r"flue[-\s]gas",
    r"waste[-\s]gas",
    r"exhaust[-\s]gas",
    r"off[-\s]gas",
    r"\bgas(?:es)?\b",
    r"waste heat",
    r"scrap\b", 
    r"steel scrap"
]

# 4. 動作/目的關鍵字 (Action/Purpose Terms)
ACTION_TERMS = [
    ("recover/recovery/recovered/recovering", RECOVER_FORMS),
    ("recycle/recycling/recycled", RECYCLE_FORMS),
    r"recirculat\w*",
    r"reuse", 
    r"utilization",
    r"efficiency", 
    r"reduce (?:emissions?|co2|carbon dioxide|coke rate|carbon consumption|energy consumption)",
    r"reducing emissions?",
    r"suppress (?:co2|carbon dioxide|emissions?)",
    r"carbonat\w*",
    r"methanat\w*",
    r"sequest\w*",
    r"energy saving", 
    r"optimization",
    r"preheat\w*",
    r"generate steam",
    r"power generation",
    r"low[-\s]carbon",
    r"green\b"
]

# 編譯正規表示式
def compile_patterns(term_list):
    compiled = []
    labels = []

    for term in term_list:
        if isinstance(term, tuple):
            label, pattern = term
        else:
            label, pattern = term, term

        compiled.append(re.compile(pattern, re.IGNORECASE))
        labels.append(label)

    return compiled, labels

exclude_patterns, exclude_labels = compile_patterns(EXCLUSION_TERMS)
high_patterns, high_labels = compile_patterns(HIGH_RELEVANCE_TERMS)
medium_patterns, medium_labels = compile_patterns(MEDIUM_RELEVANCE_TERMS)
action_patterns, action_labels = compile_patterns(ACTION_TERMS)

def get_matched_terms(text, patterns, terms):
    matched = []
    for pattern, term in zip(patterns, terms):
        if pattern.search(text):
            matched.append(term)
    return matched

print("載入資料中...")
records = json.loads(src.read_text(encoding="utf-8"))

filtered_records = []
stats = {
    "total": len(records),
    "noise_excluded": 0,
    "rule_a_high": 0,
    "rule_b_medium_action": 0,
    "unmatched": 0
}

high_counter = Counter()
medium_counter = Counter()
action_counter = Counter()
exclusion_counter = Counter()

print("開始篩選...")
for record in records:
    text = f"{record.get('title_en', '')} {record.get('abstract_en', '')}"
    
    # 檢查高相關詞
    high_hits = get_matched_terms(text, high_patterns, high_labels)
    
    # 檢查排除詞
    exclude_hits = get_matched_terms(text, exclude_patterns, exclude_labels)
    
    # 檢查中度與動作詞
    medium_hits = get_matched_terms(text, medium_patterns, medium_labels)
    action_hits = get_matched_terms(text, action_patterns, action_labels)
    
    # 規則邏輯判斷
    matched_rule = None
    
    if exclude_hits and not high_hits:
        # 如果命中排除詞（如低碳鋼產品），且沒有任何高相關技術（如DRI），則視為雜訊
        matched_rule = "Noise"
        stats["noise_excluded"] += 1
        exclusion_counter.update(exclude_hits)
    elif high_hits:
        # 命中高相關詞（原生低碳路徑或已組裝好雙向特徵的短語），直接保留
        matched_rule = "Rule_A"
        stats["rule_a_high"] += 1
        high_counter.update(high_hits)
    elif medium_hits and action_hits:
        # 命中中度相關詞（設備名、廣義渣/氣），且同時有回收/優化/減排等動作詞
        matched_rule = "Rule_B"
        stats["rule_b_medium_action"] += 1
        medium_counter.update(medium_hits)
        action_counter.update(action_hits)
    else:
        # 無法歸類，視為不相關
        stats["unmatched"] += 1
        
    if matched_rule in ["Rule_A", "Rule_B"]:
        output_record = dict(record)
        output_record["carbon_neutral_rule"] = matched_rule
        # 對標籤進行 list(set()) 去重，維持 Metadata 的乾淨度
        output_record["high_relevance_hits"] = list(set(high_hits))
        output_record["medium_relevance_hits"] = list(set(medium_hits))
        output_record["action_hits"] = list(set(action_hits))
        output_record["exclude_hits"] = list(set(exclude_hits))
        filtered_records.append(output_record)

print("寫入輸出檔案...")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(filtered_records, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

# 列印統計結果
print("\n=== 鋼鐵業碳中和關鍵字篩選 V2 最終精確版結果 ===")
print(f"輸入總筆數：{stats['total']:,}")
print(f"因命中排除詞(如材料配方)而剔除的雜訊筆數：{stats['noise_excluded']:,}")
print(f"因未命中任何規則而未納入的筆數：{stats['unmatched']:,}")
print(f"規則 A (純原生低碳技術) 納入：{stats['rule_a_high']:,}")
print(f"規則 B (設備特徵 + 減碳節能動作) 納入：{stats['rule_b_medium_action']:,}")
print(f"最後輸出筆數：{len(filtered_records):,}")
print(f"輸出檔案：{out}")
print("====================================")

def print_top(counter, title, limit=10):
    print(f"\n{title}")
    for term, count in counter.most_common(limit):
        print(f"  {term}: {count:,}")

print_top(high_counter, "高度相關詞命中 Top 10")
print_top(medium_counter, "中度相關詞命中 Top 10")
print_top(action_counter, "動作/目的詞命中 Top 10")
print_top(exclusion_counter, "排除雜訊詞命中 Top 10")