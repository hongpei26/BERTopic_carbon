import json
import re
from pathlib import Path


#src = Path("/home/carbon/carbon/data/part-000000000000_domain_target_intersection.json")
src = Path("/home/carbon/carbon/data/cpc_domain_target_intersection_3.json")
out = Path("/home/carbon/carbon/data/carbon_neutral_keywords_4.json")
#out = Path("/home/carbon/carbon/data/part-000000000000_carbon_neutral_keywords.json")


HIGH_RELEVANCE_TERMS = [
    "direct reduction",
    "direct reduced iron",
    "reduced iron",
    "sponge iron",
    "shaft furnace",
    "reduction furnace",
    "reduction reactor",
    "reduction zone",
    "producing direct reduced iron",
    "reducing gas",
    "reduction gas",
    "synthesis gas",
    "syngas",
    "reformed gas",
    "hot reducing gas",
    "hydrogen",
    "hydrogen rich",
    "carbon monoxide",
    "carbon monoxide hydrogen",
    "blast furnace gas",
    "top gas",
    "blast furnace top gas",
    "electric arc furnace",
    "arc furnace",
    "EAF",
    "electric furnace",
    "scrap",
    "steel scrap",
    "iron scrap",
    "cold iron source",
    "carbon dioxide",
    "CO2",
    "carbon capture",
    "co2 capture",
    "ccus",
    "sequestration",
    "carbon storage",
    "carbon negative",
    "carbonation",
    "fixing carbon dioxide",
    "decarbonization",
    "carbon neutral",
    "carbon footprint",
    "low carbon",
    "renewable energy",
    "green steel",
    "greenhouse",
    "steel slag",
    "fluidized bed reduction",
    "fluidized bed reactor",
    "fuel cell",
    "molten carbonate fuel cell",
    "carbonate fuel cell",
    "electrolysis",
    "electrowinning",
    "molten oxide electrolysis",
    "molten oxide",
    "dri",
    "hbi",
    "hot briquetted iron",
    "midrex",
    "energiron",
    "solar thermal steelmaking",
    "solar steelmaking",
    "solar thermal reduction",
]

MEDIUM_RELEVANCE_TERMS = [
    "iron oxide",
    "iron ore",
    "blast furnace",
    "furnace gas",
    "coke oven gas",
    "hot blast",
    "stove",
    "operating blast furnace",
    "molten slag",
    "converter slag",
    "black slag",
    "sludge",
    "dust",
    "steelmaking dust",
    "furnace dust",
    "converter dust",
    "zinc oxide",
    "CO-rich gas",
    "oxygen",
    "oxygen gas",
    "oxidizing gas",
    "oxyfuel",
    "oxy fuel",
    "combustion",
    "burner",
    "reactor",
    "bed reactor",
    "flue gas",
    "exhaust gas",
    "waste gas",
    "off gas",
    "waste heat",
    "heat recovery",
    "recycling",
    "biomass",
    "charcoal",
    "renewable",
    "red mud",
    "zinc dust",
    "by-product",
    "byproduct",
    "coproduct",
]

FILLER_WORDS = [
    "a",
    "an",
    "and",
    "based",
    "containing",
    "for",
    "from",
    "in",
    "of",
    "or",
    "the",
    "to",
    "using",
    "with",
]


def term_token_pattern(token):
    escaped = re.escape(token.lower())
    return escaped.replace(r"\-", r"[-\s]").replace(r"\/", r"[/\s-]")


def flexible_phrase_pattern(*tokens, max_gap_words=2):
    gap = rf"(?:\W+(?:{'|'.join(FILLER_WORDS)}))*"
    loose_gap = rf"(?:\W+\w+){{0,{max_gap_words}}}\W+"
    parts = [term_token_pattern(token) for token in tokens]
    return re.compile(
        rf"(?<![a-z0-9]){parts[0]}"
        + "".join(rf"(?:{gap}|{loose_gap}){part}" for part in parts[1:])
        + r"(?![a-z0-9])",
        re.IGNORECASE,
    )


def compile_flexible_patterns(term_to_pattern):
    return [
        (term, re.compile(pattern, re.IGNORECASE) if isinstance(pattern, str) else pattern)
        for term, pattern in term_to_pattern.items()
    ]


HIGH_FLEXIBLE_PATTERNS = compile_flexible_patterns({
    "direct reduction": flexible_phrase_pattern("direct", "reduction", max_gap_words=2),
    "direct reduced iron": flexible_phrase_pattern("direct", "reduced", "iron", max_gap_words=1),
    "producing direct reduced iron": flexible_phrase_pattern("producing", "direct", "reduced", "iron", max_gap_words=1),
    "reducing gas": flexible_phrase_pattern("reducing", "gas", max_gap_words=2),
    "reduction furnace": flexible_phrase_pattern("reduction", "furnace", max_gap_words=1),
    "reduction reactor": flexible_phrase_pattern("reduction", "reactor", max_gap_words=1),
    "reduction zone": flexible_phrase_pattern("reduction", "zone", max_gap_words=1),
    "reduction gas": flexible_phrase_pattern("reduction", "gas", max_gap_words=2),
    "synthesis gas": flexible_phrase_pattern("synthesis", "gas", max_gap_words=1),
    "reformed gas": flexible_phrase_pattern("reformed", "gas", max_gap_words=2),
    "hot reducing gas": flexible_phrase_pattern("hot", "reducing", "gas", max_gap_words=2),
    "hydrogen rich": re.compile(
        r"(?<![a-z0-9])(?:hydrogen[-\s]+rich|rich\s+in\s+hydrogen)(?![a-z0-9])",
        re.IGNORECASE,
    ),
    "carbon monoxide hydrogen": re.compile(
        r"(?<![a-z0-9])carbon\s+monoxide(?:\W+(?:and|or|with|plus|/))*\W+hydrogen(?![a-z0-9])",
        re.IGNORECASE,
    ),
    "blast furnace gas": flexible_phrase_pattern("blast", "furnace", "gas", max_gap_words=1),
    "blast furnace top gas": flexible_phrase_pattern("blast", "furnace", "top", "gas", max_gap_words=1),
    "top gas": flexible_phrase_pattern("top", "gas", max_gap_words=1),
    "electric arc furnace": flexible_phrase_pattern("electric", "arc", "furnace", max_gap_words=1),
    "electric furnace": flexible_phrase_pattern("electric", "furnace", max_gap_words=1),
    "cold iron source": flexible_phrase_pattern("cold", "iron", "source", max_gap_words=1),
    "carbon capture": flexible_phrase_pattern("carbon", "capture", max_gap_words=2),
    "carbon dioxide": flexible_phrase_pattern("carbon", "dioxide", max_gap_words=1),
    "carbon negative": flexible_phrase_pattern("carbon", "negative", max_gap_words=2),
    "fixing carbon dioxide": flexible_phrase_pattern("fixing", "carbon", "dioxide", max_gap_words=2),
    "carbon footprint": flexible_phrase_pattern("carbon", "footprint", max_gap_words=1),
    "steel scrap": flexible_phrase_pattern("steel", "scrap", max_gap_words=1),
    "iron scrap": flexible_phrase_pattern("iron", "scrap", max_gap_words=1),
    "steel slag": flexible_phrase_pattern("steel", "slag", max_gap_words=1),
    "fluidized bed reduction": flexible_phrase_pattern("fluidized", "bed", "reduction", max_gap_words=1),
    "fluidized bed reactor": flexible_phrase_pattern("fluidized", "bed", "reactor", max_gap_words=1),
    "fuel cell": flexible_phrase_pattern("fuel", "cell", max_gap_words=1),
    "molten carbonate fuel cell": flexible_phrase_pattern("molten", "carbonate", "fuel", "cell", max_gap_words=1),
    "carbonate fuel cell": flexible_phrase_pattern("carbonate", "fuel", "cell", max_gap_words=1),
    "molten oxide": flexible_phrase_pattern("molten", "oxide", max_gap_words=1),
    "molten oxide electrolysis": re.compile(
        r"(?<![a-z0-9])(?:"
        r"molten(?:\W+\w+){0,1}\W+oxide(?:\W+\w+){0,1}\W+electrolysis"
        r"|electrolysis(?:\W+\w+){0,2}\W+molten(?:\W+\w+){0,1}\W+oxide"
        r")(?![a-z0-9])",
        re.IGNORECASE,
    ),
    "co2 capture": flexible_phrase_pattern("co2", "capture", max_gap_words=2),
    "carbon storage": flexible_phrase_pattern("carbon", "storage", max_gap_words=2),
    "low carbon": flexible_phrase_pattern("low", "carbon", max_gap_words=2),
    "renewable energy": flexible_phrase_pattern("renewable", "energy", max_gap_words=1),
    "green steel": flexible_phrase_pattern("green", "steel", max_gap_words=1),
    "hot briquetted iron": flexible_phrase_pattern("hot", "briquetted", "iron", max_gap_words=1),
    "solar thermal steelmaking": flexible_phrase_pattern("solar", "thermal", "steelmaking", max_gap_words=1),
    "solar steelmaking": flexible_phrase_pattern("solar", "steelmaking", max_gap_words=1),
    "solar thermal reduction": flexible_phrase_pattern("solar", "thermal", "reduction", max_gap_words=1),
})


MEDIUM_FLEXIBLE_PATTERNS = compile_flexible_patterns({
    "iron oxide": flexible_phrase_pattern("iron", "oxide", max_gap_words=1),
    "iron ore": flexible_phrase_pattern("iron", "ore", max_gap_words=1),
    "blast furnace": flexible_phrase_pattern("blast", "furnace", max_gap_words=1),
    "furnace gas": flexible_phrase_pattern("furnace", "gas", max_gap_words=2),
    "coke oven gas": flexible_phrase_pattern("coke", "oven", "gas", max_gap_words=1),
    "hot blast": flexible_phrase_pattern("hot", "blast", max_gap_words=1),
    "operating blast furnace": flexible_phrase_pattern("operating", "blast", "furnace", max_gap_words=1),
    "molten slag": flexible_phrase_pattern("molten", "slag", max_gap_words=1),
    "converter slag": flexible_phrase_pattern("converter", "slag", max_gap_words=1),
    "black slag": flexible_phrase_pattern("black", "slag", max_gap_words=1),
    "steelmaking dust": flexible_phrase_pattern("steelmaking", "dust", max_gap_words=1),
    "furnace dust": flexible_phrase_pattern("furnace", "dust", max_gap_words=1),
    "converter dust": flexible_phrase_pattern("converter", "dust", max_gap_words=1),
    "zinc oxide": flexible_phrase_pattern("zinc", "oxide", max_gap_words=1),
    "CO-rich gas": re.compile(
        r"(?<![a-z0-9])(?:co|carbon\s+monoxide)[-\s]+rich(?:\W+\w+){0,2}\W+gas(?![a-z0-9])",
        re.IGNORECASE,
    ),
    "oxygen gas": flexible_phrase_pattern("oxygen", "gas", max_gap_words=1),
    "oxidizing gas": flexible_phrase_pattern("oxidizing", "gas", max_gap_words=1),
    "oxy fuel": re.compile(r"(?<![a-z0-9])oxy[-\s]?fuel(?![a-z0-9])", re.IGNORECASE),
    "bed reactor": flexible_phrase_pattern("bed", "reactor", max_gap_words=1),
    "flue gas": flexible_phrase_pattern("flue", "gas", max_gap_words=1),
    "exhaust gas": flexible_phrase_pattern("exhaust", "gas", max_gap_words=1),
    "waste gas": flexible_phrase_pattern("waste", "gas", max_gap_words=1),
    "off gas": flexible_phrase_pattern("off", "gas", max_gap_words=1),
    "waste heat": flexible_phrase_pattern("waste", "heat", max_gap_words=1),
    "heat recovery": flexible_phrase_pattern("heat", "recovery", max_gap_words=2),
    "red mud": flexible_phrase_pattern("red", "mud", max_gap_words=1),
    "zinc dust": flexible_phrase_pattern("zinc", "dust", max_gap_words=1),
    "by-product": re.compile(r"(?<![a-z0-9])by[-\s]?product(?:s)?(?![a-z0-9])", re.IGNORECASE),
    "coproduct": re.compile(r"(?<![a-z0-9])co[-\s]?product(?:s)?(?![a-z0-9])", re.IGNORECASE),
})


def compile_term_patterns(terms):
    patterns = []
    for term in terms:
        normalized = term.lower()
        escaped = re.escape(normalized).replace(r"\ ", r"\s+")

        if re.fullmatch(r"[a-z0-9]+", normalized):
            pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
        else:
            pattern = re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE)

        patterns.append((term, pattern))
    return patterns


HIGH_PATTERNS = compile_term_patterns(HIGH_RELEVANCE_TERMS)
MEDIUM_PATTERNS = compile_term_patterns(MEDIUM_RELEVANCE_TERMS)


def matched_terms(text, patterns):
    return [term for term, pattern in patterns if pattern.search(text)]


def merge_hits(*hit_lists):
    seen = set()
    merged = []
    for hits in hit_lists:
        for term in hits:
            if term not in seen:
                seen.add(term)
                merged.append(term)
    return merged


records = json.loads(src.read_text(encoding="utf-8"))

filtered = []
rule_a_count = 0
rule_b_count = 0
high_counter = {}
medium_counter = {}
high_flexible_counter = {}
medium_flexible_counter = {}

for record in records:
    text = f"{record.get('title_en') or ''} {record.get('abstract_en') or ''}"

    high_exact_hits = matched_terms(text, HIGH_PATTERNS)
    high_flexible_hits = [
        term for term in matched_terms(text, HIGH_FLEXIBLE_PATTERNS)
        if term not in high_exact_hits
    ]
    high_hits = merge_hits(high_exact_hits, high_flexible_hits)

    medium_exact_hits = matched_terms(text, MEDIUM_PATTERNS)
    medium_flexible_hits = [
        term for term in matched_terms(text, MEDIUM_FLEXIBLE_PATTERNS)
        if term not in medium_exact_hits
    ]
    medium_hits = merge_hits(medium_exact_hits, medium_flexible_hits)

    matched_rule = None
    if high_hits:
        matched_rule = "A"
        rule_a_count += 1
    elif len(medium_hits) >= 2:
        matched_rule = "B"
        rule_b_count += 1

    if matched_rule is None:
        continue

    for term in high_hits:
        high_counter[term] = high_counter.get(term, 0) + 1
    for term in high_flexible_hits:
        high_flexible_counter[term] = high_flexible_counter.get(term, 0) + 1
    for term in medium_hits:
        medium_counter[term] = medium_counter.get(term, 0) + 1
    for term in medium_flexible_hits:
        medium_flexible_counter[term] = medium_flexible_counter.get(term, 0) + 1

    output_record = dict(record)
    output_record["carbon_neutral_rule"] = matched_rule
    output_record["high_relevance_hits"] = high_hits
    output_record["high_relevance_exact_hits"] = high_exact_hits
    output_record["high_relevance_flexible_hits"] = high_flexible_hits
    output_record["medium_relevance_hits"] = medium_hits
    output_record["medium_relevance_exact_hits"] = medium_exact_hits
    output_record["medium_relevance_flexible_hits"] = medium_flexible_hits
    filtered.append(output_record)

out.write_text(
    json.dumps(filtered, ensure_ascii=False, indent=2, allow_nan=False),
    encoding="utf-8",
)


def print_top(counter, title, limit=20):
    print()
    print(title)
    for term, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]:
        print(f"  {term}: {count:,}")


print()
print("=== 鋼鐵業碳中和關鍵字篩選結果 ===")
print(f"輸入總筆數：{len(records):,}")
print(f"規則 A 命中高度相關詞納入：{rule_a_count:,}")
print(f"規則 B 命中至少 2 個中度相關詞納入：{rule_b_count:,}")
print(f"最後輸出筆數：{len(filtered):,}")
print(f"未納入筆數：{len(records) - len(filtered):,}")
print(f"輸出檔案：{out}")
print("================================")

print_top(high_counter, "高度相關詞 Top 20")
print_top(high_flexible_counter, "高度相關詞 Flexible 額外命中 Top 20")
print_top(medium_counter, "中度相關詞 Top 20")
print_top(medium_flexible_counter, "中度相關詞 Flexible 額外命中 Top 20")
