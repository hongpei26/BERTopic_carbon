import json
import re
from pathlib import Path


src = Path("/home/carbon/carbon/data_global/global_abstract_dedup.json")
out = Path("/home/carbon/carbon/data_global/global_carbon_neutral_keywords.json")
MIN_PRIORITY_YEAR = 2006
MAX_PRIORITY_YEAR = 2025


STRONG_HIGH_TERMS = [
    "carbon capture",
    "co2 capture",
    "ccus",
    "carbon neutral",
    "green steel",
    "decarbonization",
    "hydrogen direct reduction",
    "hydrogen based dri",
    "direct reduced iron",
    "sponge iron",
    "molten oxide electrolysis",
    "co2 elimination",
    "co2 scrubbing",
    "co2-free reduction gas",
    "top gas recycling",
    "blast furnace gas recycling",
    "steel slag carbonation",
    "producing direct reduced iron",
    "hydrogen metallurgy",
    "hot briquetted iron",
    "midrex",
    "energiron",
    "hybrit",
    "fossil-free steel",
    "zero carbon steel",
    "hydrogen ironmaking",
    "h2-dri",
    "h2 dri",
    "flash ironmaking",
    "electric smelting furnace",
]

CONDITIONAL_HIGH_TERMS = [
    "arc furnace",
    "electric arc furnace",
    "electric furnace",
    "EAF",
    "hydrogen",
    "hydrogen rich",
    "carbon dioxide",
    "CO2",
    "carbon monoxide",
    "carbon monoxide hydrogen",
    "blast furnace gas",
    "top gas",
    "blast furnace top gas",
    "steel slag",
    "molten oxide",
    "fuel cell",
    "molten carbonate fuel cell",
    "carbonate fuel cell",
    "electrolysis",
    "electrowinning",
    "low carbon",
    "reduced iron",
    "direct reduction",
    "shaft furnace",
    "reduction furnace",
    "reduction reactor",
    "reduction zone",
    "reducing gas",
    "reduction gas",
    "synthesis gas",
    "syngas",
    "reformed gas",
    "hot reducing gas",
    "steel scrap",
    "iron scrap",
    "cold iron source",
    "sequestration",
    "carbon storage",
    "carbon negative",
    "carbonation",
    "fixing carbon dioxide",
    "carbon footprint",
    "renewable energy",
    "greenhouse",
    "fluidized bed reduction",
    "fluidized bed reactor",
    "dri",
    "hbi",
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
    "energy management system",
    "smart monitoring",
    "digital twin",
    "artificial intelligence",
    "machine learning",
]

RULE_B_STEEL_CORE_TERMS = [
    "steel",
    "steelmaking",
    "steel works",
    "steelworks",
    "ironmaking",
    "iron manufacture",
    "blast furnace",
    "operating blast furnace",
    "coke oven",
    "converter",
    "basic oxygen furnace",
    "BOF",
    "electric arc furnace",
    "arc furnace",
    "electric furnace",
    "iron ore",
    "pig iron",
    "cast iron",
    "metallic iron",
    "reduced iron",
    "sponge iron",
    "direct reduction",
    "smelting reduction",
    "scrap",
    "steel scrap",
    "iron scrap",
    "steel slag",
    "steelmaking dust",
    "converter dust",
    "converter slag",
]

RULE_B_CARBON_TRANSITION_TERMS = [
    "carbon dioxide",
    "CO2",
    "carbon monoxide",
    "hydrogen",
    "biomass",
    "charcoal",
    "renewable",
    "oxygen",
    "oxygen gas",
    "oxidizing gas",
    "oxyfuel",
    "oxy fuel",
    "combustion",
    "burner",
    "flue gas",
    "exhaust gas",
    "waste gas",
    "off gas",
    "furnace gas",
    "blast furnace gas",
    "top gas",
    "waste heat",
    "heat recovery",
    "recycling",
    "reusing",
    "by-product",
    "byproduct",
    "coproduct",
    "sludge",
    "dust",
]

CORE_DECARBON_TERMS = [
    "top gas recycling",
    "gas recycling",
    "recycle gas",
    "co2 elimination",
    "co2 removal",
    "co2 scrubbing",
    "carbon capture",
    "carbonation",
    "mineralization",
    "carbon neutral",
    "decarbonization",
    "reduce co2",
    "reduce coke",
    "reduce reducing material",
    "reduce reductant",
    "biomass",
    "charcoal",
    "waste plastics",
    "plastic waste",
    "tyre-char",
    "renewable",
]

BROAD_ENERGY_TERMS = [
    "energy saving",
    "energy efficiency",
    "reduce energy",
    "reduce energy consumption",
    "low energy consumption",
    "heat recovery",
    "waste heat",
    "sensible heat",
    "preheating",
    "scrap preheating",
    "off-gas recovery",
    "off gas recovery",
    "recycling",
    "reuse",
    "recovering",
    "resource utilization",
]

DECARBON_INTENT_TERMS = CORE_DECARBON_TERMS + BROAD_ENERGY_TERMS

EQUIPMENT_NOISE_TERMS = [
    "electrode holder",
    "bottom electrode",
    "furnace bottom electrode",
    "tilting device",
    "current supply",
    "short-circuit capacity",
    "arc deflection",
    "magnetic field",
    "furnace lining",
    "hearth swinging",
    "charging device",
    "feeding device",
    "cooled lance",
    "observation device",
    "camera",
]

NON_STEEL_NOISE_TERMS = [
    "copper anode",
    "blister copper",
    "magnesium metal",
    "rare earth metal",
    "precious metal",
    "aluminum alloy",
    "aluminium alloy",
    "sulphide concentrate",
    "nickel ore",
    "ni ore",
    "chromium ore",
    "cr ore",
    "lead slag",
    "aluminum scrap",
    "scrap aluminum",
    "aluminium scrap",
    "waste aluminum",
    "copper scrap",
    "scrap copper",
    "bauxite",
    "alumina",
    "electrolytic aluminum",
    "copper smelting",
    "magnesium smelting",
]

STEEL_DUST_CONTEXT_TERMS = [
    "steelworks dust",
    "steel works dust",
    "steelmaking dust",
    "eaf dust",
    "electric arc furnace dust",
    "blast furnace dust",
    "converter dust",
    "furnace dust",
]

NEGATIVE_TERMS = [
    "wind turbine",
    "wind power",
    "wind farm",
    "rotor",
    "vanadium",
    "titanium",
    "titano",
    "fuel cell vehicle",
    "lithium ion",
    "li-ion",
    "battery pack",
    "solar cell",
    "photovoltaic",
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


STRONG_HIGH_FLEXIBLE_PATTERNS = compile_flexible_patterns({
    "carbon capture": flexible_phrase_pattern("carbon", "capture", max_gap_words=2),
    "co2 capture": flexible_phrase_pattern("co2", "capture", max_gap_words=2),
    "hydrogen direct reduction": flexible_phrase_pattern("hydrogen", "direct", "reduction", max_gap_words=1),
    "hydrogen based dri": flexible_phrase_pattern("hydrogen", "based", "dri", max_gap_words=1),
    "direct reduced iron": flexible_phrase_pattern("direct", "reduced", "iron", max_gap_words=1),
    "molten oxide electrolysis": re.compile(
        r"(?<![a-z0-9])(?:"
        r"molten(?:\W+\w+){0,1}\W+oxide(?:\W+\w+){0,1}\W+electrolysis"
        r"|electrolysis(?:\W+\w+){0,2}\W+molten(?:\W+\w+){0,1}\W+oxide"
        r")(?![a-z0-9])",
        re.IGNORECASE,
    ),
    "co2 elimination": flexible_phrase_pattern("co2", "elimination", max_gap_words=2),
    "co2 scrubbing": flexible_phrase_pattern("co2", "scrubbing", max_gap_words=2),
    "co2-free reduction gas": re.compile(
        r"(?<![a-z0-9])co2[-\s]*free(?:\W+\w+){0,2}\W+reduction(?:\W+\w+){0,1}\W+gas(?![a-z0-9])",
        re.IGNORECASE,
    ),
    "top gas recycling": flexible_phrase_pattern("top", "gas", "recycling", max_gap_words=2),
    "blast furnace gas recycling": flexible_phrase_pattern("blast", "furnace", "gas", "recycling", max_gap_words=2),
    "steel slag carbonation": flexible_phrase_pattern("steel", "slag", "carbonation", max_gap_words=2),
    "producing direct reduced iron": flexible_phrase_pattern("producing", "direct", "reduced", "iron", max_gap_words=1),
    "hydrogen metallurgy": flexible_phrase_pattern("hydrogen", "metallurgy", max_gap_words=1),
    "hot briquetted iron": flexible_phrase_pattern("hot", "briquetted", "iron", max_gap_words=1),
})

CONDITIONAL_HIGH_FLEXIBLE_PATTERNS = compile_flexible_patterns({
    "direct reduction": flexible_phrase_pattern("direct", "reduction", max_gap_words=2),
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
    "carbon storage": flexible_phrase_pattern("carbon", "storage", max_gap_words=2),
    "low carbon": flexible_phrase_pattern("low", "carbon", max_gap_words=2),
    "renewable energy": flexible_phrase_pattern("renewable", "energy", max_gap_words=1),
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
    "energy management system": flexible_phrase_pattern("energy", "management", "system", max_gap_words=1),
    "smart monitoring": flexible_phrase_pattern("smart", "monitoring", max_gap_words=1),
    "digital twin": flexible_phrase_pattern("digital", "twin", max_gap_words=1),
    "artificial intelligence": flexible_phrase_pattern("artificial", "intelligence", max_gap_words=1),
    "machine learning": flexible_phrase_pattern("machine", "learning", max_gap_words=1),
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


STRONG_HIGH_PATTERNS = compile_term_patterns(STRONG_HIGH_TERMS)
CONDITIONAL_HIGH_PATTERNS = compile_term_patterns(CONDITIONAL_HIGH_TERMS)
MEDIUM_PATTERNS = compile_term_patterns(MEDIUM_RELEVANCE_TERMS)
RULE_B_STEEL_CORE_PATTERNS = compile_term_patterns(RULE_B_STEEL_CORE_TERMS)
RULE_B_CARBON_TRANSITION_PATTERNS = compile_term_patterns(RULE_B_CARBON_TRANSITION_TERMS)
DECARBON_INTENT_PATTERNS = compile_term_patterns(DECARBON_INTENT_TERMS)
EQUIPMENT_NOISE_PATTERNS = compile_term_patterns(EQUIPMENT_NOISE_TERMS)
NON_STEEL_NOISE_PATTERNS = compile_term_patterns(NON_STEEL_NOISE_TERMS)
STEEL_DUST_CONTEXT_PATTERNS = compile_term_patterns(STEEL_DUST_CONTEXT_TERMS)
NEGATIVE_PATTERNS = compile_term_patterns(NEGATIVE_TERMS)

LOW_CARBON_STEEL_NOISE_PATTERNS = [
    re.compile(r"low\s+carbon\s+steel", re.IGNORECASE),
    re.compile(r"low\s+carbon\s+less\s+than", re.IGNORECASE),
    re.compile(r"carbon\s+less\s+than\s+\d", re.IGNORECASE),
    re.compile(r"carbon\s+content\s+.*wt%", re.IGNORECASE),
]


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


def priority_year(record):
    priority_date = record.get("priority_date")
    if priority_date is None:
        return None

    priority_date_text = str(priority_date).strip()
    if len(priority_date_text) < 4 or not priority_date_text[:4].isdigit():
        return None

    return int(priority_date_text[:4])


records = json.loads(src.read_text(encoding="utf-8"))
input_count = len(records)
records = [
    record for record in records
    if (year := priority_year(record)) is not None
    and MIN_PRIORITY_YEAR <= year <= MAX_PRIORITY_YEAR
]

filtered = []
rule_a1_count = 0
rule_a2_count = 0
rule_b_count = 0
rule_b_rejected_count = 0
equipment_noise_excluded_count = 0
non_steel_noise_excluded_count = 0
low_carbon_noise_excluded_count = 0
strong_high_counter = {}
conditional_high_counter = {}
medium_counter = {}
strong_high_flexible_counter = {}
conditional_high_flexible_counter = {}
medium_flexible_counter = {}
decarbon_intent_counter = {}

for record in records:
    text = record.get("abstract_en") or ""

    negative_hits = matched_terms(text, NEGATIVE_PATTERNS)
    if negative_hits:
        continue

    strong_high_exact_hits = matched_terms(text, STRONG_HIGH_PATTERNS)
    strong_high_flexible_hits = [
        term for term in matched_terms(text, STRONG_HIGH_FLEXIBLE_PATTERNS)
        if term not in strong_high_exact_hits
    ]
    strong_high_hits = merge_hits(strong_high_exact_hits, strong_high_flexible_hits)

    conditional_high_exact_hits = matched_terms(text, CONDITIONAL_HIGH_PATTERNS)
    conditional_high_flexible_hits = [
        term for term in matched_terms(text, CONDITIONAL_HIGH_FLEXIBLE_PATTERNS)
        if term not in conditional_high_exact_hits
    ]
    conditional_high_hits = merge_hits(
        conditional_high_exact_hits,
        conditional_high_flexible_hits,
    )
    high_hits = merge_hits(strong_high_hits, conditional_high_hits)

    medium_exact_hits = matched_terms(text, MEDIUM_PATTERNS)
    medium_flexible_hits = [
        term for term in matched_terms(text, MEDIUM_FLEXIBLE_PATTERNS)
        if term not in medium_exact_hits
    ]
    medium_hits = merge_hits(medium_exact_hits, medium_flexible_hits)
    rule_b_steel_core_hits = matched_terms(text, RULE_B_STEEL_CORE_PATTERNS)
    rule_b_carbon_transition_hits = matched_terms(text, RULE_B_CARBON_TRANSITION_PATTERNS)
    decarbon_intent_hits = matched_terms(text, DECARBON_INTENT_PATTERNS)

    equipment_noise_hits = matched_terms(text, EQUIPMENT_NOISE_PATTERNS)
    if equipment_noise_hits and not decarbon_intent_hits:
        equipment_noise_excluded_count += 1
        continue

    non_steel_noise_hits = matched_terms(text, NON_STEEL_NOISE_PATTERNS)
    steel_dust_context_hits = matched_terms(text, STEEL_DUST_CONTEXT_PATTERNS)
    if non_steel_noise_hits and not steel_dust_context_hits:
        non_steel_noise_excluded_count += 1
        continue

    low_carbon_noise_hits = [
        pattern.pattern
        for pattern in LOW_CARBON_STEEL_NOISE_PATTERNS
        if pattern.search(text)
    ]
    non_low_carbon_intent_hits = [
        term for term in decarbon_intent_hits
        if term.lower() != "low carbon"
    ]
    if (
        low_carbon_noise_hits
        and "low carbon" in conditional_high_hits
        and not strong_high_hits
        and not non_low_carbon_intent_hits
    ):
        low_carbon_noise_excluded_count += 1
        continue

    matched_rule = None
    if strong_high_hits:
        matched_rule = "A1_STRONG"
        rule_a1_count += 1
    elif conditional_high_hits and rule_b_steel_core_hits and decarbon_intent_hits:
        matched_rule = "A2_CONDITIONAL"
        rule_a2_count += 1
    elif len(medium_hits) >= 2:
        if rule_b_steel_core_hits and decarbon_intent_hits:
            core_decarbon_hits = [t for t in decarbon_intent_hits if t in CORE_DECARBON_TERMS]
            heavy_equip_terms = {
                "blast furnace", "operating blast furnace", "coke oven", 
                "converter", "basic oxygen furnace", "BOF", 
                "electric arc furnace", "arc furnace", "electric furnace"
            }
            has_heavy_equip = any(t in heavy_equip_terms for t in rule_b_steel_core_hits)
            
            if core_decarbon_hits or len(medium_hits) >= 3 or has_heavy_equip:
                matched_rule = "B_MEDIUM"
                rule_b_count += 1
            else:
                rule_b_rejected_count += 1
        else:
            rule_b_rejected_count += 1

    if matched_rule is None:
        continue

    for term in strong_high_hits:
        strong_high_counter[term] = strong_high_counter.get(term, 0) + 1
    for term in strong_high_flexible_hits:
        strong_high_flexible_counter[term] = strong_high_flexible_counter.get(term, 0) + 1
    for term in conditional_high_hits:
        conditional_high_counter[term] = conditional_high_counter.get(term, 0) + 1
    for term in conditional_high_flexible_hits:
        conditional_high_flexible_counter[term] = conditional_high_flexible_counter.get(term, 0) + 1
    for term in medium_hits:
        medium_counter[term] = medium_counter.get(term, 0) + 1
    for term in medium_flexible_hits:
        medium_flexible_counter[term] = medium_flexible_counter.get(term, 0) + 1
    for term in decarbon_intent_hits:
        decarbon_intent_counter[term] = decarbon_intent_counter.get(term, 0) + 1

    output_record = dict(record)
    output_record["carbon_neutral_rule"] = matched_rule
    output_record["strong_high_hits"] = strong_high_hits
    output_record["strong_high_exact_hits"] = strong_high_exact_hits
    output_record["strong_high_flexible_hits"] = strong_high_flexible_hits
    output_record["conditional_high_hits"] = conditional_high_hits
    output_record["conditional_high_exact_hits"] = conditional_high_exact_hits
    output_record["conditional_high_flexible_hits"] = conditional_high_flexible_hits
    output_record["high_relevance_hits"] = high_hits
    output_record["high_relevance_exact_hits"] = merge_hits(
        strong_high_exact_hits,
        conditional_high_exact_hits,
    )
    output_record["high_relevance_flexible_hits"] = merge_hits(
        strong_high_flexible_hits,
        conditional_high_flexible_hits,
    )
    output_record["medium_relevance_hits"] = medium_hits
    output_record["medium_relevance_exact_hits"] = medium_exact_hits
    output_record["medium_relevance_flexible_hits"] = medium_flexible_hits
    output_record["steel_core_hits"] = rule_b_steel_core_hits
    output_record["decarbon_intent_hits"] = decarbon_intent_hits
    output_record["rule_b_steel_core_hits"] = rule_b_steel_core_hits if matched_rule == "B_MEDIUM" else []
    output_record["rule_b_carbon_transition_hits"] = (
        rule_b_carbon_transition_hits if matched_rule == "B_MEDIUM" else []
    )
    output_record["equipment_noise_hits"] = equipment_noise_hits
    output_record["non_steel_noise_hits"] = non_steel_noise_hits
    output_record["steel_dust_context_hits"] = steel_dust_context_hits
    output_record["low_carbon_noise_hits"] = low_carbon_noise_hits
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
print(f"輸入總筆數：{input_count:,}")
print(f"priority_date 年份 {MIN_PRIORITY_YEAR}-{MAX_PRIORITY_YEAR} 篩選後筆數：{len(records):,}")
print(f"規則 A1 強碳中和詞直接納入：{rule_a1_count:,}")
print(f"規則 A2 條件高相關詞 + 鋼鐵核心詞 + 碳中和意圖詞納入：{rule_a2_count:,}")
print(f"規則 B 至少 2 個中度詞 + 鋼鐵核心詞 + 碳中和意圖詞納入：{rule_b_count:,}")
print(f"規則 B 命中至少 2 個中度相關詞但未通過雙條件排除：{rule_b_rejected_count:,}")
print(f"設備型雜訊且無碳中和意圖排除：{equipment_noise_excluded_count:,}")
print(f"非鋼鐵金屬場域且無鋼鐵粉塵/副產物脈絡排除：{non_steel_noise_excluded_count:,}")
print(f"low carbon 材料成分誤判排除：{low_carbon_noise_excluded_count:,}")
print(f"最後輸出筆數：{len(filtered):,}")
print(f"未納入筆數：{len(records) - len(filtered):,}")
print(f"輸出檔案：{out}")
print("================================")

print_top(strong_high_counter, "強碳中和詞 Top 20")
print_top(strong_high_flexible_counter, "強碳中和詞 Flexible 額外命中 Top 20")
print_top(conditional_high_counter, "條件高相關詞 Top 20")
print_top(conditional_high_flexible_counter, "條件高相關詞 Flexible 額外命中 Top 20")
print_top(medium_counter, "中度相關詞 Top 20")
print_top(medium_flexible_counter, "中度相關詞 Flexible 額外命中 Top 20")
print_top(decarbon_intent_counter, "碳中和意圖詞 Top 20")
