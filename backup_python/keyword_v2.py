import json
import re
import math
from pathlib import Path
from collections import Counter, defaultdict


# =============================================================================
# 0. Input / Output Path
# =============================================================================

INPUT_PATH = Path(
    "/home/carbon/carbon/data_global_v2/Carbon_onlycpc_global_morecpc_v2/"
    "global_onlycpc_domain_target_intersection.json"
)

OUTPUT_DIR = Path(
    "/home/carbon/carbon/data_global_v2/Carbon_onlycpc_global_morecpc_v2/"
    "weighted_relevance_output"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FULL = OUTPUT_DIR / "weighted_relevance_review.json"
OUT_RELATED = OUTPUT_DIR / "weighted_related_patents.json"
OUT_UNRELATED = OUTPUT_DIR / "weighted_unrelated_patents.json"
OUT_AUDIT = OUTPUT_DIR / "weighted_unrelated_high_confidence_false_negative.json"
OUT_SUMMARY = OUTPUT_DIR / "weighted_relevance_summary.json"


# =============================================================================
# 1. CPC Weight Settings
# =============================================================================

CORE_CPC_WEIGHTS = {
    # Y02C: greenhouse gas capture / storage / utilization
    "Y02C20/10": 3.0,
    "Y02C20/20": 3.0,
    "Y02C20/30": 3.0,
    "Y02C20/40": 3.0,

    # Y02P10: iron / steel decarbonization core
    "Y02P10/122": 3.0,
    "Y02P10/134": 3.0,
    "Y02P10/143": 3.0,
    "Y02P10/146": 3.0,
    "Y02P10/32": 3.0,

    # CO2-specific separation
    "B01D53/62": 3.0,
}

SUPPORT_CPC_WEIGHTS = {
    # Broad gas treatment
    "B01D53": 1.0,
    "B01D2053": 1.0,

    # Broad metal production / steel process decarbonization
    "Y02P10/10": 1.0,
    "Y02P10/20": 1.0,

    # Energy / gas / heat related, but still broad
    "Y02P10/25": 1.5,

    # Mineral processing
    "Y02P40/10": 1.0,
    "Y02P40/121": 1.0,
    "Y02P40/125": 1.0,
    "Y02P40/18": 1.0,

    # Cross-sector energy efficiency / waste reduction
    "Y02P80/10": 1.0,
    "Y02P80/15": 1.0,
    "Y02P80/30": 1.0,
    "Y02P80/40": 1.0,

    # Smart manufacturing / hydrogen / GHG management
    "Y02P90/02": 1.0,
    "Y02P90/30": 1.0,
    "Y02P90/40": 1.0,
    "Y02P90/45": 1.0,
    "Y02P90/50": 1.0,
    "Y02P90/80": 1.0,
    "Y02P90/82": 1.0,
    "Y02P90/84": 1.0,
    "Y02P90/845": 1.0,
}

PERIPHERAL_CPC_WEIGHTS = {
    # Chemical industry decarbonization
    "Y02P20/129": 0.5,
    "Y02P20/143": 0.5,
    "Y02P20/145": 0.5,
    "Y02P20/151": 0.5,
    "Y02P20/582": 0.5,
    "Y02P20/584": 0.5,

    # Final product manufacturing
    "Y02P70/10": 0.5,
}


# =============================================================================
# 2. Keyword Weight Settings
# =============================================================================

HIGH_RELEVANCE_TERMS = {
    # CO2 / CCUS
    r"\bccus\b": 3.0,
    r"carbon\s+capture": 3.0,
    r"carbon\s+capture\s+utilization\s+and\s+storage": 3.0,
    r"co\s*2\s+captur\w*": 3.0,
    r"co\s*2\s+remov\w*": 3.0,
    r"co\s*2\s+liquef\w*": 3.0,
    r"co\s*2\s+utili[sz]\w*": 3.0,
    r"co\s*2\s+sequest\w*": 3.0,
    r"carbon\s+dioxide\s+captur\w*": 3.0,
    r"carbon\s+dioxide\s+remov\w*": 3.0,
    r"carbon\s+dioxide\s+liquef\w*": 3.0,
    r"carbon\s+dioxide\s+utili[sz]\w*": 3.0,
    r"carbon\s+dioxide\s+sequest\w*": 3.0,
    r"carbon[-\s]?negative": 3.0,
    r"carbon\s+neutral": 3.0,
    r"carbon-neutral": 3.0,
    r"green\s+steel": 3.0,

    # Hydrogen / DRI / low-carbon reduction
    r"hydrogen\s+metallurgy": 2.5,
    r"hydrogen[-\s]?rich": 2.5,
    r"hydrogen\s+reduction": 2.5,
    r"hydrogen\s+concentration": 2.0,
    r"direct\s+reduced\s+iron": 2.5,
    r"\bDRI\b": 2.5,
    r"sponge\s+iron": 2.0,
    r"fluidi[sz]ed\s+bed\s+reduction": 2.0,
    r"reducing\s+gas": 1.5,

    # Biomass / biochar / alternative reductant
    r"biomass": 2.5,
    r"biochar": 2.5,
    r"renewable\s+carbon": 2.5,
    r"bio[-\s]?reduction": 2.5,
    r"biogenic\s+carbon": 2.0,
    r"carbon-neutral\s+substitute": 2.5,

    # Gas recovery / utilization
    r"blast\s+furnace\s+gas\s+recover\w*": 2.5,
    r"blast\s+furnace\s+gas\s+full\s+recovery": 2.5,
    r"top\s+gas\s+recover\w*": 2.5,
    r"top\s+gas\s+recirculat\w*": 2.5,
    r"converter\s+gas\s+recover\w*": 2.5,
    r"off[-\s]?gas\s+recover\w*": 2.0,
    r"process\s+gas\s+direct\s+recycl\w*": 2.0,
    r"byproduct\s+gas\s+recover\w*": 2.0,
    r"equalizing\s+gas\s+recover\w*": 2.5,
    r"pressure\s+equalizing\s+gas\s+recover\w*": 2.5,

    # Waste heat / energy
    r"waste\s+heat\s+recovery": 2.5,
    r"heat\s+recovery": 2.0,
    r"thermal\s+energy\s+recover\w*": 2.0,
    r"residual\s+energy\s+recover\w*": 2.0,
    r"molten\s+slag\s+energy\s+extraction": 2.0,
    r"slag\s+waste\s+heat": 2.0,
    r"minimum\s+energy\s+utilization": 2.0,

    # Slag / solid waste / recycling
    r"steel\s+slag\s+recycl\w*": 2.0,
    r"converter\s+slag\s+recycl\w*": 2.0,
    r"blast\s+furnace\s+slag\s+recycl\w*": 2.0,
    r"steelmaking\s+slag": 2.0,
    r"metallurgical\s+solid\s+waste": 2.5,
    r"steelworks\s+waste": 2.5,
    r"steelmaking\s+revert": 2.0,
    r"metallurgical\s+waste\s+powder": 2.0,

    # EAF / scrap
    r"scrap\s+preheat\w*": 2.0,
    r"electric\s+arc\s+furnace\s+scrap": 2.0,
    r"scrap\s+steel\s+recycl\w*": 2.0,

    # Electrolysis / advanced low-carbon ironmaking
    r"molten\s+oxide\s+electrolysis": 3.0,
    r"electrolysis\s+of\s+carbon\s+dioxide": 3.0,
    r"electrochemically\s+reduc\w*.*iron": 2.5,
}

DECARB_INTENT_TERMS = {
    # Direct emission reduction
    r"reduce\s+carbon\s+emission\w*": 1.5,
    r"reduce\s+co\s*2": 1.5,
    r"reduce\s+carbon\s+dioxide": 1.5,
    r"reduce\s+greenhouse\s+gas": 1.5,
    r"lower\s+carbon\s+emission\w*": 1.5,
    r"emission\s+reduction": 1.5,
    r"co\s*2\s+emission\w*": 1.5,
    r"carbon\s+dioxide\s+reduction": 1.5,
    r"greenhouse\s+gas": 1.0,

    # Energy efficiency
    r"energy\s+saving": 1.0,
    r"energy\s+conservation": 1.0,
    r"reduce\s+energy\s+consumption": 1.5,
    r"low\s+energy\s+consumption": 1.0,
    r"thermal\s+efficiency": 1.0,
    r"heat\s+efficiency": 1.0,
    r"fuel\s+consumption": 1.0,
    r"reduce\s+fuel\s+consumption": 1.5,
    r"coke\s+rate": 1.5,
    r"reduce\s+coke\s+rate": 1.5,
    r"reducing\s+material\s+required": 1.5,
    r"amount\s+of\s+reducing\s+agent": 1.0,

    # Heat / power
    r"waste\s+heat": 1.0,
    r"exhaust\s+heat": 1.0,
    r"residual\s+heat": 1.0,
    r"residual\s+energy": 1.0,
    r"power\s+generation": 1.0,
    r"steam\s+generat\w*": 1.0,

    # Gas recovery / use
    r"gas\s+recovery": 1.5,
    r"gas\s+recycling": 1.5,
    r"gas\s+recirculation": 1.5,
    r"recirculating\s+gas": 1.5,
    r"off[-\s]?gas.*(recover|recycl|recirculat|utili[sz]|fuel|heat|steam|power|co\s*2|carbon)": 1.5,
    r"top\s+gas.*(recover|recycl|recirculat|utili[sz]|fuel|heat|steam|power|co\s*2|carbon)": 1.5,
    r"blast\s+furnace\s+gas": 1.0,
    r"converter\s+gas": 1.0,
    r"process\s+gas": 1.0,
    r"export\s+gas": 1.0,
    r"gas[-\s]?conducting\s+system": 1.0,
    r"gas\s+turbine": 1.0,

    # Resource circularity
    r"recycl\w*": 1.0,
    r"reuse": 1.0,
    r"resource\s+utili[sz]\w*": 1.0,
    r"solid\s+waste": 1.0,
    r"slag\s+utili[sz]\w*": 1.0,
    r"slag\s+recycl\w*": 1.0,
    r"dust\s+recycl\w*": 1.0,
    r"sludge\s+recycl\w*": 1.0,
    r"waste\s+residue": 1.0,
    r"waste\s+material": 1.0,
    r"recover\w*\s+(iron|valuable\s+metal|valuable\s+components).*slag": 1.5,
}

STEEL_CONTEXT_TERMS = {
    r"blast\s+furnace": 0.5,
    r"converter": 0.5,
    r"basic\s+oxygen\s+furnace": 0.5,
    r"electric\s+arc\s+furnace": 0.5,
    r"\bEAF\b": 0.5,
    r"ladle\s+furnace": 0.5,
    r"\bLF\b": 0.5,
    r"\bRH\b": 0.5,
    r"direct\s+reduction": 0.5,
    r"shaft\s+furnace": 0.5,
    r"fluidi[sz]ed\s+bed": 0.5,
    r"steelmaking": 0.5,
    r"ironmaking": 0.5,
    r"pig\s+iron": 0.5,
    r"molten\s+iron": 0.5,
    r"molten\s+steel": 0.5,
    r"steel\s+slag": 0.5,
    r"converter\s+slag": 0.5,
    r"blast\s+furnace\s+slag": 0.5,
    r"steelworks": 0.5,
    r"steel\s+mill": 0.5,
    r"metallurgical\s+plant": 0.5,
}

ACTION_TERMS = {
    r"recover\w*": 0.5,
    r"recycl\w*": 0.5,
    r"reuse": 0.5,
    r"utili[sz]\w*": 0.5,
    r"recirculat\w*": 0.5,
    r"separat\w*": 0.5,
    r"captur\w*": 0.5,
    r"remov\w*": 0.5,
    r"reduc\w*": 0.5,
    r"preheat\w*": 0.5,
    r"optimi[sz]\w*": 0.5,
    r"save\w*": 0.5,
    r"generat\w*": 0.5,
    r"substitut\w*": 0.5,
    r"replace\w*": 0.5,
}

NOISE_TERMS = {
    # Steel grade / product quality
    r"low[-\s]?carbon\s+steel": -2.0,
    r"carbon\s+content": -1.5,
    r"bearing\s+steel": -2.0,
    r"die\s+steel": -2.0,
    r"tool\s+steel": -2.0,
    r"pipeline\s+steel": -1.5,
    r"weathering\s+steel": -1.5,
    r"stainless\s+steel": -1.5,
    r"ductile\s+iron": -1.5,
    r"cast\s+iron": -1.5,
    r"gear\s+steel": -1.5,
    r"spring\s+steel": -1.5,
    r"rail\s+steel": -1.5,
    r"steel\s+plate": -1.0,
    r"steel\s+strip": -1.0,
    r"mechanical\s+propert\w*": -2.0,
    r"impact\s+toughness": -2.0,
    r"wear\s+resistance": -2.0,
    r"yield\s+strength": -2.0,
    r"tensile\s+strength": -2.0,
    r"surface\s+quality": -1.5,

    # Refining quality control
    r"inclusion\s+control": -2.0,
    r"inclusion\s+morphology": -2.0,
    r"cleanliness\s+of\s+molten\s+steel": -1.5,
    r"desulfurization\s+rate": -1.5,
    r"desulfurizing\s+steel": -1.5,
    r"deoxidation": -1.5,
    r"calcium\s+treatment": -1.5,
    r"nitrogen\s+content": -2.0,
    r"hydrogen\s+content": -2.0,
    r"oxygen\s+content": -1.5,
    r"low\s+nitrogen": -2.0,
    r"low\s+hydrogen": -2.0,
    r"ultra[-\s]?low\s+sulfur": -1.5,
    r"molten\s+steel\s+purity": -1.5,

    # Equipment / measurement / maintenance
    r"oxygen\s+lance": -2.0,
    r"sublance": -2.0,
    r"measuring\s+probe": -2.0,
    r"temperature\s+measuring\s+gun": -2.0,
    r"replacement\s+device": -2.0,
    r"maintenance": -2.0,
    r"repair": -2.0,
    r"construction\s+method": -2.0,
    r"dismantl\w*": -2.0,
    r"cooling\s+wall": -2.0,
    r"tuyere": -1.5,
    r"material\s+level\s+detection": -2.0,
    r"ladle\s+maintenance": -2.0,
    r"furnace\s+lining": -1.5,

    # Non-steel or peripheral mineral / non-ferrous context
    r"alumina": -2.0,
    r"magnesium\s+metal": -2.0,
    r"nickel\s+oxide\s+ore": -2.0,
    r"copper\s+anode": -2.0,
    r"zinc\s+sulfate": -2.0,
    r"lead\s+slag": -1.5,
    r"ilmenite": -1.5,
    r"titanium\s+oxide": -1.5,
}


# =============================================================================
# 3. Caps / Thresholds
# =============================================================================

CAPS = {
    "high": 4.0,
    "decarbon": 3.0,
    "steel": 1.0,
    "action": 1.0,
    "noise_penalty": -3.0,
}

THRESHOLDS = {
    "related": 4.0,
    "weak_related": 2.5,
}


# =============================================================================
# 4. False Negative Audit Terms
# =============================================================================

FALSE_NEGATIVE_AUDIT_TERMS = {
    # CO2 / CCUS / carbon-negative
    r"co\s*2\s+captur\w*": "CO2_CCUS",
    r"co\s*2\s+remov\w*": "CO2_CCUS",
    r"co\s*2\s+emission\w*": "CO2_CCUS",
    r"carbon\s+dioxide\s+captur\w*": "CO2_CCUS",
    r"carbon\s+dioxide\s+remov\w*": "CO2_CCUS",
    r"carbon\s+dioxide\s+reduction": "CO2_CCUS",
    r"carbon[-\s]?negative": "CO2_CCUS",
    r"reducing\s+co\s*2\s+emissions": "CO2_CCUS",
    r"method\s+for\s+reducing\s+co\s*2\s+emissions": "CO2_CCUS",

    # DRI / hydrogen / reducing gas
    r"direct\s+reduced\s+iron": "DRI_HYDROGEN_REDUCTION",
    r"\bDRI\b": "DRI_HYDROGEN_REDUCTION",
    r"fluidi[sz]ed\s+bed\s+direct\s+reduction": "DRI_HYDROGEN_REDUCTION",
    r"hydrogen[-\s]?rich": "DRI_HYDROGEN_REDUCTION",
    r"hydrogen\s+concentration": "DRI_HYDROGEN_REDUCTION",
    r"reducing\s+gas.*direct\s+reduction": "DRI_HYDROGEN_REDUCTION",
    r"top\s+gas.*direct\s+reduction": "DRI_HYDROGEN_REDUCTION",
    r"exhaust\s+gas.*direct\s+reduction": "DRI_HYDROGEN_REDUCTION",

    # Gas recovery / process gas use
    r"blast\s+furnace\s+gas": "GAS_RECOVERY_UTILIZATION",
    r"top\s+gas\s+recirculat\w*": "GAS_RECOVERY_UTILIZATION",
    r"converter\s+gas": "GAS_RECOVERY_UTILIZATION",
    r"off[-\s]?gas.*fuel\s+gas": "GAS_RECOVERY_UTILIZATION",
    r"process\s+gas": "GAS_RECOVERY_UTILIZATION",
    r"export\s+gas": "GAS_RECOVERY_UTILIZATION",
    r"gas[-\s]?conducting\s+system": "GAS_RECOVERY_UTILIZATION",
    r"gas\s+turbine": "GAS_RECOVERY_UTILIZATION",

    # Waste heat / energy
    r"waste\s+heat": "WASTE_HEAT_ENERGY",
    r"heat\s+recovery": "WASTE_HEAT_ENERGY",
    r"residual\s+energy\s+recover\w*": "WASTE_HEAT_ENERGY",
    r"steam\s+generat\w*": "WASTE_HEAT_ENERGY",
    r"minimum\s+energy\s+utilization": "WASTE_HEAT_ENERGY",
    r"low\s+energy\s+consumption": "WASTE_HEAT_ENERGY",

    # Slag / solid waste circularity
    r"steelworks\s+waste": "SLAG_SOLID_WASTE_RECYCLING",
    r"metallurgical\s+solid\s+waste": "SLAG_SOLID_WASTE_RECYCLING",
    r"metallurgical\s+waste\s+powder": "SLAG_SOLID_WASTE_RECYCLING",
    r"steelmaking\s+slag": "SLAG_SOLID_WASTE_RECYCLING",
    r"steelmaking\s+revert": "SLAG_SOLID_WASTE_RECYCLING",
    r"slag.*sludge.*recycl": "SLAG_SOLID_WASTE_RECYCLING",
    r"recover\w*\s+(iron|valuable\s+metal|valuable\s+components).*slag": "SLAG_SOLID_WASTE_RECYCLING",

    # EAF / scrap
    r"electric\s+arc\s+furnace.*minimum\s+energy": "EAF_SCRAP_ENERGY",
    r"minimum\s+energy\s+utilization.*electric\s+arc\s+furnace": "EAF_SCRAP_ENERGY",
    r"electric\s+arc\s+furnace.*scrap.*(preheat|energy|heat|recover|recycl|efficien|saving)": "EAF_SCRAP_ENERGY",
    r"scrap\s+preheat\w*": "EAF_SCRAP_ENERGY",
    r"scrap\s+steel\s+recycl\w*": "EAF_SCRAP_ENERGY",

    # Biomass / biochar
    r"biomass\s+direct\s+reduced\s+iron": "BIOMASS_BIOCHAR_REDUCTANT",
    r"biochar": "BIOMASS_BIOCHAR_REDUCTANT",
    r"renewable\s+carbon": "BIOMASS_BIOCHAR_REDUCTANT",
    r"bio[-\s]?reduction": "BIOMASS_BIOCHAR_REDUCTANT",
    r"coffee\s+as\s+a\s+carbon\s+source": "BIOMASS_BIOCHAR_REDUCTANT",

    # Electrolysis
    r"molten\s+oxide\s+electrolysis": "ELECTROLYSIS_LOW_CARBON",
    r"electrolysis\s+of\s+carbon\s+dioxide": "ELECTROLYSIS_LOW_CARBON",
    r"electrochemically\s+reduc\w*.*iron": "ELECTROLYSIS_LOW_CARBON",
}


# =============================================================================
# 5. Utility Functions
# =============================================================================

def normalize_text(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\u00a0", " ")
    text = text.replace("₂", "2")
    text = text.replace("CO₂", "CO2")
    text = text.replace("co₂", "co2")
    text = re.sub(r"&[#a-zA-Z0-9]+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def load_json_records(path):
    """
    支援兩種格式：
    1. JSON array: [{...}, {...}]
    2. JSONL: 每行一筆 JSON
    """
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON root is not a list.")
            return data

        records = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records


def matched_prefixes(cpc_codes, weight_dict):
    hits = []
    if not cpc_codes:
        return hits

    for code in cpc_codes:
        if not code:
            continue
        code = str(code).strip()
        for prefix, weight in weight_dict.items():
            if code.startswith(prefix):
                hits.append((code, prefix, weight))
    return hits


def max_cpc_score(cpc_codes):
    core_hits = matched_prefixes(cpc_codes, CORE_CPC_WEIGHTS)
    support_hits = matched_prefixes(cpc_codes, SUPPORT_CPC_WEIGHTS)
    peripheral_hits = matched_prefixes(cpc_codes, PERIPHERAL_CPC_WEIGHTS)

    all_hits = core_hits + support_hits + peripheral_hits

    if not all_hits:
        return {
            "cpc_score": 0.0,
            "cpc_strength": "none",
            "core_hits": [],
            "support_hits": [],
            "peripheral_hits": [],
        }

    max_score = max(hit[2] for hit in all_hits)

    if any(hit[2] == max_score for hit in core_hits):
        strength = "core"
    elif any(hit[2] == max_score for hit in support_hits):
        strength = "support"
    elif any(hit[2] == max_score for hit in peripheral_hits):
        strength = "peripheral"
    else:
        strength = "none"

    return {
        "cpc_score": max_score,
        "cpc_strength": strength,
        "core_hits": sorted(set([h[0] for h in core_hits])),
        "support_hits": sorted(set([h[0] for h in support_hits])),
        "peripheral_hits": sorted(set([h[0] for h in peripheral_hits])),
    }


def score_patterns(text, pattern_weight_dict, cap=None):
    hits = []
    score = 0.0

    for pattern, weight in pattern_weight_dict.items():
        try:
            if re.search(pattern, text, flags=re.IGNORECASE):
                hits.append(pattern)
                score += weight
        except re.error as e:
            print(f"[Regex error] pattern={pattern}, error={e}")

    if cap is not None:
        score = min(score, cap)

    return score, sorted(set(hits))


def score_noise(original_text):
    hits = []
    penalty = 0.0

    for pattern, weight in NOISE_TERMS.items():
        try:
            if re.search(pattern, original_text, flags=re.IGNORECASE):
                hits.append(pattern)
                penalty += weight
        except re.error as e:
            print(f"[Regex error] noise pattern={pattern}, error={e}")

    # penalty 是負數，所以用 max 限制最低只扣到 -3
    penalty = max(penalty, CAPS["noise_penalty"])

    return penalty, sorted(set(hits))


def audit_false_negative(record, text):
    audit_hits = []
    audit_categories = []

    for pattern, category in FALSE_NEGATIVE_AUDIT_TERMS.items():
        try:
            if re.search(pattern, text, flags=re.IGNORECASE):
                audit_hits.append(pattern)
                audit_categories.append(category)
        except re.error as e:
            print(f"[Regex error] audit pattern={pattern}, error={e}")

    audit_hits = sorted(set(audit_hits))
    audit_categories = sorted(set(audit_categories))

    # 高信心 audit：至少命中一個 audit 類別，且不是明顯產品品質/量測/維修雜訊主軸
    high_confidence = bool(audit_hits)

    return {
        "audit_high_confidence": high_confidence,
        "audit_hits": audit_hits,
        "audit_categories": audit_categories,
    }


def classify_record(record):
    title = normalize_text(record.get("title_en", ""))
    abstract = normalize_text(record.get("abstract_en", ""))
    text = f"{title} {abstract}".strip()

    cpc_codes = record.get("matched_cpc_codes", [])
    if cpc_codes is None:
        cpc_codes = []

    cpc_info = max_cpc_score(cpc_codes)

    high_score, high_hits = score_patterns(
        text, HIGH_RELEVANCE_TERMS, CAPS["high"]
    )

    decarb_score, decarb_hits = score_patterns(
        text, DECARB_INTENT_TERMS, CAPS["decarbon"]
    )

    steel_score, steel_hits = score_patterns(
        text, STEEL_CONTEXT_TERMS, CAPS["steel"]
    )

    action_score, action_hits = score_patterns(
        text, ACTION_TERMS, CAPS["action"]
    )

    noise_penalty, noise_hits = score_noise(text)

    # 免死金牌：如果高相關語境很強，不讓 noise penalty 誤殺
    if len(high_hits) >= 2 or (high_score >= 3.0 and decarb_score >= 1.5):
        adjusted_noise_penalty = 0.0
        noise_adjusted = True
    else:
        adjusted_noise_penalty = noise_penalty
        noise_adjusted = False

    total_score = (
        cpc_info["cpc_score"]
        + high_score
        + decarb_score
        + steel_score
        + action_score
        + adjusted_noise_penalty
    )

    has_decarbon_context = bool(high_hits or decarb_hits)

    if not has_decarbon_context:
        relevance_label = "unrelated"
        relevance_rule = "Rule_0_no_decarbon_context"
    elif total_score >= THRESHOLDS["related"]:
        relevance_label = "related"
        relevance_rule = "Rule_1_score_related"
    elif total_score >= THRESHOLDS["weak_related"]:
        relevance_label = "weak_related"
        relevance_rule = "Rule_2_score_weak_related"
    else:
        relevance_label = "unrelated"
        relevance_rule = "Rule_3_score_below_threshold"

    audit_info = {
        "audit_high_confidence": False,
        "audit_hits": [],
        "audit_categories": [],
    }

    final_label = relevance_label

    if relevance_label == "unrelated":
        audit_info = audit_false_negative(record, text)

        if audit_info["audit_high_confidence"]:
            final_label = "weak_related_audit"
        else:
            final_label = "unrelated"

    result = dict(record)

    result.update({
        "relevance_label": relevance_label,
        "final_label": final_label,
        "relevance_rule": relevance_rule,

        "total_score": round(total_score, 3),
        "cpc_score": round(cpc_info["cpc_score"], 3),
        "high_score": round(high_score, 3),
        "decarbon_score": round(decarb_score, 3),
        "steel_score": round(steel_score, 3),
        "action_score": round(action_score, 3),
        "noise_penalty": round(noise_penalty, 3),
        "adjusted_noise_penalty": round(adjusted_noise_penalty, 3),
        "noise_adjusted": noise_adjusted,

        "has_decarbon_context": has_decarbon_context,
        "cpc_strength": cpc_info["cpc_strength"],

        "core_cpc_hits": cpc_info["core_hits"],
        "support_cpc_hits": cpc_info["support_hits"],
        "peripheral_cpc_hits": cpc_info["peripheral_hits"],

        "high_hits": high_hits,
        "decarbon_hits": decarb_hits,
        "steel_hits": steel_hits,
        "action_hits": action_hits,
        "noise_hits": noise_hits,

        "audit_high_confidence": audit_info["audit_high_confidence"],
        "audit_categories": audit_info["audit_categories"],
        "audit_hits": audit_info["audit_hits"],
    })

    return result


def make_json_safe(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    return value


def write_json(path, rows):
    if not rows:
        print(f"[WARN] No rows to write: {path}")
        path.write_text("[]\n", encoding="utf-8")
        return

    # 統一欄位順序：核心欄位放前面，其餘欄位放後面
    preferred_cols = [
        "publication_number",
        "application_number",
        "country_code",
        "kind_code",
        "priority_date",
        "title_en",
        "abstract_en",
        "matched_cpc_codes",

        "relevance_label",
        "final_label",
        "relevance_rule",
        "total_score",
        "cpc_score",
        "high_score",
        "decarbon_score",
        "steel_score",
        "action_score",
        "noise_penalty",
        "adjusted_noise_penalty",
        "noise_adjusted",
        "has_decarbon_context",
        "cpc_strength",

        "core_cpc_hits",
        "support_cpc_hits",
        "peripheral_cpc_hits",
        "high_hits",
        "decarbon_hits",
        "steel_hits",
        "action_hits",
        "noise_hits",
        "audit_high_confidence",
        "audit_categories",
        "audit_hits",
    ]

    all_cols = []
    seen = set()

    for col in preferred_cols:
        if any(col in r for r in rows):
            all_cols.append(col)
            seen.add(col)

    for r in rows:
        for col in r.keys():
            if col not in seen:
                all_cols.append(col)
                seen.add(col)

    ordered_rows = []
    for r in rows:
        ordered_rows.append({col: make_json_safe(r.get(col)) for col in all_cols})

    path.write_text(
        json.dumps(ordered_rows, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_summary(path, classified_rows):
    final_counter = Counter(r["final_label"] for r in classified_rows)
    raw_counter = Counter(r["relevance_label"] for r in classified_rows)
    cpc_counter = Counter(r["cpc_strength"] for r in classified_rows)

    summary_rows = []

    total = len(classified_rows)

    for label, count in final_counter.most_common():
        summary_rows.append({
            "summary_type": "final_label",
            "category": label,
            "count": count,
            "share": round(count / total, 6) if total else 0,
        })

    for label, count in raw_counter.most_common():
        summary_rows.append({
            "summary_type": "raw_relevance_label",
            "category": label,
            "count": count,
            "share": round(count / total, 6) if total else 0,
        })

    for label, count in cpc_counter.most_common():
        summary_rows.append({
            "summary_type": "cpc_strength",
            "category": label,
            "count": count,
            "share": round(count / total, 6) if total else 0,
        })

    # audit 類別統計
    audit_category_counter = Counter()
    for r in classified_rows:
        raw_cats = r.get("audit_categories", [])
        if isinstance(raw_cats, str):
            cats = raw_cats.split(";")
        else:
            cats = raw_cats
        for c in cats:
            c = c.strip()
            if c:
                audit_category_counter[c] += 1

    for label, count in audit_category_counter.most_common():
        summary_rows.append({
            "summary_type": "audit_category",
            "category": label,
            "count": count,
            "share": round(count / total, 6) if total else 0,
        })

    write_json(path, summary_rows)


# =============================================================================
# 6. Main
# =============================================================================

def main():
    print(f"[INFO] Input: {INPUT_PATH}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    records = load_json_records(INPUT_PATH)
    print(f"[INFO] Loaded records: {len(records):,}")

    classified_rows = []

    for i, record in enumerate(records, start=1):
        classified_rows.append(classify_record(record))

        if i % 1000 == 0:
            print(f"[INFO] Processed {i:,}/{len(records):,}")

    related_rows = [
        r for r in classified_rows
        if r["final_label"] in {"related", "weak_related", "weak_related_audit"}
    ]

    unrelated_rows = [
        r for r in classified_rows
        if r["final_label"] == "unrelated"
    ]

    audit_rows = [
        r for r in classified_rows
        if r["final_label"] == "weak_related_audit"
    ]

    write_json(OUT_FULL, classified_rows)
    write_json(OUT_RELATED, related_rows)
    write_json(OUT_UNRELATED, unrelated_rows)
    write_json(OUT_AUDIT, audit_rows)
    write_summary(OUT_SUMMARY, classified_rows)

    print("\n[DONE] Weighted relevance filtering completed.")
    print(f"[INFO] Total records: {len(classified_rows):,}")

    final_counts = Counter(r["final_label"] for r in classified_rows)
    for label, count in final_counts.most_common():
        share = count / len(classified_rows) * 100 if classified_rows else 0
        print(f"  - {label}: {count:,} ({share:.2f}%)")

    print("\n[OUTPUT FILES]")
    print(f"  Full review: {OUT_FULL}")
    print(f"  Related patents: {OUT_RELATED}")
    print(f"  Unrelated patents: {OUT_UNRELATED}")
    print(f"  Audit recovered: {OUT_AUDIT}")
    print(f"  Summary: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
