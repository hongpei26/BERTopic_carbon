import json
import re
import random
from collections import Counter

src = "/home/carbon/carbon/data_globalmorecpc/global_onlycpc_domain_target_intersection.json"

# We will re-run the classification logic locally to find the unmatched ones
# so we don't have to load two big files.

EXCLUSION_TERMS = [r"low[-\s]carbon steel", r"low[-\s]carbon equivalent", r"stainless steel", r"bearing steel", r"marine steel", r"high[-\s]strength steel", r"spring steel", r"gear steel", r"free[-\s]cutting steel"]
HIGH_RELEVANCE_TERMS = [r"direct reduction", r"direct reduced iron", r"dri\b", r"sponge iron", r"shaft furnace", r"hydrogen metallurgy", r"hydrogen[-\s]based", r"hydrogen reduction", r"electric arc furnace", r"arc furnace", r"\beaf\b", r"carbon capture", r"co2 capture", r"ccus\b", r"carbon storage", r"carbon sequestration", r"green steel", r"carbon footprint", r"carbon neutral", r"decarbonization", r"biochar", r"biomass", r"low[-\s]carbon fuel", r"low[-\s]carbon reducing agent", r"waste heat recovery", r"gas recovery", r"slag recycling", r"top gas recovery", r"full scrap", r"100% scrap", r"solar thermal steelmaking"]
MEDIUM_RELEVANCE_TERMS = [r"blast furnace", r"converter", r"smelting furnace", r"molten slag", r"converter slag", r"steelmaking dust", r"furnace dust", r"flue gas", r"waste gas", r"exhaust gas", r"off gas", r"waste heat", r"coke oven gas", r"furnace gas", r"scrap\b", r"steel scrap"]
ACTION_TERMS = [r"recovery", r"recycling", r"recycle", r"reuse", r"efficiency", r"reduce emissions?", r"reducing emissions?", r"energy saving", r"optimization", r"low[-\s]carbon", r"green\b"]

def compile_patterns(term_list):
    return [re.compile(term, re.IGNORECASE) for term in term_list]

exclude_patterns = compile_patterns(EXCLUSION_TERMS)
high_patterns = compile_patterns(HIGH_RELEVANCE_TERMS)
medium_patterns = compile_patterns(MEDIUM_RELEVANCE_TERMS)
action_patterns = compile_patterns(ACTION_TERMS)

def get_matched_terms(text, patterns, terms):
    return [term for pattern, term in zip(patterns, terms) if pattern.search(text)]

records = json.load(open(src, "r", encoding="utf-8"))

unmatched_records = []

for record in records:
    text = f"{record.get('title_en', '')} {record.get('abstract_en', '')}"
    
    high_hits = get_matched_terms(text, high_patterns, HIGH_RELEVANCE_TERMS)
    exclude_hits = get_matched_terms(text, exclude_patterns, EXCLUSION_TERMS)
    medium_hits = get_matched_terms(text, medium_patterns, MEDIUM_RELEVANCE_TERMS)
    action_hits = get_matched_terms(text, action_patterns, ACTION_TERMS)
    
    if exclude_hits and not high_hits:
        continue
    elif high_hits:
        continue
    elif medium_hits and action_hits:
        continue
    else:
        unmatched_records.append({"title": record.get('title_en', ''), "abstract": record.get('abstract_en', ''), "text": text})

print(f"Total unmatched records: {len(unmatched_records)}")

# Let's search for potential missed terms in unmatched records
potential_missed_terms = [
    "co2", "carbon dioxide", "emission", "energy consumption", 
    "slag", "dust", "h2", "hydrogen", "reducing gas", "syngas", "synthesis gas",
    "electric furnace", "reduction furnace", "iron ore", "pellet", "sinter",
    "energy", "heat", "environment", "pollution", "waste"
]

missed_patterns = [re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE) for term in potential_missed_terms]

missed_counter = Counter()
for r in unmatched_records:
    text = r['text']
    for pattern, term in zip(missed_patterns, potential_missed_terms):
        if pattern.search(text):
            missed_counter[term] += 1

with open("/home/carbon/carbon/unmatched_analysis.txt", "w", encoding="utf-8") as f:
    f.write(f"Total unmatched: {len(unmatched_records)}\n\n")
    f.write("Potential missed keyword frequencies in unmatched records:\n")
    for term, count in missed_counter.most_common():
        f.write(f"  {term}: {count}\n")
    
    f.write("\n\n--- Sample of 20 Unmatched Records ---\n")
    sample = random.sample(unmatched_records, min(20, len(unmatched_records)))
    for i, r in enumerate(sample):
        f.write(f"\n[{i+1}] Title: {r['title']}\nAbstract: {r['abstract']}\n")

print("Analysis written to /home/carbon/carbon/unmatched_analysis.txt")
