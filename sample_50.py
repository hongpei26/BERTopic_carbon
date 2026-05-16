import json
import random

with open('/home/carbon/carbon/data_globalmorecpc/global_onlycpc_domain_target_intersection.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total records: {len(data)}")

# If data is a dict (like {id: {}}), we convert to list of values or items
if isinstance(data, dict):
    records = list(data.values())
else:
    records = data

# Filter out records without abstract_en if any
records_with_abstract = [r for r in records if r.get('abstract_en')]

# Sample 50
sample_size = min(50, len(records_with_abstract))
sampled = random.sample(records_with_abstract, sample_size)

# Write to a file for easy reading
output = []
for i, record in enumerate(sampled):
    title = record.get('title_en', 'No Title')
    abstract = record.get('abstract_en', 'No Abstract')
    output.append(f"[{i+1}] Title: {title}\nAbstract: {abstract}\n")

with open('/home/carbon/carbon/sampled_50_abstracts.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(output))

print("Done. Wrote to /home/carbon/carbon/sampled_50_abstracts.txt")
