"""Diagnostic: check FAISS index health and concept coverage."""
import pickle
import faiss
import sys
sys.stdout.reconfigure(encoding='utf-8')

data = pickle.load(open('models/metadata.pkl', 'rb'))
idx = faiss.read_index('models/faiss_index.bin')

print(f"Total metadata records: {len(data)}")
print(f"FAISS vectors: {idx.ntotal}")
print(f"FAISS dimension: {idx.d}")
print(f"Match: {len(data) == idx.ntotal}")

# Analyze record quality
srcs = set()
ipc_empty = 0
bns_empty = 0
crime_empty = 0
uuid_ids = 0
int_ids = 0

for d in data:
    srcs.add(d.get('source_document', '?'))
    if not d.get('ipc_section', '') and not d.get('section_number', ''):
        ipc_empty += 1
    if not d.get('bns_section', ''):
        bns_empty += 1
    if not d.get('crime_name', ''):
        crime_empty += 1
    cid = d.get('chunk_id', '')
    if isinstance(cid, str) and '-' in cid:
        uuid_ids += 1
    elif isinstance(cid, int):
        int_ids += 1

print(f"\nSources: {srcs}")
print(f"IPC/Section empty: {ipc_empty}/{len(data)}")
print(f"BNS empty: {bns_empty}/{len(data)}")
print(f"Crime name empty: {crime_empty}/{len(data)}")
print(f"UUID chunk_ids: {uuid_ids}")
print(f"Int chunk_ids: {int_ids}")

# Concept coverage
print("\n--- CONCEPT COVERAGE ---")
concepts = ['murder', 'culpable homicide', 'criminal breach of trust',
            'dishonest misappropriation', 'forgery', 'extortion',
            'cheating', 'theft', 'robbery', 'kidnapping']
for concept in concepts:
    found = False
    for d in data:
        text = str(d.get('text', '')).lower()
        crime = str(d.get('crime_name', '')).lower()
        if concept in text or concept in crime:
            found = True
            break
    print(f"  {concept}: {'FOUND' if found else 'MISSING'}")
