import pickle

data = pickle.load(open('models/metadata.pkl', 'rb'))
print(f"Total chunks: {len(data)}")
print()

for i, d in enumerate(data):
    sec = d.get("section_number", "")
    crime = d.get("crime_name", "")
    src = d.get("source_document", "")
    text_preview = d.get("text", "")[:100].replace("\n", " ")
    print(f"[{i}] section={sec} | crime={crime} | src={src}")
    print(f"    text: {text_preview}")
    print()
