# TODO - Record-based Legal RAG Upgrade

## Step 1: chunking.py
- [ ] Remove pure character/sliding-window chunking for legal records; rely on extracted legal records as single chunks.
- [ ] Ensure chunk metadata matches required schema: chunk_id, source_document, ipc_section, bns_section, crime_name, description, punishment, text.
- [ ] Normalize section numbering fields consistently (store as digits/letters only).

## Step 2: vector_store.py
- [ ] Update metadata validation keys to required schema.
- [ ] Implement normalize_section() and robust exact searches over ipc_section, bns_section, crime_name.
- [ ] Keep existing FAISS semantic search working and persist metadata.pkl.

## Step 3: retriever.py
- [ ] Implement hybrid retrieval: exact ipc, exact bns, exact crime name + FAISS + BM25.
- [ ] Merge scores with heavy boosts for exact matches.
- [ ] Ensure outputs include fields used downstream.

## Step 4: context_builder.py
- [ ] Build structured context blocks (IPC Section, Crime, Description, Punishment, BNS Section, Source).
- [ ] Deduplicate and keep only top 3-5 chunks.

## Step 5: enhanced_rag_engine.py
- [ ] Add debugging prints: retrieved chunks list + context before calling Ollama.
- [ ] Ensure final prompt receives structured context.

## Step 6: Rebuild index + tests
- [ ] Rebuild FAISS index + metadata.pkl using existing ingestion pipeline.
- [ ] Run pytest.

