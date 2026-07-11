"""
End-to-end pipeline diagnostic.
Traces every stage and prints output so we can find exactly where it fails.
"""
import sys, os
sys.path.insert(0, os.getcwd())

print("=" * 60)
print("STAGE 0: Load FAISS index and metadata")
print("=" * 60)
from src.vector_store import VectorStore
vs = VectorStore(index_path="models/faiss_index.bin", metadata_path="models/metadata.pkl")
vs.load_index()
print(f"  Index loaded: {vs.index.ntotal} vectors")
all_chunks = vs.get_all_chunks()
print(f"  Metadata loaded: {len(all_chunks)} chunks")
print()

print("=" * 60)
print("STAGE 1: Retriever — retrieve chunks for 'What is Culpable Homicide'")
print("=" * 60)
from src.embedding_generator import EmbeddingGenerator
from src.retriever import Retriever
embedder = EmbeddingGenerator("all-MiniLM-L6-v2")
retriever = Retriever(vs, embedder)
query = "What is Culpable Homicide"
chunks = retriever.retrieve(query, top_k=20)
print(f"  Retrieved {len(chunks)} chunks")
for i, c in enumerate(chunks[:5]):
    print(f"  [{i}] sim={c.get('similarity',0):.3f} section={c.get('section_number','')} crime={c.get('crime_name','')} text={c.get('text','')[:80]}")
print()

print("=" * 60)
print("STAGE 2: Cross-Encoder Reranking")
print("=" * 60)
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs = [[query, c.get("text", "")] for c in chunks]
scores = cross_encoder.predict(pairs)
for i, score in enumerate(scores):
    chunks[i]["cross_score"] = float(score)
chunks.sort(key=lambda x: x.get("cross_score", -999), reverse=True)
valid_chunks = chunks[:5]
print(f"  Top 5 after reranking:")
for i, c in enumerate(valid_chunks):
    print(f"  [{i}] cross_score={c.get('cross_score',0):.3f} sim={c.get('similarity',0):.3f} section={c.get('section_number','')} crime={c.get('crime_name','')}")
print()

print("=" * 60)
print("STAGE 3: Context Builder")
print("=" * 60)
from src.context_builder import ContextBuilder
cb = ContextBuilder()
context = cb.build(valid_chunks)
print(f"  Context length: {len(context)} chars")
print(f"  Context preview (first 500 chars):")
print(context[:500])
print()

print("=" * 60)
print("STAGE 4: Confidence calculation")
print("=" * 60)
confidence = max([c.get("similarity", 0) for c in valid_chunks]) if valid_chunks else 0
confidence_label = "High" if confidence >= 0.70 else "Medium" if confidence >= 0.50 else "Low"
print(f"  Max similarity: {confidence:.3f}")
print(f"  Confidence label: {confidence_label}")
if confidence_label == "Low":
    print("  *** THIS IS THE PROBLEM! Confidence is Low so HallucinationGuard will reject everything! ***")
print()

print("=" * 60)
print("STAGE 5: Ollama Generation")
print("=" * 60)
from src.ollama_generator import OllamaGenerator
gen = OllamaGenerator(model_name="qwen3:0.6b")
answer = ""
for chunk in gen.generate_answer(query, context, history=""):
    print(chunk, end="", flush=True)
    answer += chunk
print()
print(f"\n  Raw answer length: {len(answer)} chars")
print()

print("=" * 60)
print("STAGE 6: Hallucination Guard")
print("=" * 60)
from src.hallucination_guard import HallucinationGuard
guard = HallucinationGuard()
guard_result = guard.guard(answer, confidence_label, valid_chunks)
print(f"  is_grounded: {guard_result.is_grounded}")
print(f"  final_answer preview: {guard_result.final_answer[:200]}")
print()

print("=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
