import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# === CONFIG ===
VAULT_FILE = "graph_schema.txt"  # <-- your final data file
CHUNK_SIZE = 300  # characters per chunk
EMBEDDING_FILE = "embeddings.npy"
CHUNK_INDEX_FILE = "chunks.json"

model = SentenceTransformer("thenlper/gte-large")

# === Chunk the document ===
def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# === Main ===
with open(VAULT_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()

chunks = chunk_text(full_text)
embeddings = model.encode(chunks, convert_to_numpy=True)

# === Save ===
np.save(EMBEDDING_FILE, embeddings)
with open(CHUNK_INDEX_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f)

print(f"✅ Saved {len(chunks)} chunks & embeddings")
