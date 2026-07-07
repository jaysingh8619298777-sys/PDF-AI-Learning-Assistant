import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load knowledge base
with open("knowledge_base.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
with open("knowledge_base.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = []

chunk_size = 500

for i in range(0, len(text), 2500):
    chunks.append(text[i:i + 2500])

print("Total Chunks:", len(chunks))

print("Chunks loaded:", len(chunks))

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(chunks)

# Convert to numpy
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(384)
index.add(embeddings)

print("Total chunks:", len(chunks))
print("First chunk:")
print(repr(chunks[0]))
print("\nChunk 10:")
print(chunks[10])

print("\nChunk 20:")
print(chunks[20])

for i in range(5):
    print(f"\n--- Chunk {i} ---")
    print(chunks[i])

print("FAISS vectors:", index.ntotal)
while True:

    query = input("\nAsk Question: ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
   
    distances, indices = index.search(query_embedding, k=1)
    answer = chunks[indices[0][0]]

    print("\nAnswer:\n")
    print(answer)
    print("\n" + "-" * 50)