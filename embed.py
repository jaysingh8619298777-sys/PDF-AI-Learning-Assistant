from sentence_transformers import SentenceTransformer

with open("knowledge_base.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split into chunks
chunks = [c.strip() for c in text.split("\n") if c.strip()]

print("Chunks:", len(chunks))

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("Embedding dimension:", len(embeddings[0]))
print("Embeddings created successfully!")

