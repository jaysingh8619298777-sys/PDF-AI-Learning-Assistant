import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ----------------------------
# Load Knowledge Base
# ----------------------------
with open("knowledge_base.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text into chunks
chunks = []
chunks = text.split("\n\n")
chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]

# ----------------------------
# Load Embedding Model
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create Embeddings
embeddings = model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

# ----------------------------
# Create FAISS Index
# ----------------------------
index = faiss.IndexFlatL2(384)
index.add(embeddings)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("📚 WhatsApp Group AI Assistant")
st.caption("Ask questions from PDFs, notebooks, and project files shared in the WhatsApp group.")
st.write(
    "Ask questions from PDFs shared in the WhatsApp group."
)

# Upload Section
uploaded_file = st.file_uploader(
    "📂 Upload a New PDF",
    type=["pdf"]
)

if uploaded_file:
    st.success("✅ PDF uploaded successfully!")

# Question Section
query = st.text_input(
    "💬 Ask a Question"
)

if query:

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k=1
    )

    answer = chunks[indices[0][0]]

    # Summary Section
    st.markdown("### 🎯 Summary")

    summary = answer[:200]
    st.info(summary)

    # Full Answer Section
    st.markdown("### 📖 Retrieved Answer")
    st.text_area("Answer", answer[:800], height=250)
        