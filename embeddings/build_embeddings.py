import numpy as np
from sentence_transformers import SentenceTransformer
from data.load_lenta import load_lenta_texts
from data.chunking import chunk_text

def build_embeddings(path, limit=2000, max_chars=900):
    docs = load_lenta_texts(path, limit=limit)

    chunks = []
    for doc_id, doc in enumerate(docs):
        for ch in chunk_text(doc, max_chars=max_chars):
            chunks.append((doc_id, ch))

    texts = [ch for (_, ch) in chunks]

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return chunks, emb.astype(np.float32)

if __name__ == "__main__":
    chunks, emb = build_embeddings("data/lenta.csv", limit=2000)
    print("Chunks:", len(chunks))
    print("Embeddings shape:", emb.shape)
