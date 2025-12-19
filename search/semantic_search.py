import pickle
import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer

# INDEX_PATH = "index/hnsw.index"
# META_PATH = "index/meta.pkl"
INDEX_PATH = "index/hnsw_5000.index"
META_PATH = "index/meta_5000.pkl"

def load_index(dim):
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(INDEX_PATH)
    index.set_ef(50)
    return index

def load_chunks():
    with open(META_PATH, "rb") as f:
        return pickle.load(f)  # list of (doc_id, chunk_text)

def search(query, top_k=5):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    chunks = load_chunks()
    dim = 384
    index = load_index(dim)

    q = model.encode([query], normalize_embeddings=True).astype(np.float32)
    labels, distances = index.knn_query(q, k=top_k)

    results = []
    for idx, dist in zip(labels[0], distances[0]):
        doc_id, chunk_text = chunks[idx]
        results.append((1 - dist, doc_id, chunk_text))
    return results

if __name__ == "__main__":
    query = "экономические последствия санкций"
    results = search(query, top_k=5)
    print("Запрос:", query)
    print("Результаты:")
    for score, doc_id, text in results:
        print(f"{score:.4f} | doc={doc_id} | {text[:220]}...")
