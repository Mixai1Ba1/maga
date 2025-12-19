import pickle
import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer
from data.load_lenta import load_lenta_texts

def build_index_full_docs(path, limit=2000):
    texts = load_lenta_texts(path, limit=limit)

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    emb = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype(np.float32)

    dim = emb.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(texts), ef_construction=200, M=16)
    index.add_items(emb, ids=np.arange(len(texts)))
    index.set_ef(50)

    return index, texts

def search_full_docs(query, index, texts, top_k=5):
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    q = model.encode([query], normalize_embeddings=True).astype(np.float32)

    labels, distances = index.knn_query(q, k=top_k)
    return [(1 - d, texts[i]) for i, d in zip(labels[0], distances[0])]

if __name__ == "__main__":
    query = "экономические последствия санкций"

    index, texts = build_index_full_docs("data/lenta.csv", limit=2000)
    results = search_full_docs(query, index, texts)

    print("Семантический поиск по ЦЕЛЫМ документам")
    print("Запрос:", query)
    for score, text in results:
        print(f"{score:.4f} | {text[:200]}...")
