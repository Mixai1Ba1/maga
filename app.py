import pickle
import hnswlib
import numpy as np
from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer

INDEX_PATH = "index/hnsw.index"
META_PATH = "index/meta.pkl"
DIM = 384

app = FastAPI(title="Semantic Search API", version="1.0")

# Глобальные объекты (инициализируются при старте сервиса)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

index = hnswlib.Index(space="cosine", dim=DIM)
index.load_index(INDEX_PATH)
index.set_ef(100)

with open(META_PATH, "rb") as f:
    chunks = pickle.load(f)  # list of (doc_id, chunk_text)


@app.get("/search")
def search(
    q: str = Query(..., min_length=2, description="Текст поискового запроса"),
    top_k: int = Query(5, ge=1, le=20, description="Количество возвращаемых результатов (Top-K)")
):
    q_emb = model.encode([q], normalize_embeddings=True).astype(np.float32)

    labels, distances = index.knn_query(q_emb, k=top_k)

    results = []
    for idx, dist in zip(labels[0], distances[0]):
        doc_id, text = chunks[int(idx)]
        results.append({
            "score": float(1 - dist),   # косинусная близость
            "doc_id": int(doc_id),
            "text": text[:400]          # ограничим длину ответа
        })

    return {"query": q, "top_k": top_k, "results": results}
