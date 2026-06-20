import numpy as np
from fastapi import FastAPI, Query

from runtime import load_runtime

app = FastAPI(title="Semantic Search API", version="1.0")

model, index, chunks = load_runtime()


@app.get("/search")
def search(
    q: str = Query(..., min_length=2, description="Текст поискового запроса"),
    top_k: int = Query(5, ge=1, le=20, description="Количество возвращаемых результатов (Top-K)"),
):
    q_emb = model.encode([q], normalize_embeddings=True).astype(np.float32)
    labels, distances = index.knn_query(q_emb, k=top_k)

    results = []
    for idx, dist in zip(labels[0], distances[0]):
        doc_id, chunk_text = chunks[idx]
        score = float(1.0 - dist)
        results.append(
            {
                "score": score,
                "doc_id": int(doc_id),
                "text": chunk_text[:400],
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"query": q, "results": results}