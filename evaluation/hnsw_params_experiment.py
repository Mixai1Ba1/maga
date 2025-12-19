import time
import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer

from data.load_lenta import load_lenta_texts
from data.chunking import chunk_text

def run_hnsw_test(M, ef, chunks, embeddings, query_emb):
    dim = embeddings.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)

    index.init_index(
        max_elements=len(chunks),
        ef_construction=200,
        M=M
    )
    index.add_items(embeddings, ids=np.arange(len(chunks)))
    index.set_ef(ef)

    t0 = time.time()
    labels, distances = index.knn_query(query_emb, k=5)
    search_time = time.time() - t0

    avg_dist = float(np.mean(distances[0]))
    return search_time, avg_dist

if __name__ == "__main__":
    query = "экономические последствия санкций"

    docs = load_lenta_texts("data/lenta.csv", limit=2000)
    chunks = []
    for d in docs:
        chunks.extend(chunk_text(d))

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype(np.float32)

    query_emb = model.encode(
        [query],
        normalize_embeddings=True
    ).astype(np.float32)

    params = [
        (8, 20),
        (16, 50),
        (32, 100),
    ]

    for M, ef in params:
        t, d = run_hnsw_test(M, ef, chunks, embeddings, query_emb)
        print({
            "M": M,
            "ef": ef,
            "search_time": t,
            "avg_distance": d
        })
