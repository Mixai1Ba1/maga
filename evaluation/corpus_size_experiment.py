import time
import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer

from data.load_lenta import load_lenta_texts
from data.chunking import chunk_text

def run_experiment(limit, query, max_chars=900):
    # загрузка данных
    docs = load_lenta_texts("data/lenta.csv", limit=limit)

    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc, max_chars=max_chars))

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # эмбеддинги
    t0 = time.time()
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype(np.float32)
    embed_time = time.time() - t0

    # индекс
    dim = embeddings.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)

    t1 = time.time()
    index.init_index(max_elements=len(chunks), ef_construction=200, M=16)
    index.add_items(embeddings, ids=np.arange(len(chunks)))
    index.set_ef(50)
    index_time = time.time() - t1

    # поиск
    q = model.encode([query], normalize_embeddings=True).astype(np.float32)

    t2 = time.time()
    labels, distances = index.knn_query(q, k=5)
    search_time = time.time() - t2

    return {
        "documents": limit,
        "chunks": len(chunks),
        "embed_time": embed_time,
        "index_time": index_time,
        "search_time": search_time,
    }

if __name__ == "__main__":
    query = "экономические последствия санкций"
    for limit in [1000, 2000, 5000]:
        res = run_experiment(limit, query)
        print(res)
