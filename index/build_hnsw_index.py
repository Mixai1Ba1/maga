import hnswlib
import numpy as np
import pickle
from embeddings.build_embeddings import build_embeddings

def build_index(
    data_path="data/lenta.csv",
    limit=2000,
    index_path="index/hnsw.index",
    meta_path="index/meta.pkl"
):
    chunks, embeddings = build_embeddings(data_path, limit=limit)

    dim = embeddings.shape[1]
    n = embeddings.shape[0]

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, ef_construction=200, M=16)
    index.add_items(embeddings, ids=np.arange(n))
    index.set_ef(50)

    index.save_index(index_path)

    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)  # (doc_id, chunk_text)

    print(f"Index built: {n} chunks (from {limit} docs)")
    print(f"Index saved to: {index_path}")
    print(f"Metadata saved to: {meta_path}")

# if __name__ == "__main__":
    # build_index()

if __name__ == "__main__":
    for limit in [1000, 2000, 5000]:
        print(f"\n=== Building index for {limit} documents ===")
        build_index(
            data_path="data/lenta.csv",
            limit=limit,
            index_path=f"index/hnsw_{limit}.index",
            meta_path=f"index/meta_{limit}.pkl"
        )

    
