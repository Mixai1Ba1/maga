import numpy as np

from runtime import load_runtime

model, index, chunks = load_runtime()


def search(query: str, top_k: int = 5):
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    labels, distances = index.knn_query(q_emb, k=top_k)

    results = []
    for idx, dist in zip(labels[0], distances[0]):
        doc_id, chunk_text = chunks[idx]
        score = float(1.0 - dist)
        results.append((score, doc_id, chunk_text))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


if __name__ == "__main__":
    q = input("Запрос: ").strip()
    res = search(q, top_k=5)

    print("Результаты:")
    for score, doc_id, text in res:
        print(f"{score:.4f} | doc_id={doc_id} | {text[:200]}")