import argparse
from collections import OrderedDict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import DATA_PATH, DEFAULT_LIMIT, DEFAULT_MAX_CHARS
from data.load_lenta import load_lenta_texts
from data.chunking import chunk_text
from runtime import load_runtime
from search.semantic_search_full_docs import build_index_full_docs, search_full_docs


def build_tfidf_corpus(path, limit=2000, max_chars=900):
    docs = load_lenta_texts(str(path), limit=limit)
    corpus = []
    corpus_doc_ids = []

    for doc_id, doc in enumerate(docs):
        for ch in chunk_text(doc, max_chars=max_chars):
            corpus.append(ch)
            corpus_doc_ids.append(doc_id)

    return docs, corpus, corpus_doc_ids


def semantic_chunked_candidates(query: str, top_k: int = 10):
    model, index, chunks = load_runtime()
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    labels, distances = index.knn_query(q_emb, k=top_k)

    results = []
    for idx, dist in zip(labels[0], distances[0]):
        doc_id, chunk_text = chunks[idx]
        results.append(
            {
                "method": "semantic_chunks",
                "doc_id": int(doc_id),
                "score": float(1.0 - dist),
                "preview": chunk_text[:300],
            }
        )
    return results


def semantic_full_docs_candidates(query: str, top_k: int = 10, limit: int = DEFAULT_LIMIT):
    index, texts = build_index_full_docs(str(DATA_PATH), limit=limit)
    results = search_full_docs(query, index, texts, top_k=top_k)

    pooled = []
    for score, doc_id, text in results:
        pooled.append(
            {
                "method": "semantic_full_docs",
                "doc_id": int(doc_id),
                "score": float(score),
                "preview": text[:300],
            }
        )
    return pooled


def tfidf_candidates(query: str, top_k: int = 10, limit: int = DEFAULT_LIMIT, max_chars: int = DEFAULT_MAX_CHARS):
    docs, corpus, corpus_doc_ids = build_tfidf_corpus(
        DATA_PATH,
        limit=limit,
        max_chars=max_chars,
    )

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words=None,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in top_idx:
        results.append(
            {
                "method": "tfidf",
                "doc_id": int(corpus_doc_ids[i]),
                "score": float(scores[i]),
                "preview": corpus[i][:300],
            }
        )
    return results


def build_pool(query: str, top_k: int = 10, limit: int = DEFAULT_LIMIT, max_chars: int = DEFAULT_MAX_CHARS):
    pool = []
    pool.extend(semantic_chunked_candidates(query, top_k=top_k))
    pool.extend(semantic_full_docs_candidates(query, top_k=top_k, limit=limit))
    pool.extend(tfidf_candidates(query, top_k=top_k, limit=limit, max_chars=max_chars))

    unique_docs = OrderedDict()

    for item in pool:
        doc_id = item["doc_id"]
        if doc_id not in unique_docs:
            unique_docs[doc_id] = {
                "doc_id": doc_id,
                "methods": [item["method"]],
                "best_score": item["score"],
                "preview": item["preview"],
            }
        else:
            unique_docs[doc_id]["methods"].append(item["method"])
            unique_docs[doc_id]["best_score"] = max(unique_docs[doc_id]["best_score"], item["score"])

    docs = list(unique_docs.values())
    docs.sort(key=lambda x: (len(set(x["methods"])), x["best_score"]), reverse=True)
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--show", type=int, default=20)

    args = parser.parse_args()

    docs = build_pool(
        query=args.query,
        top_k=args.top_k,
        limit=args.limit,
        max_chars=args.max_chars,
    )

    print(f"Пул кандидатов для запроса: {args.query}")
    print(f"Уникальных doc_id: {len(docs)}")
    print()

    for item in docs[: args.show]:
        print("=" * 100)
        print(f"doc_id={item['doc_id']}")
        print(f"methods={sorted(set(item['methods']))}")
        print(f"best_score={item['best_score']:.4f}")
        print(item["preview"])
        print()


if __name__ == "__main__":
    main()