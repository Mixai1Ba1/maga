import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import DATA_PATH, DEFAULT_LIMIT, DEFAULT_MAX_CHARS
from data.load_lenta import load_lenta_texts
from data.chunking import chunk_text
from runtime import load_runtime
from search.semantic_search_full_docs import build_index_full_docs, search_full_docs
from evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank, mean


JUDGMENTS_PATH = Path("evaluation/data/judgments.json")


def load_judgments():
    with open(JUDGMENTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_semantic_chunked(query: str, top_k: int = 10):
    model, index, chunks = load_runtime()
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    labels, distances = index.knn_query(q_emb, k=top_k)

    retrieved_doc_ids = []
    for idx in labels[0]:
        doc_id, _chunk_text = chunks[idx]
        retrieved_doc_ids.append(int(doc_id))
    return retrieved_doc_ids


def build_tfidf_corpus(path, limit=2000, max_chars=900):
    docs = load_lenta_texts(path, limit=limit)
    corpus = []
    doc_ids = []

    for doc_id, doc in enumerate(docs):
        chunks = chunk_text(doc, max_chars=max_chars)
        for ch in chunks:
            corpus.append(ch)
            doc_ids.append(doc_id)

    return corpus, doc_ids


def run_tfidf(query: str, corpus, corpus_doc_ids, vectorizer, tfidf_matrix, top_k: int = 10):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]

    retrieved_doc_ids = [int(corpus_doc_ids[i]) for i in top_idx]
    return retrieved_doc_ids


def run_semantic_full_docs(query: str, index, texts, top_k: int = 10):
    results = search_full_docs(query, index, texts, top_k=top_k)
    return [doc_id for _score, doc_id, _text in results]

def evaluate_run(name: str, retrieval_fn, judgments, top_k: int = 10):
    p5_scores = []
    r5_scores = []
    mrr_scores = []

    for item in judgments:
        query = item["query"]
        relevant = set(item["relevant_doc_ids"])
        retrieved = retrieval_fn(query, top_k)

        p5_scores.append(precision_at_k(retrieved, relevant, k=5))
        r5_scores.append(recall_at_k(retrieved, relevant, k=5))
        mrr_scores.append(reciprocal_rank(retrieved, relevant))

    return {
        "method": name,
        "Precision@5": round(mean(p5_scores), 4),
        "Recall@5": round(mean(r5_scores), 4),
        "MRR": round(mean(mrr_scores), 4),
    }


if __name__ == "__main__":
    judgments = load_judgments()

    # TF-IDF
    corpus, corpus_doc_ids = build_tfidf_corpus(
        DATA_PATH,
        limit=DEFAULT_LIMIT,
        max_chars=DEFAULT_MAX_CHARS,
    )
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words=None,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    tfidf_result = evaluate_run(
        "TF-IDF",
        lambda q, k: run_tfidf(q, corpus, corpus_doc_ids, vectorizer, tfidf_matrix, top_k=k),
        judgments,
    )

    # Semantic search by chunks
    semantic_chunked_result = evaluate_run(
        "Semantic HNSW (chunks)",
        lambda q, k: run_semantic_chunked(q, top_k=k),
        judgments,
    )

    # Semantic search by full docs
    full_index, full_texts = build_index_full_docs(str(DATA_PATH), limit=DEFAULT_LIMIT)
    semantic_full_result = evaluate_run(
        "Semantic HNSW (full docs)",
        lambda q, k: run_semantic_full_docs(q, full_index, full_texts, top_k=k),
        judgments,
    )

    print(tfidf_result)
    print(semantic_chunked_result)
    print(semantic_full_result)