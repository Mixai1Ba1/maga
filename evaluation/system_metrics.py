import json
import time
from pathlib import Path
from statistics import mean, median

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    DATA_PATH,
    DEFAULT_LIMIT,
    DEFAULT_MAX_CHARS,
    EMBEDDINGS_PATH,
    META_PATH,
    INDEX_PATH,
    EMBEDDINGS_MANIFEST_PATH,
    INDEX_MANIFEST_PATH,
    MODEL_NAME,
)
from artifacts import (
    load_json,
    get_file_size_bytes,
    bytes_to_mb,
)
from runtime import load_runtime
from data.load_lenta import load_lenta_texts
from data.chunking import chunk_text
from search.semantic_search_full_docs import build_index_full_docs


RESULTS_PATH = Path("evaluation/results/system_metrics.json")


def summarize_times_ms(times_ms: list[float]) -> dict:
    return {
        "avg_ms": round(mean(times_ms), 4),
        "median_ms": round(median(times_ms), 4),
        "min_ms": round(min(times_ms), 4),
        "max_ms": round(max(times_ms), 4),
    }


def build_tfidf_corpus(path, limit=2000, max_chars=900):
    docs = load_lenta_texts(str(path), limit=limit)
    corpus = []

    for doc in docs:
        corpus.extend(chunk_text(doc, max_chars=max_chars))

    return corpus


def prepare_semantic_chunked_runtime():
    model, index, chunks = load_runtime()
    return model, index, chunks


def prepare_semantic_full_docs_runtime(limit=DEFAULT_LIMIT):
    model = SentenceTransformer(MODEL_NAME)
    index, texts = build_index_full_docs(str(DATA_PATH), limit=limit)
    return model, index, texts


def prepare_tfidf_runtime(limit=DEFAULT_LIMIT, max_chars=DEFAULT_MAX_CHARS):
    corpus = build_tfidf_corpus(
        DATA_PATH,
        limit=limit,
        max_chars=max_chars,
    )

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words=None,
    )

    fit_start = time.perf_counter()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    fit_time_ms = (time.perf_counter() - fit_start) * 1000

    return corpus, vectorizer, tfidf_matrix, round(fit_time_ms, 4)


def warmup_semantic(model, index, query="тестовый запрос"):
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    index.knn_query(q_emb, k=5)


def warmup_tfidf(vectorizer, tfidf_matrix, query="тестовый запрос"):
    q_vec = vectorizer.transform([query])
    cosine_similarity(q_vec, tfidf_matrix)[0]


def measure_semantic_chunked(query: str, model, index, chunks, runs: int = 5, top_k: int = 5):
    warmup_semantic(model, index, query=query)

    encode_times = []
    search_times = []
    total_times = []

    for _ in range(runs):
        total_start = time.perf_counter()

        encode_start = time.perf_counter()
        q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
        encode_elapsed = (time.perf_counter() - encode_start) * 1000

        search_start = time.perf_counter()
        labels, distances = index.knn_query(q_emb, k=top_k)
        _results = [(float(1.0 - dist), int(chunks[idx][0])) for idx, dist in zip(labels[0], distances[0])]
        search_elapsed = (time.perf_counter() - search_start) * 1000

        total_elapsed = (time.perf_counter() - total_start) * 1000

        encode_times.append(encode_elapsed)
        search_times.append(search_elapsed)
        total_times.append(total_elapsed)

    return {
        "method": "Semantic HNSW (chunks)",
        "encode": summarize_times_ms(encode_times),
        "search": summarize_times_ms(search_times),
        "total": summarize_times_ms(total_times),
    }


def measure_semantic_full_docs(query: str, model, index, texts, runs: int = 5, top_k: int = 5):
    warmup_semantic(model, index, query=query)

    encode_times = []
    search_times = []
    total_times = []

    for _ in range(runs):
        total_start = time.perf_counter()

        encode_start = time.perf_counter()
        q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
        encode_elapsed = (time.perf_counter() - encode_start) * 1000

        search_start = time.perf_counter()
        labels, distances = index.knn_query(q_emb, k=top_k)
        _results = [(float(1.0 - d), int(i)) for i, d in zip(labels[0], distances[0])]
        search_elapsed = (time.perf_counter() - search_start) * 1000

        total_elapsed = (time.perf_counter() - total_start) * 1000

        encode_times.append(encode_elapsed)
        search_times.append(search_elapsed)
        total_times.append(total_elapsed)

    return {
        "method": "Semantic HNSW (full docs)",
        "encode": summarize_times_ms(encode_times),
        "search": summarize_times_ms(search_times),
        "total": summarize_times_ms(total_times),
    }


def measure_tfidf(query: str, vectorizer, tfidf_matrix, fit_time_ms: float, runs: int = 5, top_k: int = 5):
    warmup_tfidf(vectorizer, tfidf_matrix, query=query)

    encode_times = []
    search_times = []
    total_times = []

    for _ in range(runs):
        total_start = time.perf_counter()

        encode_start = time.perf_counter()
        q_vec = vectorizer.transform([query])
        encode_elapsed = (time.perf_counter() - encode_start) * 1000

        search_start = time.perf_counter()
        scores = cosine_similarity(q_vec, tfidf_matrix)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        _results = [(float(scores[i]), int(i)) for i in top_idx]
        search_elapsed = (time.perf_counter() - search_start) * 1000

        total_elapsed = (time.perf_counter() - total_start) * 1000

        encode_times.append(encode_elapsed)
        search_times.append(search_elapsed)
        total_times.append(total_elapsed)

    return {
        "method": "TF-IDF",
        "fit_time_ms": fit_time_ms,
        "encode": summarize_times_ms(encode_times),
        "search": summarize_times_ms(search_times),
        "total": summarize_times_ms(total_times),
    }


def build_report():
    embeddings_manifest = load_json(EMBEDDINGS_MANIFEST_PATH)
    index_manifest = load_json(INDEX_MANIFEST_PATH)

    embeddings_size = get_file_size_bytes(EMBEDDINGS_PATH)
    meta_size = get_file_size_bytes(META_PATH)
    index_size = get_file_size_bytes(INDEX_PATH)

    report = {
        "artifacts": {
            "embeddings_npy": {
                "bytes": embeddings_size,
                "mb": bytes_to_mb(embeddings_size),
            },
            "meta_pkl": {
                "bytes": meta_size,
                "mb": bytes_to_mb(meta_size),
            },
            "hnsw_index": {
                "bytes": index_size,
                "mb": bytes_to_mb(index_size),
            },
        },
        "build_metrics": {
            "embeddings_build_time_sec": embeddings_manifest.get("build_time_sec"),
            "index_build_time_sec": index_manifest.get("build_time_sec"),
            "embeddings_shape": embeddings_manifest.get("embeddings_shape"),
            "chunks_count": embeddings_manifest.get("chunks_count"),
        },
        "queries": [],
    }

    # prepare runtimes once
    chunk_model, chunk_index, chunk_chunks = prepare_semantic_chunked_runtime()
    full_model, full_index, full_texts = prepare_semantic_full_docs_runtime(limit=DEFAULT_LIMIT)
    _corpus, tfidf_vectorizer, tfidf_matrix, tfidf_fit_time_ms = prepare_tfidf_runtime(
        limit=DEFAULT_LIMIT,
        max_chars=DEFAULT_MAX_CHARS,
    )

    queries = [
        "экономические последствия санкций",
        "рост цен на нефть",
        "чемпионат мира по футболу",
    ]

    for q in queries:
        report["queries"].append(
            {
                "query": q,
                "semantic_chunks": measure_semantic_chunked(
                    q, chunk_model, chunk_index, chunk_chunks, runs=5, top_k=5
                ),
                "semantic_full_docs": measure_semantic_full_docs(
                    q, full_model, full_index, full_texts, runs=5, top_k=5
                ),
                "tfidf": measure_tfidf(
                    q, tfidf_vectorizer, tfidf_matrix, tfidf_fit_time_ms, runs=5, top_k=5
                ),
            }
        )

    return report


def print_report(report: dict):
    print("=== Artifact sizes ===")
    print(
        f"embeddings.npy: {report['artifacts']['embeddings_npy']['bytes']} bytes "
        f"({report['artifacts']['embeddings_npy']['mb']} MB)"
    )
    print(
        f"meta.pkl:       {report['artifacts']['meta_pkl']['bytes']} bytes "
        f"({report['artifacts']['meta_pkl']['mb']} MB)"
    )
    print(
        f"hnsw.index:     {report['artifacts']['hnsw_index']['bytes']} bytes "
        f"({report['artifacts']['hnsw_index']['mb']} MB)"
    )
    print()

    print("=== Build metrics ===")
    print(f"Embeddings build time: {report['build_metrics']['embeddings_build_time_sec']} sec")
    print(f"Index build time:      {report['build_metrics']['index_build_time_sec']} sec")
    print(f"Embeddings shape:      {report['build_metrics']['embeddings_shape']}")
    print(f"Chunks count:          {report['build_metrics']['chunks_count']}")
    print()

    print("=== Query timing benchmark ===")
    for item in report["queries"]:
        print(f"--- Query: {item['query']} ---")
        print(item["semantic_chunks"])
        print(item["semantic_full_docs"])
        print(item["tfidf"])
        print()


def save_report(report: dict):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved benchmark results to: {RESULTS_PATH}")


def main():
    report = build_report()
    print_report(report)
    save_report(report)


if __name__ == "__main__":
    main()