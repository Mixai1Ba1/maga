import numpy as np
import time
from sentence_transformers import SentenceTransformer

from config import (
    DATA_PATH,
    MODEL_NAME,
    EMBEDDING_DIM,
    DEFAULT_LIMIT,
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP,
    EMBEDDINGS_PATH,
    META_PATH,
    EMBEDDINGS_MANIFEST_PATH,
)
from data.load_lenta import load_lenta_texts
from data.chunking import chunk_text
from artifacts import (
    save_numpy,
    save_pickle,
    save_json,
    build_embeddings_manifest,
)


def build_embeddings(
    path=DATA_PATH,
    limit=DEFAULT_LIMIT,
    max_chars=DEFAULT_MAX_CHARS,
    overlap=DEFAULT_OVERLAP,
):
    docs = load_lenta_texts(str(path), limit=limit)

    chunks = []
    for doc_id, doc in enumerate(docs):
        for ch in chunk_text(doc, max_chars=max_chars, overlap=overlap):
            chunks.append((doc_id, ch))

    texts = [ch for (_, ch) in chunks]

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    return chunks, emb

def make_current_embeddings_manifest(
    path=DATA_PATH,
    limit=DEFAULT_LIMIT,
    max_chars=DEFAULT_MAX_CHARS,
    overlap=DEFAULT_OVERLAP,
    chunks_count: int | None = None,
):
    return build_embeddings_manifest(
        data_path=str(path),
        limit=limit,
        max_chars=max_chars,
        overlap=overlap,
        model_name=MODEL_NAME,
        embedding_dim=EMBEDDING_DIM,
        normalize_embeddings=True,
        chunks_count=chunks_count if chunks_count is not None else -1,
    )

def save_embeddings_artifacts(
    path=DATA_PATH,
    limit=DEFAULT_LIMIT,
    max_chars=DEFAULT_MAX_CHARS,
    overlap=DEFAULT_OVERLAP,
):
    started_at = time.perf_counter()

    chunks, emb = build_embeddings(
        path=path,
        limit=limit,
        max_chars=max_chars,
        overlap=overlap,
    )

    duration_sec = round(time.perf_counter() - started_at, 4)

    manifest = make_current_embeddings_manifest(
        path=path,
        limit=limit,
        max_chars=max_chars,
        overlap=overlap,
        chunks_count=len(chunks),
    )
    manifest["build_time_sec"] = duration_sec
    manifest["embeddings_shape"] = list(emb.shape)

    save_numpy(EMBEDDINGS_PATH, emb)
    save_pickle(META_PATH, chunks)
    save_json(EMBEDDINGS_MANIFEST_PATH, manifest)

    print(f"Embeddings saved to: {EMBEDDINGS_PATH}")
    print(f"Metadata saved to: {META_PATH}")
    print(f"Embeddings manifest saved to: {EMBEDDINGS_MANIFEST_PATH}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings shape: {emb.shape}")
    print(f"Embeddings build time: {duration_sec} sec")


if __name__ == "__main__":
    save_embeddings_artifacts()