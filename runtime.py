import hnswlib
from sentence_transformers import SentenceTransformer

from config import (
    MODEL_NAME,
    EMBEDDING_DIM,
    INDEX_PATH,
    META_PATH,
    EMBEDDINGS_MANIFEST_PATH,
    INDEX_MANIFEST_PATH,
    HNSW_SPACE,
    HNSW_EF_SEARCH,
)
from artifacts import (
    load_pickle,
    load_json,
    embeddings_artifacts_exist,
    index_artifacts_exist,
)


def validate_runtime_artifacts() -> None:
    if not embeddings_artifacts_exist():
        raise FileNotFoundError(
            "Embeddings artifacts are missing. Run: python pipeline.py build"
        )

    if not index_artifacts_exist():
        raise FileNotFoundError(
            "Index artifacts are missing. Run: python pipeline.py build or python pipeline.py reindex"
        )

    emb_manifest = load_json(EMBEDDINGS_MANIFEST_PATH)
    idx_manifest = load_json(INDEX_MANIFEST_PATH)

    if idx_manifest.get("embeddings_manifest") != emb_manifest:
        raise ValueError(
            "Artifacts are inconsistent: index manifest does not match embeddings manifest. Rebuild index."
        )

    if idx_manifest.get("embedding_dim") != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: manifest={idx_manifest.get('embedding_dim')} config={EMBEDDING_DIM}"
        )

    if idx_manifest.get("space") != HNSW_SPACE:
        raise ValueError(
            f"Index space mismatch: manifest={idx_manifest.get('space')} config={HNSW_SPACE}"
        )


def load_runtime():
    validate_runtime_artifacts()

    model = SentenceTransformer(MODEL_NAME)

    index = hnswlib.Index(space=HNSW_SPACE, dim=EMBEDDING_DIM)
    index.load_index(str(INDEX_PATH))
    index.set_ef(HNSW_EF_SEARCH)

    chunks = load_pickle(META_PATH)

    return model, index, chunks