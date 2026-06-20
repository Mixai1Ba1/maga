import time

import hnswlib
import numpy as np

from config import (
    EMBEDDINGS_PATH,
    META_PATH,
    EMBEDDINGS_MANIFEST_PATH,
    INDEX_PATH,
    INDEX_MANIFEST_PATH,
    HNSW_SPACE,
    HNSW_M,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
)
from artifacts import (
    load_numpy,
    load_pickle,
    load_json,
    save_json,
    build_index_manifest,
)


def make_current_index_manifest(
    *,
    embeddings_manifest: dict,
    elements_count: int,
    embedding_dim: int,
    space=HNSW_SPACE,
    M=HNSW_M,
    ef_construction=HNSW_EF_CONSTRUCTION,
    ef_search=HNSW_EF_SEARCH,
):
    return build_index_manifest(
        embeddings_manifest=embeddings_manifest,
        space=space,
        M=M,
        ef_construction=ef_construction,
        ef_search=ef_search,
        elements_count=elements_count,
        embedding_dim=embedding_dim,
    )


def build_index(
    embeddings_path=EMBEDDINGS_PATH,
    meta_path=META_PATH,
    embeddings_manifest_path=EMBEDDINGS_MANIFEST_PATH,
    index_path=INDEX_PATH,
    index_manifest_path=INDEX_MANIFEST_PATH,
    space=HNSW_SPACE,
    M=HNSW_M,
    ef_construction=HNSW_EF_CONSTRUCTION,
    ef_search=HNSW_EF_SEARCH,
):
    embeddings = load_numpy(embeddings_path)
    chunks = load_pickle(meta_path)
    embeddings_manifest = load_json(embeddings_manifest_path)

    n, dim = embeddings.shape

    if len(chunks) != n:
        raise ValueError(
            f"Metadata size mismatch: len(chunks)={len(chunks)} != embeddings rows={n}"
        )

    started_at = time.perf_counter()

    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
    index.add_items(embeddings, ids=np.arange(n))
    index.set_ef(ef_search)
    index.save_index(str(index_path))

    duration_sec = round(time.perf_counter() - started_at, 4)

    index_manifest = make_current_index_manifest(
        embeddings_manifest=embeddings_manifest,
        elements_count=n,
        embedding_dim=dim,
        space=space,
        M=M,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )
    index_manifest["build_time_sec"] = duration_sec

    save_json(index_manifest_path, index_manifest)

    print(f"Index built for {n} vectors")
    print(f"Index saved to: {index_path}")
    print(f"Index manifest saved to: {index_manifest_path}")
    print(f"Index build time: {duration_sec} sec")


if __name__ == "__main__":
    build_index()