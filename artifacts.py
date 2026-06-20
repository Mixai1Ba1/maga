import json
import pickle
from pathlib import Path
from typing import Any
import numpy as np

from config import (
    ARTIFACTS_DIR,
    EMBEDDINGS_PATH,
    META_PATH,
    EMBEDDINGS_MANIFEST_PATH,
    INDEX_PATH,
    INDEX_MANIFEST_PATH,
)


def ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def save_numpy(path: Path, array: np.ndarray) -> None:
    ensure_artifacts_dir()
    np.save(path, array)


def load_numpy(path: Path) -> np.ndarray:
    return np.load(path)


def save_pickle(path: Path, obj: Any) -> None:
    ensure_artifacts_dir()
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(path: Path, obj: dict) -> None:
    ensure_artifacts_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def embeddings_artifacts_exist() -> bool:
    return (
        file_exists(EMBEDDINGS_PATH)
        and file_exists(META_PATH)
        and file_exists(EMBEDDINGS_MANIFEST_PATH)
    )


def index_artifacts_exist() -> bool:
    return file_exists(INDEX_PATH) and file_exists(INDEX_MANIFEST_PATH)


def build_embeddings_manifest(
    *,
    data_path: str,
    limit: int,
    max_chars: int,
    overlap: int,
    model_name: str,
    embedding_dim: int,
    normalize_embeddings: bool,
    chunks_count: int,
) -> dict:
    return {
        "data_path": data_path,
        "limit": limit,
        "max_chars": max_chars,
        "overlap": overlap,
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "normalize_embeddings": normalize_embeddings,
        "chunks_count": chunks_count,
    }


def build_index_manifest(
    *,
    embeddings_manifest: dict,
    space: str,
    M: int,
    ef_construction: int,
    ef_search: int,
    elements_count: int,
    embedding_dim: int,
) -> dict:
    return {
        "embeddings_manifest": embeddings_manifest,
        "space": space,
        "M": M,
        "ef_construction": ef_construction,
        "ef_search": ef_search,
        "elements_count": elements_count,
        "embedding_dim": embedding_dim,
    }

def compare_embeddings_manifest(current: dict, saved: dict) -> dict:
    keys = [
        "data_path",
        "limit",
        "max_chars",
        "overlap",
        "model_name",
        "embedding_dim",
        "normalize_embeddings",
    ]
    return {k: (current.get(k) == saved.get(k)) for k in keys}


def compare_index_manifest(current: dict, saved: dict) -> dict:
    keys = [
        "space",
        "M",
        "ef_construction",
        "ef_search",
        "elements_count",
        "embedding_dim",
    ]
    return {k: (current.get(k) == saved.get(k)) for k in keys}


def embeddings_manifest_matches(current: dict, saved: dict) -> bool:
    result = compare_embeddings_manifest(current, saved)
    return all(result.values())


def index_manifest_matches(current: dict, saved: dict) -> bool:
    result = compare_index_manifest(current, saved)
    return all(result.values())

from datetime import datetime
from config import PIPELINE_LOG_PATH


def append_log(message: str) -> None:
    ensure_artifacts_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(PIPELINE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

from pathlib import Path


def get_file_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return path.stat().st_size


def bytes_to_mb(size_bytes: int) -> float:
    return round(size_bytes / (1024 * 1024), 4)