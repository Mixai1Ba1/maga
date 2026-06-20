from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "lenta.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
META_PATH = ARTIFACTS_DIR / "meta.pkl"
EMBEDDINGS_MANIFEST_PATH = ARTIFACTS_DIR / "embeddings_manifest.json"

INDEX_PATH = ARTIFACTS_DIR / "hnsw.index"
INDEX_MANIFEST_PATH = ARTIFACTS_DIR / "index_manifest.json"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

DEFAULT_LIMIT = 2000
DEFAULT_MAX_CHARS = 900
DEFAULT_OVERLAP = 150

HNSW_SPACE = "cosine"
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 100

PIPELINE_LOG_PATH = ARTIFACTS_DIR / "pipeline.log"