import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------
# Данные
# -----------------------
documents = [
    "Порядок оформления отпуска сотруднику",
    "Инструкция по технике безопасности на предприятии",
    "Рецепт классического борща",
    "Семантический поиск по текстовым документам",
    "Правила информационной безопасности в организации"
]

# -----------------------
# Эмбеддинги
# -----------------------
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

embeddings = model.encode(
    documents,
    normalize_embeddings=True
).astype(np.float32)

dim = embeddings.shape[1]

# -----------------------
# ANN индекс (HNSW)
# -----------------------
index = hnswlib.Index(space="cosine", dim=dim)
index.init_index(
    max_elements=len(documents),
    ef_construction=100,
    M=16
)

index.add_items(embeddings, ids=np.arange(len(documents)))
index.set_ef(50)

# -----------------------
# Поиск
# -----------------------
query = "поиск документов по смыслу"
query_emb = model.encode(
    [query],
    normalize_embeddings=True
).astype(np.float32)

labels, distances = index.knn_query(query_emb, k=3)

print("Запрос:", query)
print("Результаты:")
for idx, dist in zip(labels[0], distances[0]):
    print(f"{1 - dist:.4f} | {documents[idx]}")
