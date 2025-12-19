from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

texts = [
    "Семантический поиск по документам",
    "Рецепт борща",
    "Инструкция по технике безопасности"
]

emb = model.encode(texts, normalize_embeddings=True)

print("Shape:", emb.shape)
print("Norm:", np.linalg.norm(emb[0]))
