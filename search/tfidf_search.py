import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data.load_lenta import load_lenta_texts
from data.chunking import chunk_text

def build_tfidf_corpus(path, limit=2000, max_chars=900):
    docs = load_lenta_texts(path, limit=limit)

    corpus = []
    for doc in docs:
        corpus.extend(chunk_text(doc, max_chars=max_chars))

    return corpus

def tfidf_search(query, corpus, vectorizer, tfidf_matrix, top_k=5):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]

    return [(scores[i], corpus[i]) for i in top_idx]

if __name__ == "__main__":
    corpus = build_tfidf_corpus("data/lenta.csv", limit=2000)
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words=None
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    query = "экономические последствия санкций"
    results = tfidf_search(query, corpus, vectorizer, tfidf_matrix)

    print("TF-IDF поиск")
    print("Запрос:", query)
    for score, text in results:
        print(f"{score:.4f} | {text[:200]}...")
