import pandas as pd

def load_lenta_df(path, limit=None):
    df = pd.read_csv(path, low_memory=False)
    # Оставляем только нужное
    cols = [c for c in ["title", "text"] if c in df.columns]
    df = df[cols].dropna()
    if limit:
        df = df.head(limit)
    return df

def load_lenta_texts(path, limit=1000):
    df = load_lenta_df(path, limit=limit)
    # Склеиваем title + text: обычно сильно улучшает релевантность
    if "title" in df.columns:
        texts = (df["title"].astype(str) + ". " + df["text"].astype(str)).tolist()
    else:
        texts = df["text"].astype(str).tolist()
    return texts
