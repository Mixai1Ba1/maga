import re

def chunk_text(text: str, max_chars: int = 900, overlap: int = 150):
    """
    Простое разбиение по символам с перекрытием.
    Для диплома достаточно: стабильно, прозрачно, воспроизводимо.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks
