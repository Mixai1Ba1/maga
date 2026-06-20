import argparse

from config import DATA_PATH, DEFAULT_LIMIT
from data.load_lenta import load_lenta_texts


def print_docs(docs, start=0, count=20, preview=500):
    end = min(start + count, len(docs))
    for i in range(start, end):
        print("=" * 100)
        print(f"doc_id={i}")
        print(docs[i][:preview])
        print()


def search_docs(docs, query, preview=500, max_results=20):
    query_lower = query.lower()
    found = 0

    for i, doc in enumerate(docs):
        if query_lower in doc.lower():
            print("=" * 100)
            print(f"doc_id={i}")
            print(doc[:preview])
            print()
            found += 1
            if found >= max_results:
                break

    if found == 0:
        print("Совпадения не найдены")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--preview", type=int, default=500)
    parser.add_argument("--find", type=str, default=None)
    parser.add_argument("--max-results", type=int, default=20)

    args = parser.parse_args()

    docs = load_lenta_texts(str(DATA_PATH), limit=args.limit)

    if args.find:
        search_docs(
            docs,
            query=args.find,
            preview=args.preview,
            max_results=args.max_results,
        )
    else:
        print_docs(
            docs,
            start=args.start,
            count=args.count,
            preview=args.preview,
        )


if __name__ == "__main__":
    main()