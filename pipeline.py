import argparse

from config import (
    DATA_PATH,
    DEFAULT_LIMIT,
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP,
    EMBEDDINGS_MANIFEST_PATH,
    INDEX_MANIFEST_PATH,
    EMBEDDINGS_PATH,
)
from artifacts import (
    embeddings_artifacts_exist,
    index_artifacts_exist,
    load_json,
    load_numpy,
    compare_embeddings_manifest,
    compare_index_manifest,
    embeddings_manifest_matches,
    index_manifest_matches,
    append_log,
)
from embeddings.build_embeddings import (
    save_embeddings_artifacts,
    make_current_embeddings_manifest,
)
from index.build_hnsw_index import (
    build_index,
    make_current_index_manifest,
)


def status(limit=DEFAULT_LIMIT, max_chars=DEFAULT_MAX_CHARS, overlap=DEFAULT_OVERLAP):
    print("=== Pipeline status ===")

    if not embeddings_artifacts_exist():
        print("Embeddings artifacts: missing")
        embeddings_ok = False
    else:
        saved_embeddings_manifest = load_json(EMBEDDINGS_MANIFEST_PATH)
        current_embeddings_manifest = make_current_embeddings_manifest(
            path=DATA_PATH,
            limit=limit,
            max_chars=max_chars,
            overlap=overlap,
            chunks_count=saved_embeddings_manifest.get("chunks_count", -1),
        )
        cmp_result = compare_embeddings_manifest(
            current_embeddings_manifest,
            saved_embeddings_manifest,
        )
        embeddings_ok = embeddings_manifest_matches(
            current_embeddings_manifest,
            saved_embeddings_manifest,
        )
        print("Embeddings artifacts: present")
        print("Embeddings manifest comparison:", cmp_result)
        print("Embeddings up-to-date:", embeddings_ok)

    if not index_artifacts_exist():
        print("Index artifacts: missing")
        index_ok = False
    else:
        if not embeddings_artifacts_exist():
            print("Index artifacts: present but embeddings artifacts missing")
            print("Index up-to-date: False")
            index_ok = False
        elif not embeddings_ok:
            print("Index artifacts: present but embeddings are stale for current configuration")
            print("Index up-to-date: False")
            index_ok = False
        else:
            embeddings = load_numpy(EMBEDDINGS_PATH)
            saved_embeddings_manifest = load_json(EMBEDDINGS_MANIFEST_PATH)
            saved_index_manifest = load_json(INDEX_MANIFEST_PATH)

            current_index_manifest = make_current_index_manifest(
                embeddings_manifest=saved_embeddings_manifest,
                elements_count=embeddings.shape[0],
                embedding_dim=embeddings.shape[1],
                space=saved_index_manifest.get("space"),
                M=saved_index_manifest.get("M"),
                ef_construction=saved_index_manifest.get("ef_construction"),
                ef_search=saved_index_manifest.get("ef_search"),
            )

            cmp_result = compare_index_manifest(
                current_index_manifest,
                saved_index_manifest,
            )
            same_embeddings_ref = (
                saved_index_manifest.get("embeddings_manifest") == saved_embeddings_manifest
            )

            index_ok = index_manifest_matches(
                current_index_manifest,
                saved_index_manifest,
            ) and same_embeddings_ref

            print("Index artifacts: present")
            print("Index manifest comparison:", cmp_result)
            print("Index references current embeddings manifest:", same_embeddings_ref)
            print("Index up-to-date:", index_ok)

    if not embeddings_ok:
        print("Recommended action: run 'python pipeline.py build' to rebuild embeddings and index")
    elif not index_ok:
        print("Recommended action: run 'python pipeline.py reindex' to rebuild only the index")
    else:
        print("Recommended action: no rebuild required")

    return embeddings_ok, index_ok


def validate(limit=DEFAULT_LIMIT, max_chars=DEFAULT_MAX_CHARS, overlap=DEFAULT_OVERLAP):
    embeddings_ok, index_ok = status(
        limit=limit,
        max_chars=max_chars,
        overlap=overlap,
    )
    if embeddings_ok and index_ok:
        print("Validation passed")
    else:
        print("Validation failed")


def build(limit=DEFAULT_LIMIT, max_chars=DEFAULT_MAX_CHARS, overlap=DEFAULT_OVERLAP):
    append_log(
        f"BUILD started: limit={limit}, max_chars={max_chars}, overlap={overlap}"
    )
    save_embeddings_artifacts(
        path=DATA_PATH,
        limit=limit,
        max_chars=max_chars,
        overlap=overlap,
    )
    build_index()
    append_log("BUILD completed")


def reindex(M=None, ef_construction=None, ef_search=None):
    kwargs = {}
    if M is not None:
        kwargs["M"] = M
    if ef_construction is not None:
        kwargs["ef_construction"] = ef_construction
    if ef_search is not None:
        kwargs["ef_search"] = ef_search

    append_log(
        f"REINDEX started: M={kwargs.get('M')}, ef_construction={kwargs.get('ef_construction')}, ef_search={kwargs.get('ef_search')}"
    )
    build_index(**kwargs)
    append_log("REINDEX completed")


def sync(limit=DEFAULT_LIMIT, max_chars=DEFAULT_MAX_CHARS, overlap=DEFAULT_OVERLAP):
    append_log(
        f"SYNC check: limit={limit}, max_chars={max_chars}, overlap={overlap}"
    )
    embeddings_ok, index_ok = status(
        limit=limit,
        max_chars=max_chars,
        overlap=overlap,
    )

    if embeddings_ok and index_ok:
        append_log("SYNC decision: no rebuild required")
        print("SYNC: artifacts are already актуальны")
        return

    if not embeddings_ok:
        append_log("SYNC decision: full build required")
        print("SYNC: выполняется полный rebuild эмбеддингов и индекса")
        build(limit=limit, max_chars=max_chars, overlap=overlap)
        return

    if not index_ok:
        append_log("SYNC decision: reindex required")
        print("SYNC: выполняется только перестроение индекса")
        reindex()
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    build_parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    build_parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    validate_parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    validate_parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    status_parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    status_parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)

    sync_parser = subparsers.add_parser("sync")
    sync_parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    sync_parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    sync_parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)

    reindex_parser = subparsers.add_parser("reindex")
    reindex_parser.add_argument("--M", type=int, default=None)
    reindex_parser.add_argument("--ef-construction", type=int, default=None)
    reindex_parser.add_argument("--ef-search", type=int, default=None)

    args = parser.parse_args()

    if args.command == "build":
        build(limit=args.limit, max_chars=args.max_chars, overlap=args.overlap)
    elif args.command == "validate":
        validate(limit=args.limit, max_chars=args.max_chars, overlap=args.overlap)
    elif args.command == "status":
        status(limit=args.limit, max_chars=args.max_chars, overlap=args.overlap)
    elif args.command == "sync":
        sync(limit=args.limit, max_chars=args.max_chars, overlap=args.overlap)
    elif args.command == "reindex":
        reindex(
            M=args.M,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
        )