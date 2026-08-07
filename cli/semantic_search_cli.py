import argparse

from lib.semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
    embed_query_text,
    embed_text,
    load_movies,
    semantic_chunk_text,
    verify_embeddings,
    verify_model,
)


def search_command(
    query: str,
    limit: int,
) -> None:
    semantic_search = SemanticSearch()
    documents = load_movies()

    semantic_search.load_or_create_embeddings(
        documents
    )

    results = semantic_search.search(
        query,
        limit,
    )

    for position, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{position}. {result['title']} "
            f"(score: {result['score']:.4f})"
        )

        print(
            f"  {result['description']}"
        )

        print()


def chunk_command(
    text: str,
    chunk_size: int = 200,
    overlap: int = 0,
) -> None:
    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be greater than zero."
        )

    if overlap < 0:
        raise ValueError(
            "Overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    words = text.split()
    chunks: list[str] = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        start += chunk_size - overlap

    print(
        f"Chunking {len(text)} characters"
    )

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        print(
            f"{index}. {chunk}"
        )


def semantic_chunk_command(
    text: str,
    max_chunk_size: int = 4,
    overlap: int = 0,
) -> list[str]:
    chunks = semantic_chunk_text(
        text,
        max_chunk_size,
        overlap,
    )

    print(
        f"Semantically chunking "
        f"{len(text)} characters"
    )

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        print(
            f"{index}. {chunk}"
        )

    return chunks


def embed_chunks_command() -> None:
    documents = load_movies()

    semantic_search = ChunkedSemanticSearch()

    embeddings = (
        semantic_search.load_or_create_chunk_embeddings(
            documents
        )
    )

    print(
        f"Generated {len(embeddings)} "
        f"chunked embeddings"
    )


def search_chunked_command(
    query: str,
    limit: int = 5,
) -> None:
    documents = load_movies()

    semantic_search = ChunkedSemanticSearch()

    semantic_search.load_or_create_chunk_embeddings(
        documents
    )

    results = semantic_search.search_chunks(
        query,
        limit,
    )

    for i, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n{i}. {result['title']} "
            f"(score: {result['score']:.4f})"
        )

        print(
            f"   {result['document']}..."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic Search CLI",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    subparsers.add_parser(
        "verify",
        help="Load and verify the embedding model",
    )

    embed_text_parser = subparsers.add_parser(
        "embed_text",
        help="Generate an embedding for text",
    )

    embed_text_parser.add_argument(
        "text",
        type=str,
        help="Text to embed",
    )

    embed_query_parser = subparsers.add_parser(
        "embed_query",
        help="Generate an embedding for a search query",
    )

    embed_query_parser.add_argument(
        "query",
        type=str,
        help="Search query to embed",
    )

    subparsers.add_parser(
        "verify_embeddings",
        help="Build or load movie embeddings",
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search movies using semantic similarity",
    )

    search_parser.add_argument(
        "query",
        type=str,
        help="Semantic search query",
    )

    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results",
    )

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Split text into fixed-size word chunks",
    )

    chunk_parser.add_argument(
        "text",
        type=str,
        help="Text to chunk",
    )

    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of words per chunk",
    )

    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help=(
            "Number of shared words between "
            "consecutive chunks"
        ),
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk",
        help="Split text into sentence-based chunks",
    )

    semantic_chunk_parser.add_argument(
        "text",
        type=str,
        help="Text to chunk semantically",
    )

    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help=(
            "Maximum number of sentences "
            "per chunk"
        ),
    )

    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help=(
            "Number of shared sentences between "
            "consecutive chunks"
        ),
    )

    subparsers.add_parser(
        "embed_chunks",
        help="Build or load chunk embeddings",
    )

    search_chunked_parser = subparsers.add_parser(
        "search_chunked",
        help="Search movies using chunk embeddings",
    )

    search_chunked_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )

    search_chunked_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(
                args.text
            )

        case "embed_query":
            embed_query_text(
                args.query
            )

        case "verify_embeddings":
            verify_embeddings()

        case "search":
            search_command(
                args.query,
                args.limit,
            )

        case "chunk":
            chunk_command(
                args.text,
                args.chunk_size,
                args.overlap,
            )

        case "semantic_chunk":
            semantic_chunk_command(
                args.text,
                args.max_chunk_size,
                args.overlap,
            )

        case "embed_chunks":
            embed_chunks_command()

        case "search_chunked":
            search_chunked_command(
                args.query,
                args.limit,
            )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()