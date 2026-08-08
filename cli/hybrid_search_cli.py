import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from lib.hybrid_search import (
    HybridSearch,
    normalize,
)
from lib.query_enhancer import (
    correct_spelling,
    rewrite_query,
)
from lib.semantic_search import load_movies


def normalize_command(
    scores: list[float],
) -> None:
    normalized_scores = normalize(scores)

    for score in normalized_scores:
        print(f"* {score:.4f}")


def weighted_search_command(
    query: str,
    alpha: float = 0.5,
    limit: int = 5,
) -> None:
    documents = load_movies()

    hybrid_search = HybridSearch(
        documents
    )

    results = hybrid_search.weighted_search(
        query,
        alpha,
        limit,
    )

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{index}. {result['title']}"
        )

        print(
            f"  Hybrid Score: "
            f"{result['hybrid_score']:.3f}"
        )

        print(
            f"  BM25: "
            f"{result['bm25_score']:.3f}, "
            f"Semantic: "
            f"{result['semantic_score']:.3f}"
        )

        document = result["document"]

        print(
            f"  {document[:100]}..."
        )


def rrf_search_command(
    query: str,
    k: int = 60,
    limit: int = 5,
    enhance: str | None = None,
) -> None:
    search_query = query

    if enhance == "spell":
        enhanced_query = correct_spelling(query)

        print(
            f"Enhanced query ({enhance}): "
            f"'{query}' -> '{enhanced_query}'\n"
        )

        search_query = enhanced_query

    elif enhance == "rewrite":
        enhanced_query = rewrite_query(query)

        print(
            f"Enhanced query ({enhance}): "
            f"'{query}' -> '{enhanced_query}'\n"
        )

        search_query = enhanced_query

    documents = load_movies()

    hybrid_search = HybridSearch(
        documents
    )

    results = hybrid_search.rrf_search(
        search_query,
        k,
        limit,
    )

    for index, result in enumerate(
        results,
        start=1,
    ):
        bm25_rank = result["bm25_rank"]
        semantic_rank = result["semantic_rank"]

        bm25_display = (
            str(bm25_rank)
            if bm25_rank is not None
            else "N/A"
        )

        semantic_display = (
            str(semantic_rank)
            if semantic_rank is not None
            else "N/A"
        )

        print(
            f"{index}. {result['title']}"
        )

        print(
            f"  RRF Score: "
            f"{result['rrf_score']:.3f}"
        )

        print(
            f"  BM25 Rank: {bm25_display}, "
            f"Semantic Rank: {semantic_display}"
        )

        document = result["document"]

        print(
            f"  {document[:100]}..."
        )

        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid Search CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize scores to a 0-1 range",
    )

    normalize_parser.add_argument(
        "scores",
        type=float,
        nargs="*",
        help="Scores to normalize",
    )

    weighted_search_parser = (
        subparsers.add_parser(
            "weighted-search",
            help=(
                "Search using weighted BM25 "
                "and semantic scores"
            ),
        )
    )

    weighted_search_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )

    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help=(
            "Keyword search weight "
            "between 0.0 and 1.0"
        ),
    )

    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results",
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search",
        help="Search using Reciprocal Rank Fusion",
    )

    rrf_search_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )

    rrf_search_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="RRF rank constant",
    )

    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results",
    )

    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite"],
        help="Query enhancement method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(
                args.scores
            )

        case "weighted-search":
            weighted_search_command(
                args.query,
                args.alpha,
                args.limit,
            )

        case "rrf-search":
            rrf_search_command(
                args.query,
                args.k,
                args.limit,
                args.enhance,
            )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()