import argparse
import logging
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
    evaluate_results,
    expand_query,
    rerank_batch,
    rerank_cross_encoder,
    rerank_individual,
    rewrite_query,
)
from lib.semantic_search import load_movies


logger = logging.getLogger(__name__)


def enable_debug_logging() -> None:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("DEBUG: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def log_results(
    stage: str,
    results: list[dict],
) -> None:
    formatted_results = ", ".join(
        f"{index}. {result['title']} "
        f"(RRF: {result['rrf_score']:.4f})"
        for index, result in enumerate(results, start=1)
    )
    logger.debug("%s: %s", stage, formatted_results)


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
        print(f"{index}. {result['title']}")

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

        print()


def rrf_search_command(
    query: str,
    k: int = 60,
    limit: int = 5,
    enhance: str | None = None,
    rerank_method: str | None = None,
    debug: bool = False,
    evaluate: bool = False,
) -> None:
    if debug:
        logger.debug("Original query: %s", query)

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

    elif enhance == "expand":
        enhanced_query = expand_query(query)

        print(
            f"Enhanced query ({enhance}): "
            f"'{query}' -> '{enhanced_query}'\n"
        )

        search_query = enhanced_query

    if debug:
        logger.debug("Query after enhancements: %s", search_query)

    documents = load_movies()

    hybrid_search = HybridSearch(
        documents
    )

    search_limit = limit

    if rerank_method is not None:
        search_limit = limit * 5

    results = hybrid_search.rrf_search(
        search_query,
        k,
        search_limit,
    )

    if debug:
        log_results("Results after RRF search", results)

    if rerank_method == "individual":
        print(
            f"Re-ranking top {len(results)} results "
            f"using individual method..."
        )

        results = rerank_individual(
            search_query,
            results,
        )

    elif rerank_method == "batch":
        print(
            f"Re-ranking top {len(results)} results "
            f"using batch method..."
        )

        results = rerank_batch(
            search_query,
            results,
        )

    elif rerank_method == "cross_encoder":
        print(
            f"Re-ranking top {len(results)} results "
            f"using cross_encoder method..."
        )

        results = rerank_cross_encoder(
            search_query,
            results,
        )

    results = results[:limit]

    if debug:
        log_results("Final results after re-ranking", results)

    print(
        f"Reciprocal Rank Fusion Results "
        f"for '{search_query}' (k={k}):\n"
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

        if rerank_method == "individual":
            print(
                f"   Re-rank Score: "
                f"{result['rerank_score']:.3f}/10"
            )

        elif rerank_method == "batch":
            print(
                f"   Re-rank Rank: "
                f"{result['rerank_rank']}"
            )

        elif rerank_method == "cross_encoder":
            print(
                f"   Cross Encoder Score: "
                f"{result['cross_encoder_score']:.3f}"
            )

        print(
            f"   RRF Score: "
            f"{result['rrf_score']:.3f}"
        )

        print(
            f"   BM25 Rank: {bm25_display}, "
            f"Semantic Rank: {semantic_display}"
        )

        document = result["document"]

        print(
            f"   {document[:100]}..."
        )

        print()

    if evaluate:
        scores = evaluate_results(search_query, results)

        for index, (result, score) in enumerate(
            zip(results, scores),
            start=1,
        ):
            print(f"{index}. {result['title']}: {score}/3")


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
        choices=[
            "spell",
            "rewrite",
            "expand",
        ],
        help="Query enhancement method",
    )

    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=[
            "individual",
            "batch",
            "cross_encoder",
        ],
        help="Re-ranking method",
    )

    rrf_search_parser.add_argument(
        "--debug",
        action="store_true",
        help="Log each stage of the RRF search pipeline",
    )

    rrf_search_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate final result relevance with an LLM",
    )

    args = parser.parse_args()

    if getattr(args, "debug", False):
        enable_debug_logging()

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
                args.rerank_method,
                args.debug,
                args.evaluate,
            )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
