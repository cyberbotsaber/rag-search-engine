from typing import Any

from inverted_index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


Movie = dict[str, Any]
HybridResult = dict[str, Any]


def normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0 for _ in scores]

    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]


def hybrid_score(
    bm25_score: float,
    semantic_score: float,
    alpha: float = 0.5,
) -> float:
    return (
        alpha * bm25_score
        + (1 - alpha) * semantic_score
    )


def rrf_score(
    rank: int,
    k: int = 60,
) -> float:
    return 1 / (k + rank)


class HybridSearch:
    def __init__(
        self,
        documents: list[Movie],
    ) -> None:
        self.documents = documents

        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(
            documents
        )

        self.idx = InvertedIndex()

        try:
            self.idx.load()
        except FileNotFoundError:
            self.idx.build()
            self.idx.save()

    def _bm25_search(
        self,
        query: str,
        limit: int,
    ):
        self.idx.load()

        return self.idx.bm25_search(
            query,
            limit,
        )

    def weighted_search(
        self,
        query: str,
        alpha: float,
        limit: int = 5,
    ) -> list[HybridResult]:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "Alpha must be between 0.0 and 1.0."
            )

        candidate_limit = limit * 500

        bm25_results = self._bm25_search(
            query,
            candidate_limit,
        )

        semantic_results = (
            self.semantic_search.search_chunks(
                query,
                candidate_limit,
            )
        )

        bm25_scores = [
            score
            for _, score in bm25_results
        ]

        semantic_scores = [
            result["score"]
            for result in semantic_results
        ]

        normalized_bm25_scores = normalize(
            bm25_scores
        )

        normalized_semantic_scores = normalize(
            semantic_scores
        )

        combined: dict[int, HybridResult] = {}

        for (
            bm25_result,
            normalized_score,
        ) in zip(
            bm25_results,
            normalized_bm25_scores,
        ):
            movie, _ = bm25_result
            doc_id = movie["id"]

            combined[doc_id] = {
                "id": doc_id,
                "title": movie["title"],
                "document": movie["description"],
                "bm25_score": normalized_score,
                "semantic_score": 0.0,
                "hybrid_score": 0.0,
            }

        for (
            semantic_result,
            normalized_score,
        ) in zip(
            semantic_results,
            normalized_semantic_scores,
        ):
            doc_id = semantic_result["id"]

            if doc_id not in combined:
                combined[doc_id] = {
                    "id": doc_id,
                    "title": semantic_result["title"],
                    "document": semantic_result["document"],
                    "bm25_score": 0.0,
                    "semantic_score": normalized_score,
                    "hybrid_score": 0.0,
                }
            else:
                combined[doc_id][
                    "semantic_score"
                ] = normalized_score

        for result in combined.values():
            result["hybrid_score"] = hybrid_score(
                result["bm25_score"],
                result["semantic_score"],
                alpha,
            )

        ranked_results = sorted(
            combined.values(),
            key=lambda result: result["hybrid_score"],
            reverse=True,
        )

        return ranked_results[:limit]

    def rrf_search(
        self,
        query: str,
        k: int = 60,
        limit: int = 10,
    ) -> list[HybridResult]:
        candidate_limit = limit * 500

        bm25_results = self._bm25_search(
            query,
            candidate_limit,
        )

        semantic_results = (
            self.semantic_search.search_chunks(
                query,
                candidate_limit,
            )
        )

        combined: dict[int, HybridResult] = {}

        # BM25 rankings start at 1.
        for rank, bm25_result in enumerate(
            bm25_results,
            start=1,
        ):
            movie, _ = bm25_result
            doc_id = movie["id"]

            combined[doc_id] = {
                "id": doc_id,
                "title": movie["title"],
                "document": movie["description"],
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_score": rrf_score(rank, k),
            }

        # Semantic rankings also start at 1.
        for rank, semantic_result in enumerate(
            semantic_results,
            start=1,
        ):
            doc_id = semantic_result["id"]

            if doc_id not in combined:
                combined[doc_id] = {
                    "id": doc_id,
                    "title": semantic_result["title"],
                    "document": semantic_result["document"],
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": rrf_score(rank, k),
                }
            else:
                combined[doc_id]["semantic_rank"] = rank
                combined[doc_id]["rrf_score"] += (
                    rrf_score(rank, k)
                )

        ranked_results = sorted(
            combined.values(),
            key=lambda result: result["rrf_score"],
            reverse=True,
        )

        return ranked_results[:limit]