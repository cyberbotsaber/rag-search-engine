from typing import Any


SCORE_PRECISION = 4


def format_search_result(
    doc_id: int,
    title: str,
    document: str,
    score: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata or {},
    }
