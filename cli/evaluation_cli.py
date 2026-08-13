import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from lib.hybrid_search import HybridSearch
from lib.semantic_search import load_movies


GOLDEN_DATASET_PATH = (
    PROJECT_ROOT / "data" / "golden_dataset.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search Evaluation CLI"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help=(
            "Number of results to evaluate "
            "(k for precision@k, recall@k)"
        ),
    )

    args = parser.parse_args()
    limit = args.limit

    with GOLDEN_DATASET_PATH.open(encoding="utf-8") as file:
        golden_dataset: dict[str, Any] = json.load(file)

    documents = load_movies()
    hybrid_search = HybridSearch(documents)

    print(f"k={limit}\n")

    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        relevant_titles = test_case["relevant_docs"]
        relevant_title_set = set(relevant_titles)

        results = hybrid_search.rrf_search(
            query,
            k=60,
            limit=limit,
        )
        retrieved_titles = [
            result["title"] for result in results
        ]
        relevant_retrieved = sum(
            title in relevant_title_set
            for title in retrieved_titles
        )
        precision = (
            relevant_retrieved / len(retrieved_titles)
            if retrieved_titles
            else 0.0
        )
        recall = (
            relevant_retrieved / len(relevant_titles)
            if relevant_titles
            else 0.0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1_score:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_titles)}")
        print()


if __name__ == "__main__":
    main()
