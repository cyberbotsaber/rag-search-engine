import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


def get_openrouter_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable not set"
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def correct_spelling(query: str) -> str:
    client = get_openrouter_client()

    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    enhanced_query = response.choices[0].message.content

    if not enhanced_query:
        return query

    return enhanced_query.strip()


def rewrite_query(query: str) -> str:
    client = get_openrouter_client()

    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    enhanced_query = response.choices[0].message.content

    if not enhanced_query:
        return query

    return enhanced_query.strip()


def expand_query(query: str) -> str:
    client = get_openrouter_client()

    prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    additional_terms = response.choices[0].message.content

    if not additional_terms:
        return query

    additional_terms = additional_terms.strip()

    if not additional_terms:
        return query

    return f"{query} {additional_terms}"


def rerank_individual(
    query: str,
    documents: list[dict[str, Any]],
    sleep_seconds: float = 3.0,
) -> list[dict[str, Any]]:
    client = get_openrouter_client()

    reranked_results: list[dict[str, Any]] = []

    for index, doc in enumerate(documents):
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""

        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        content = response.choices[0].message.content

        try:
            rerank_score = (
                float(content.strip())
                if content
                else 0.0
            )
        except ValueError:
            rerank_score = 0.0

        rerank_score = max(
            0.0,
            min(10.0, rerank_score),
        )

        result = doc.copy()
        result["rerank_score"] = rerank_score

        reranked_results.append(result)

        if index < len(documents) - 1:
            time.sleep(sleep_seconds)

    reranked_results.sort(
        key=lambda result: result["rerank_score"],
        reverse=True,
    )

    return reranked_results


def rerank_batch(
    query: str,
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    client = get_openrouter_client()

    doc_lines = []

    for doc in documents:
        doc_lines.append(
            f'{doc.get("id")}: '
            f'{doc.get("title", "")} - '
            f'{doc.get("document", "")}'
        )

    doc_list_str = "\n".join(doc_lines)

    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown.
Do not use a JSON code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    content = response.choices[0].message.content

    if not content:
        return documents

    try:
        ranked_ids = json.loads(content.strip())
    except json.JSONDecodeError:
        return documents

    if not isinstance(ranked_ids, list):
        return documents

    rank_map: dict[int, int] = {}

    for rank, movie_id in enumerate(
        ranked_ids,
        start=1,
    ):
        if isinstance(movie_id, int):
            rank_map[movie_id] = rank

    fallback_rank = len(documents) + 1

    reranked_results: list[dict[str, Any]] = []

    for doc in documents:
        result = doc.copy()

        result["rerank_rank"] = rank_map.get(
            doc["id"],
            fallback_rank,
        )

        reranked_results.append(result)

    reranked_results.sort(
        key=lambda result: result["rerank_rank"]
    )

    return reranked_results


def rerank_cross_encoder(
    query: str,
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs: list[list[str]] = []

    for doc in documents:
        pairs.append(
            [
                query,
                (
                    f"{doc.get('title', '')} - "
                    f"{doc.get('document', '')}"
                ),
            ]
        )

    cross_encoder = CrossEncoder(
        "cross-encoder/ms-marco-TinyBERT-L2-v2",
        device="cpu",
    )

    scores = cross_encoder.predict(pairs)

    reranked_results: list[dict[str, Any]] = []

    for doc, score in zip(
        documents,
        scores,
    ):
        result = doc.copy()
        result["cross_encoder_score"] = float(score)

        reranked_results.append(result)

    reranked_results.sort(
        key=lambda result: result["cross_encoder_score"],
        reverse=True,
    )

    return reranked_results