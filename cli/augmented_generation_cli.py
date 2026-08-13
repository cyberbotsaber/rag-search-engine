import argparse
import sys
from pathlib import Path
from typing import Any


CLI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLI_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CLI_DIR))


from lib.hybrid_search import HybridSearch
from lib.query_enhancer import get_openrouter_client
from lib.semantic_search import load_movies


def format_documents(results: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"{index}. {result['title']}\n{result['document']}"
        for index, result in enumerate(results, start=1)
    )


def search_movies(
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    return hybrid_search.rrf_search(
        query,
        k=60,
        limit=limit,
    )


def generate_answer(
    query: str,
    results: list[dict[str, Any]],
) -> str:
    docs = format_documents(results)

    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""

    client = get_openrouter_client()
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError("The LLM returned an empty response")

    return answer.strip()


def rag_command(query: str) -> None:
    results = search_movies(query)
    answer = generate_answer(query, results)

    print("Search Results:")

    for result in results:
        print(f"- {result['title']}")

    print("\nRAG Response:")
    print(answer)


def generate_summary(
    query: str,
    results: list[dict[str, Any]],
) -> str:
    formatted_results = format_documents(results)

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{formatted_results}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    client = get_openrouter_client()
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    summary = response.choices[0].message.content

    if not summary:
        raise RuntimeError("The LLM returned an empty summary")

    return summary.strip()


def summarize_command(query: str, limit: int = 5) -> None:
    results = search_movies(query, limit)
    summary = generate_summary(query, results)

    print("Search Results:")

    for result in results:
        print(f"  - {result['title']}")

    print("\nLLM Summary:")
    print(summary)


def generate_cited_answer(
    query: str,
    results: list[dict[str, Any]],
) -> str:
    documents = format_documents(results)

    prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    client = get_openrouter_client()
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError("The LLM returned an empty answer")

    return answer.strip()


def citations_command(query: str, limit: int = 5) -> None:
    results = search_movies(query, limit)
    answer = generate_cited_answer(query, results)

    print("Search Results:")

    for result in results:
        print(f"  - {result['title']}")

    print("\nLLM Answer:")
    print(answer)


def generate_question_answer(
    question: str,
    results: list[dict[str, Any]],
) -> str:
    context = format_documents(results)

    prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

    client = get_openrouter_client()
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError("The LLM returned an empty answer")

    return answer.strip()


def question_command(question: str, limit: int = 5) -> None:
    results = search_movies(question, limit)
    answer = generate_question_answer(question, results)

    print("Search Results:")

    for result in results:
        print(f"  - {result['title']}")

    print("\nAnswer:")
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval Augmented Generation CLI"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    rag_parser = subparsers.add_parser(
        "rag",
        help="Perform RAG (search + generate answer)",
    )
    rag_parser.add_argument(
        "query",
        type=str,
        help="Search query for RAG",
    )

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize multiple search results",
    )
    summarize_parser.add_argument(
        "query",
        type=str,
        help="Search query to summarize",
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of search results",
    )

    citations_parser = subparsers.add_parser(
        "citations",
        help="Answer a query with citations to search results",
    )
    citations_parser.add_argument(
        "query",
        type=str,
        help="Search query to answer with citations",
    )
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of search results",
    )

    question_parser = subparsers.add_parser(
        "question",
        help="Answer a question using search results",
    )
    question_parser.add_argument(
        "question",
        type=str,
        help="Question to answer",
    )
    question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of search results",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            rag_command(args.query)
        case "summarize":
            summarize_command(args.query, args.limit)
        case "citations":
            citations_command(args.query, args.limit)
        case "question":
            question_command(args.question, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
