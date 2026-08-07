import json
import re
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import format_search_result


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MOVIES_PATH = PROJECT_ROOT / "data" / "movies.json"

CACHE_DIR = PROJECT_ROOT / "cache"

EMBEDDINGS_PATH = CACHE_DIR / "movie_embeddings.npy"
CHUNK_EMBEDDINGS_PATH = CACHE_DIR / "chunk_embeddings.npy"
CHUNK_METADATA_PATH = CACHE_DIR / "chunk_metadata.json"


Movie = dict[str, Any]
SearchResult = dict[str, Any]


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


def cosine_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray,
) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def semantic_chunk_text(
    text: str,
    max_chunk_size: int = 4,
    overlap: int = 0,
) -> list[str]:
    if max_chunk_size <= 0:
        raise ValueError(
            "Max chunk size must be greater than zero."
        )

    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")

    if overlap >= max_chunk_size:
        raise ValueError(
            "Overlap must be smaller than max chunk size."
        )

    cleaned_text = text.strip()

    if not cleaned_text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        cleaned_text,
    )

    if (
        len(sentences) == 1
        and not cleaned_text.endswith((".", "!", "?"))
    ):
        sentences = [cleaned_text]

    cleaned_sentences: list[str] = []

    for sentence in sentences:
        cleaned_sentence = sentence.strip()

        if cleaned_sentence:
            cleaned_sentences.append(cleaned_sentence)

    chunks: list[str] = []

    start = 0
    step = max_chunk_size - overlap

    while start < len(cleaned_sentences):
        if (
            start > 0
            and start + overlap >= len(cleaned_sentences)
        ):
            break

        end = start + max_chunk_size

        chunk = " ".join(
            cleaned_sentences[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


class SemanticSearch:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.model = SentenceTransformer(
            model_name,
            device="cpu",
        )

        self.embeddings: np.ndarray | None = None
        self.documents: list[Movie] | None = None
        self.document_map: dict[int, Movie] = {}

    def generate_embedding(
        self,
        text: str,
    ) -> np.ndarray:
        if not text.strip():
            raise ValueError(
                "Text cannot be empty or contain only whitespace."
            )

        embeddings = self.model.encode([text])

        return embeddings[0]

    def build_embeddings(
        self,
        documents: list[Movie],
    ) -> np.ndarray:
        self.documents = documents

        self.document_map = {
            document["id"]: document
            for document in documents
        }

        document_texts = [
            f"{document['title']}: {document['description']}"
            for document in documents
        ]

        self.embeddings = self.model.encode(
            document_texts,
            show_progress_bar=True,
        )

        CACHE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            EMBEDDINGS_PATH,
            self.embeddings,
        )

        return self.embeddings

    def load_or_create_embeddings(
        self,
        documents: list[Movie],
    ) -> np.ndarray:
        self.documents = documents

        self.document_map = {
            document["id"]: document
            for document in documents
        }

        if EMBEDDINGS_PATH.exists():
            cached_embeddings = np.load(
                EMBEDDINGS_PATH
            )

            if len(cached_embeddings) == len(documents):
                self.embeddings = cached_embeddings

                return self.embeddings

        return self.build_embeddings(documents)

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SearchResult]:
        if (
            self.embeddings is None
            or self.documents is None
        ):
            raise ValueError(
                "No embeddings loaded. "
                "Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        scored_documents: list[
            tuple[float, Movie]
        ] = []

        for embedding, document in zip(
            self.embeddings,
            self.documents,
        ):
            score = cosine_similarity(
                query_embedding,
                embedding,
            )

            scored_documents.append(
                (score, document)
            )

        scored_documents.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        results: list[SearchResult] = []

        for score, document in scored_documents[:limit]:
            results.append(
                {
                    "score": score,
                    "title": document["title"],
                    "description": document["description"],
                }
            )

        return results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        super().__init__(model_name)

        self.chunk_embeddings: np.ndarray | None = None
        self.chunk_metadata: list[ChunkMetadata] | None = None

    def build_chunk_embeddings(
        self,
        documents: list[Movie],
    ) -> np.ndarray:
        self.documents = documents

        self.document_map = {
            document["id"]: document
            for document in documents
        }

        all_chunks: list[str] = []
        chunk_metadata: list[ChunkMetadata] = []

        for movie_idx, document in enumerate(documents):
            description = document.get(
                "description",
                "",
            )

            if not description.strip():
                continue

            chunks = semantic_chunk_text(
                description,
                max_chunk_size=4,
                overlap=1,
            )

            total_chunks = len(chunks)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)

                chunk_metadata.append(
                    {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": total_chunks,
                    }
                )

        self.chunk_embeddings = self.model.encode(
            all_chunks,
            show_progress_bar=True,
        )

        self.chunk_metadata = chunk_metadata

        CACHE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            CHUNK_EMBEDDINGS_PATH,
            self.chunk_embeddings,
        )

        with open(
            CHUNK_METADATA_PATH,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "chunks": chunk_metadata,
                    "total_chunks": len(all_chunks),
                },
                file,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(
        self,
        documents: list[Movie],
    ) -> np.ndarray:
        self.documents = documents

        self.document_map = {
            document["id"]: document
            for document in documents
        }

        if (
            CHUNK_EMBEDDINGS_PATH.exists()
            and CHUNK_METADATA_PATH.exists()
        ):
            self.chunk_embeddings = np.load(
                CHUNK_EMBEDDINGS_PATH
            )

            with open(
                CHUNK_METADATA_PATH,
                "r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(file)

            self.chunk_metadata = metadata["chunks"]

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if (
            self.chunk_embeddings is None
            or self.chunk_metadata is None
            or self.documents is None
        ):
            raise ValueError(
                "No chunk embeddings loaded. "
                "Call `load_or_create_chunk_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        chunk_scores: list[dict[str, Any]] = []

        for chunk_embedding, metadata in zip(
            self.chunk_embeddings,
            self.chunk_metadata,
        ):
            score = cosine_similarity(
                query_embedding,
                chunk_embedding,
            )

            chunk_scores.append(
                {
                    "chunk_idx": metadata["chunk_idx"],
                    "movie_idx": metadata["movie_idx"],
                    "score": score,
                }
            )

        movie_scores: dict[int, dict[str, Any]] = {}

        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]

            if (
                movie_idx not in movie_scores
                or chunk_score["score"]
                > movie_scores[movie_idx]["score"]
            ):
                movie_scores[movie_idx] = chunk_score

        sorted_scores = sorted(
            movie_scores.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        top_scores = sorted_scores[:limit]

        results: list[dict[str, Any]] = []

        for movie_score in top_scores:
            movie_idx = movie_score["movie_idx"]
            document = self.documents[movie_idx]

            result = format_search_result(
                doc_id=document["id"],
                title=document["title"],
                document=document["description"][:100],
                score=movie_score["score"],
                metadata={
                    "chunk_idx": movie_score["chunk_idx"],
                },
            )

            results.append(result)

        return results


def load_movies() -> list[Movie]:
    with open(
        MOVIES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return data["movies"]


def verify_model() -> None:
    semantic_search = SemanticSearch()

    print(
        f"Model loaded: {semantic_search.model}"
    )

    print(
        "Max sequence length: "
        f"{semantic_search.model.max_seq_length}"
    )


def embed_text(text: str) -> None:
    semantic_search = SemanticSearch()

    embedding = semantic_search.generate_embedding(
        text
    )

    print(f"Text: {text}")
    print(
        f"First 3 dimensions: {embedding[:3]}"
    )
    print(
        f"Dimensions: {embedding.shape[0]}"
    )


def embed_query_text(query: str) -> None:
    semantic_search = SemanticSearch()

    embedding = semantic_search.generate_embedding(
        query
    )

    print(f"Query: {query}")
    print(
        f"First 3 dimensions: {embedding[:3]}"
    )
    print(
        f"Shape: {embedding.shape}"
    )


def verify_embeddings() -> None:
    semantic_search = SemanticSearch()

    documents = load_movies()

    embeddings = (
        semantic_search.load_or_create_embeddings(
            documents
        )
    )

    print(
        f"Number of docs:   {len(documents)}"
    )

    print(
        f"Embeddings shape: "
        f"{embeddings.shape[0]} vectors "
        f"in {embeddings.shape[1]} dimensions"
    )