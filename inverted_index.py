import json
import math
import os
import pickle
from collections import Counter
from typing import Any

from constants import BM25_B, BM25_K1
from text_processing import preprocess_text


CACHE_DIR = "cache"

Movie = dict[str, Any]


def load_movies() -> list[Movie]:
    with open("data/movies.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["movies"]


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, Movie] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.doc_lengths: dict[int, int] = {}

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(
            CACHE_DIR,
            "term_frequencies.pkl",
        )
        self.doc_lengths_path = os.path.join(
            CACHE_DIR,
            "doc_lengths.pkl",
        )

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = preprocess_text(text)

        self.term_frequencies[doc_id] = Counter()
        self.doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0

        total_length = sum(self.doc_lengths.values())
        return total_length / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        document_ids = self.index.get(term, set())
        return sorted(document_ids)

    def get_tf(self, doc_id: int, term: str) -> int:
        document_counter = self.term_frequencies.get(doc_id)

        if document_counter is None:
            return 0

        return document_counter.get(term, 0)

    def get_bm25_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        document_frequency = len(self.get_documents(term))

        return math.log(
            (
                total_doc_count
                - document_frequency
                + 0.5
            )
            / (document_frequency + 0.5)
            + 1
        )

    def get_bm25_tf(
        self,
        doc_id: int,
        term: str,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> float:
        tf = self.get_tf(doc_id, term)

        if tf == 0:
            return 0.0

        avg_doc_length = self.__get_avg_doc_length()

        if avg_doc_length == 0:
            return 0.0

        doc_length = self.doc_lengths.get(doc_id, 0)

        length_norm = (
            1
            - b
            + b * (doc_length / avg_doc_length)
        )

        return (
            tf * (k1 + 1)
        ) / (
            tf + k1 * length_norm
        )

    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)

        return bm25_tf * bm25_idf

    def bm25_search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[tuple[Movie, float]]:
        query_tokens = preprocess_text(query)
        scores: dict[int, float] = {}

        for doc_id in self.docmap:
            total_score = 0.0

            for token in query_tokens:
                total_score += self.bm25(doc_id, token)

            if total_score > 0:
                scores[doc_id] = total_score

        ranked_documents = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )

        results: list[tuple[Movie, float]] = []

        for doc_id, score in ranked_documents[:limit]:
            movie = self.docmap[doc_id]
            results.append((movie, score))

        return results

    def build(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

        movies = load_movies()

        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"

            self.docmap[doc_id] = movie
            self.__add_document(doc_id, text)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.index_path, "wb") as file:
            pickle.dump(self.index, file)

        with open(self.docmap_path, "wb") as file:
            pickle.dump(self.docmap, file)

        with open(
            self.term_frequencies_path,
            "wb",
        ) as file:
            pickle.dump(self.term_frequencies, file)

        with open(self.doc_lengths_path, "wb") as file:
            pickle.dump(self.doc_lengths, file)

    def load(self) -> None:
        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file)

        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)

        with open(
            self.term_frequencies_path,
            "rb",
        ) as file:
            self.term_frequencies = pickle.load(file)

        with open(self.doc_lengths_path, "rb") as file:
            self.doc_lengths = pickle.load(file)