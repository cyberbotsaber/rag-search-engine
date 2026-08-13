from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from .semantic_search import load_movies


Movie = dict[str, object]
MultimodalResult = dict[str, object]


class MultimodalSearch:
    def __init__(
        self,
        documents: list[Movie],
        model_name: str = "clip-ViT-B-32",
    ) -> None:
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.texts = [
            f"{document['title']}: {document['description']}"
            for document in documents
        ]
        self.text_embeddings = (
            self.model.encode(
                self.texts,
                show_progress_bar=True,
            )
            if self.texts
            else np.empty((0, 0))
        )

    def embed_image(
        self,
        image_path: str | Path,
    ) -> np.ndarray:
        with Image.open(image_path) as image:
            embedding = self.model.encode([image])[0]

        return embedding

    def search_with_image(
        self,
        image_path: str | Path,
    ) -> list[MultimodalResult]:
        image_embedding = self.embed_image(image_path)
        image_norm = np.linalg.norm(image_embedding)
        scored_documents: list[tuple[float, Movie]] = []

        for text_embedding, document in zip(
            self.text_embeddings,
            self.documents,
        ):
            text_norm = np.linalg.norm(text_embedding)

            if image_norm == 0 or text_norm == 0:
                similarity = 0.0
            else:
                similarity = float(
                    np.dot(text_embedding, image_embedding)
                    / (text_norm * image_norm)
                )

            scored_documents.append((similarity, document))

        scored_documents.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            {
                "id": document["id"],
                "title": document["title"],
                "description": document["description"],
                "similarity": similarity,
            }
            for similarity, document in scored_documents[:5]
        ]


def verify_image_embedding(image_path: str | Path) -> None:
    multimodal_search = MultimodalSearch([])
    embedding = multimodal_search.embed_image(image_path)
    print(
        f"Embedding shape: {embedding.shape[0]} dimensions"
    )


def image_search_command(
    image_path: str | Path,
) -> list[MultimodalResult]:
    documents = load_movies()
    multimodal_search = MultimodalSearch(documents)

    return multimodal_search.search_with_image(image_path)
