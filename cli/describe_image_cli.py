import argparse
import base64
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal Movie Query Rewriting CLI"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to an image file",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Text query to rewrite based on the image",
    )

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as image_file:
        image = image_file.read()

    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable not set"
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Identify and include exact character, movie, actor, or franchise names whenever they are recognizable
- Use the image filename as a clue when it contains a meaningful movie or character name
- Return one short keyword-style search query only
- Do not include an explanation, introduction, labels, or quotation marks
"""

    data_url = (
        f"data:{mime};base64,"
        f"{base64.b64encode(image).decode()}"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt.strip(),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
                {
                    "type": "text",
                    "text": (
                        f"Image filename: {Path(args.image).name}\n"
                        f"Text query: {args.query.strip()}"
                    ),
                },
            ],
        }
    ]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("The model returned an empty query")

    print(f"Rewritten query: {content.strip()}")

    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
