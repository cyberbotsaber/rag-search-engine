import argparse
import sys
from pathlib import Path


CLI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLI_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CLI_DIR))


from lib.multimodal_search import (
    image_search_command,
    verify_image_embedding,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal Search CLI"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    verify_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Generate an image embedding and print its shape",
    )
    verify_parser.add_argument(
        "image_path",
        type=str,
        help="Path to an image file",
    )

    image_search_parser = subparsers.add_parser(
        "image_search",
        help="Search for movies using an image",
    )
    image_search_parser.add_argument(
        "image_path",
        type=str,
        help="Path to an image file",
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            results = image_search_command(args.image_path)

            for index, result in enumerate(results, start=1):
                print(
                    f"{index}. {result['title']} "
                    f"(similarity: {result['similarity']:.3f})"
                )
                print(f"   {result['description'][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
