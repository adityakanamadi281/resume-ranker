"""Script to pre-download and cache the default embedding model for offline usage."""

import sys
from sentence_transformers import SentenceTransformer


def main() -> None:
    model_name = "BAAI/bge-small-en-v1.5"
    print(f"Downloading and caching embedding model '{model_name}'...")
    try:
        # Initializing the SentenceTransformer downloads and caches the model
        SentenceTransformer(model_name)
        print("Model downloaded and cached successfully!")
    except Exception as e:
        print(f"ERROR: Failed to download model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
