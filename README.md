# RAG Search Engine

A command-line movie search engine that demonstrates the retrieval layer of a
retrieval-augmented generation (RAG) system. It supports lexical, semantic, and
hybrid retrieval over movie titles and descriptions.

## Current features

- Text preprocessing with lowercasing, punctuation removal, stop-word
  filtering, and Porter stemming
- A persistent inverted index containing document mappings, term frequencies,
  and document lengths
- Basic keyword lookup with duplicate removal
- TF, IDF, and TF-IDF inspection commands
- BM25 ranking with configurable `k1` and `b` scoring parameters
- Semantic search using Sentence Transformers and cosine similarity
- Whole-document and sentence-chunk embeddings
- Fixed-size word chunking and overlapping sentence-based chunking
- Cached indexes, embeddings, and chunk metadata for faster subsequent runs
- Weighted hybrid search with min-max normalized BM25 and semantic scores
- Reciprocal Rank Fusion (RRF) hybrid search

The included search commands operate on movie records, but the retrieval
components can be adapted to other document collections.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/) (recommended), or another Python package
  manager
- A local movie dataset and stop-word file in `data/`

The semantic features use the `all-MiniLM-L6-v2` Sentence Transformers model on
the CPU. The model may be downloaded the first time a semantic or hybrid command
is run.

## Installation

Clone the repository, enter the project directory, and install the locked
dependencies:

```bash
uv sync
```

If you are not using `uv`, create a Python 3.14 virtual environment and install
the dependencies declared in `pyproject.toml`.

## Data files

The `data/` directory is ignored by Git. Create these files locally before using
the search commands:

```text
data/
├── movies.json
└── stopwords.txt
```

`movies.json` must contain a top-level `movies` array. Each movie needs a unique
integer `id`, a `title`, and a `description`:

```json
{
  "movies": [
    {
      "id": 1,
      "title": "Example Movie",
      "description": "A short description of the movie."
    }
  ]
}
```

Add stop words to `stopwords.txt`, separated by whitespace or lines.

## Usage

Run commands from the project root with `uv run`.

### Keyword and BM25 search

Build the inverted index before running keyword commands:

```bash
uv run python cli/keyword_search_cli.py build
```

The generated index is stored in `cache/`.

```bash
# Return up to five movies containing terms from the query
uv run python cli/keyword_search_cli.py search "space adventure"

# Rank matches with BM25
uv run python cli/keyword_search_cli.py bm25search "space adventure" --limit 10

# Inspect lexical statistics
uv run python cli/keyword_search_cli.py tf 1 "adventure"
uv run python cli/keyword_search_cli.py idf "adventure"
uv run python cli/keyword_search_cli.py tfidf 1 "adventure"
uv run python cli/keyword_search_cli.py bm25idf "adventure"

# Optional positional k1 and b values
uv run python cli/keyword_search_cli.py bm25tf 1 "adventure" 1.5 0.75
```

The default BM25 parameters are `k1 = 1.5` and `b = 0.75`.

### Semantic search

Whole-document semantic search embeds each movie's title and description:

```bash
uv run python cli/semantic_search_cli.py search "a hopeful story about robots" --limit 5
```

Chunked search splits descriptions into overlapping sentence groups, scores
each chunk, and keeps the best-scoring chunk for each movie:

```bash
uv run python cli/semantic_search_cli.py embed_chunks
uv run python cli/semantic_search_cli.py search_chunked "surviving alone in space" --limit 5
```

Additional semantic utilities are available for model verification, embedding
inspection, and chunking experiments:

```bash
uv run python cli/semantic_search_cli.py verify
uv run python cli/semantic_search_cli.py verify_embeddings
uv run python cli/semantic_search_cli.py embed_text "some text"
uv run python cli/semantic_search_cli.py embed_query "search query"
uv run python cli/semantic_search_cli.py chunk "one two three four" --chunk-size 2 --overlap 1
uv run python cli/semantic_search_cli.py semantic_chunk "First sentence. Second sentence." --max-chunk-size 1
```

### Hybrid search

Weighted search combines normalized BM25 and semantic scores. `alpha` controls
the keyword weight: `1.0` is entirely BM25 and `0.0` is entirely semantic.

```bash
uv run python cli/hybrid_search_cli.py weighted-search "space adventure" --alpha 0.6 --limit 5
```

RRF combines the positions of results in the BM25 and semantic rankings rather
than combining their raw scores:

```bash
uv run python cli/hybrid_search_cli.py rrf-search "space adventure" -k 60 --limit 5
```

You can also inspect the min-max normalization used by weighted search:

```bash
uv run python cli/hybrid_search_cli.py normalize 2 5 10
```

Hybrid search automatically builds a missing inverted index and creates or
loads the chunk embeddings.

## Cache behavior

Generated artifacts are written to `cache/`, which is ignored by Git:

- Pickled inverted-index data for keyword and BM25 search
- NumPy arrays for document and chunk embeddings
- JSON metadata mapping chunks back to movies

Delete the relevant cache files and rerun the build or embedding command after
changing the source dataset. Document embeddings are rebuilt automatically when
the movie count changes; chunk embeddings currently load whenever their cache
files exist.

## Project structure

```text
.
├── cli/
│   ├── keyword_search_cli.py
│   ├── semantic_search_cli.py
│   ├── hybrid_search_cli.py
│   └── lib/
│       ├── hybrid_search.py
│       ├── search_utils.py
│       └── semantic_search.py
├── constants.py
├── inverted_index.py
├── text_processing.py
└── pyproject.toml
```

## Scope

This project currently implements retrieval and ranking. It does not yet
generate answers with a large language model, expose an HTTP API, or provide a
graphical interface.
