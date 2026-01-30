# Periscope

[![Release](https://img.shields.io/github/v/release/trummelbummel/periscope)](https://img.shields.io/github/v/release/trummelbummel/periscope)
[![Build status](https://img.shields.io/github/actions/workflow/status/trummelbummel/periscope/main.yml?branch=main)](https://github.com/trummelbummel/periscope/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/trummelbummel/periscope/branch/main/graph/badge.svg)](https://codecov.io/gh/trummelbummel/periscope)
[![Commit activity](https://img.shields.io/github/commit-activity/m/trummelbummel/periscope)](https://img.shields.io/github/commit-activity/m/trummelbummel/periscope)
[![License](https://img.shields.io/github/license/trummelbummel/periscope)](https://img.shields.io/github/license/trummelbummel/periscope)

This is a module implementing a rag use case with cursor.

- **Github repository**: <https://github.com/trummelbummel/periscope/>
- **Documentation** <https://trummelbummel.github.io/periscope/>


# Documentation

To find them Deep research is often too high level and Google won't find anything of value either. That is why there is Periscope a RAG system containing information on the latests Arxiv papers on context engineering, chunked based on understanding of research paper structure can give us an edge to find information that will make a difference.

More information on this project can also be found in `./docs`.

## Configuration

Configuration is read from environment variables; you can use a `.env` file in the project root (loaded automatically).

**Required for the prototype to run:**

- **`HUGGINGFACE_TOKEN`** (or `HF_TOKEN`) – Used for answer generation via the Hugging Face Inference API. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**Recommended to set for the scraper:**

- **`ARXIV_DEFAULT_QUERY`** – arXiv search query used by `make run-scraper` (default is set in `config.py`; override to fetch papers for your topic).

**Optional settings** (see `config.py` for defaults and full list):

- `PORT`, `API_HOST` – API server
- `DATA_DIR`, `ARXIV_DATA_DIR` – Where PDFs are read from
- `GENERATION_MODEL` – Hugging Face model id for generation
- `EMBEDDING_MODEL` – Embedding model for the vector index
- `TOP_K` – Number of retrieved chunks per query
- `CHUNK_SIZE`, `CHUNK_OVERLAP` – Chunking for ingestion
- `ENABLE_GUARDRAILS`, `SIMILARITY_THRESHOLD` – When to abstain from answering
- `INDEX_VERSION` – Version string written to ingestion stats, retrieval evaluation JSON, and Chroma collection metadata (default `1`)
- `RETRIEVAL_EXPERIMENT_MAX_NODES`, `RETRIEVAL_EXPERIMENT_NUM_QUESTIONS_PER_CHUNK` – Max nodes and questions per chunk for `make run-monitoring` retrieval evaluation (defaults: `50`, `1`)
- `MIRO_ACCESS_TOKEN`, `MIRO_BOARD_ID` – For the Miro MCP server

## Getting started with your project

### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:trummelbummel/periscope.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 3. Ingest arXiv papers (one-time)

Before you can query the RAG API, you need to download and index a small corpus
of arXiv papers used as context. This is done via the bundled scraper:

```bash
make run-scraper
```

This command fetches relevant arXiv PDFs into the `data/arxiv/` folder. After
that, you can run ingestion (or let the API trigger it on first query).

### 4. Run the API

To start the RAG HTTP API (including the minimal UI under `/ui`), run:

```bash
make run-api
```

This will launch the FastAPI server on the configured `PORT` (default `8000`).
Open `http://localhost:8000/ui/` in your browser to use the UI, or `http://localhost:8000/docs`
for the interactive OpenAPI docs.

### 5. Run retrieval monitoring (optional)

To evaluate retrieval quality on the current index and write metrics to `monitoring/data/retrieval_evaluation.json`, run:

```bash
make run-monitoring
```

This loads the persisted Chroma index and BM25 nodes (or runs ingestion first if missing), then runs a retrieval experiment (hit rate, MRR, precision, recall, etc.) and saves the results. Use `RETRIEVAL_EXPERIMENT_MAX_NODES` and `RETRIEVAL_EXPERIMENT_NUM_QUESTIONS_PER_CHUNK` in config to control experiment size.

### 6. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 7. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version



---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
