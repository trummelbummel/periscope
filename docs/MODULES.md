# Top-level module documentation

Documentation is derived strictly from the codebase (docstrings, `__all__`, and public APIs).

---

## Entry points

### `main_api`

- **Purpose:** Command-line entrypoint for running the RAG API server (docstring: "Command-line entrypoint for running the RAG API server").
- **Key responsibilities:**
  - Run uvicorn with the FastAPI app from `periscope.app.api`.
  - Use host, port, and reload from config; when reload is on, watch the package source directory.
- **Important public interfaces:**
  - `run() -> None` — Run the uvicorn server with config host, port, and optional reload.

---

### `main_scraper`

- **Purpose:** Command-line entrypoint for running the arXiv scraper (docstring: "Command-line entrypoint for running the arXiv scraper").
- **Key responsibilities:**
  - Read default arXiv query and data directory from config.
  - Use `ArxivScraper` to fetch papers and store PDFs in the configured data directory.
  - Configure logging for CLI usage; log fetch and download results.
- **Important public interfaces:**
  - `run() -> None` — Run the arXiv scraper with configuration from config.

---

### `run_monitoring`

- **Purpose:** CLI entrypoint for monitoring: run retrieval evaluation and write results to disk (docstring).
- **Key responsibilities:**
  - Load persisted index (Chroma + BM25 nodes) from config paths; if missing or empty, run ingestion first.
  - On ingestion failure (e.g. `NoDocumentsError`), print message and return exit code 1.
  - Call `run_retrieval_experiment(vector_index, bm25_nodes)` and print the written path.
- **Important public interfaces:**
  - `main() -> int` — Load or build index, run retrieval evaluation, write JSON. Returns 0 on success, 1 on error.

---

## `app` package

### `app.api`

- **Purpose:** REST API service: expose endpoints for retrieval and generation workflow (docstring: "Per PRD: FastAPI, PORT = 8000, API_HOST = '0.0.0.0'").
- **Key responsibilities:**
  - Define FastAPI app with CORS, mount static UI at `/ui`, redirect `/` to `/ui/`.
  - Maintain in-memory `_vector_index` and `_bm25_nodes`; load from Chroma + pickle or run full ingestion when needed.
  - Compare current pipeline config (embedding_model, chunk_size, chunk_overlap, preprocessing_config) to persisted ingestion stats to decide whether to reuse index or re-ingest.
  - Expose `/health`, `POST /query`, `POST /ingest`; on query/ingest ensure index then delegate to `run_query` or return success.
- **Important public interfaces:**
  - `app` — FastAPI application instance.
  - `GET /` — Redirect to `/ui/`.
  - `GET /health` — Health check; returns `{"status": "ok"}`.
  - `POST /query` — Accept `QueryRequest`; ensure index, then call `run_query(...)`; return `QueryResponse`.
  - `POST /ingest` — Ensure index (trigger ingestion if needed); return `{"status": "ok", "message": "..."}`.

---

### `app.pipeline`

- **Purpose:** Orchestrate retrieval and generation with guardrails and observability (docstring: "Composes: hybrid retrieval -> guardrails -> answer generation. Returns structured QueryResponse with answer, sources, metadata (per PRD).").
- **Key responsibilities:**
  - Run hybrid retrieval via `HybridRetriever.hybrid_retrieve`.
  - Convert `NodeWithScore` to `RetrievedNode` (including `tables_display` when metadata has `tables`).
  - If guardrails enabled and `should_abstain(sources)` True, return `QueryResponse` with empty answer and `abstained=True`.
  - Otherwise call `AnswerGenerator.generate_answer_with_options(query, context_nodes=sources)` and return `QueryResponse` with answer, sources, metadata (retrieval_time_seconds, num_sources, generation_time_seconds), and `abstained=False`; on generation exception return response with `generation_error` in metadata.
- **Important public interfaces:**
  - `Pipeline` — Class that orchestrates retrieval, guardrails, and answer generation.
  - `Pipeline.run_query(query, vector_index, bm25_nodes, top_k=None) -> QueryResponse` — Static method implementing the full flow.
  - `run_query(query, vector_index, bm25_nodes, top_k=None) -> QueryResponse` — Module-level wrapper delegating to `Pipeline.run_query`.

---

### `pipeline` (top-level re-export)

- **Purpose:** Re-export pipeline for `periscope.pipeline` (docstring).
- **Key responsibilities:** Expose `run_query` from `periscope.app.pipeline`.
- **Important public interfaces:**
  - `run_query` — Same as `app.pipeline.run_query`.

---

## Config and models

### `config` (root `config.py`; package uses stub `periscope.config`)

- **Purpose:** Configuration for the periscope RAG service (docstring: "All paths, models, ports, and API keys are configurable via environment variables or defaults. Load with python-dotenv for .env support. Lives at project root (next to pyproject.toml).").
- **Key responsibilities:**
  - Define and export all settings from environment (with defaults): API (PORT, API_HOST, API_RELOAD), data paths (DATA_DIR, ARXIV_DATA_DIR), arXiv (ARXIV_DEFAULT_QUERY, ARXIV_MAX_RESULTS, ARXIV_API_BASE_URL, ARXIV_HTTP_TIMEOUT, ARXIV_USER_AGENT), document extensions, Chroma/INDEX paths, embedding/generation models and prompt, TOP_K, chunking, preprocessing flags, guardrails and SIMILARITY_THRESHOLD, INDEX_VERSION, INGESTION_STATS_PATH, RETRIEVAL_EXPERIMENT_*, Miro-related variables.
- **Important public interfaces:**
  - All uppercase names listed above (e.g. `PORT`, `API_HOST`, `CHROMA_PERSIST_DIR`, `EMBEDDING_MODEL`, `GENERATION_MODEL`, `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `INGESTION_STATS_PATH`, `RETRIEVAL_EXPERIMENT_MAX_NODES`, `RETRIEVAL_EXPERIMENT_NUM_QUESTIONS_PER_CHUNK`, etc.) — read by other modules.

---

### `models`

- **Purpose:** Re-export data models for `periscope.models` (docstring).
- **Key responsibilities:** Re-export Pydantic models from `periscope.data_models`.
- **Important public interfaces:**
  - `QueryRequest`, `QueryResponse`, `RetrievedNode`, `IngestionStats`, `ArxivResult` — as defined in `data_models`.

---

### `data_models`

- **Purpose:** Pydantic data contracts for components (docstring: "Used for data passed between components (API, retrieval, generation, etc.).").
- **Key responsibilities:**
  - Define request/response and stats models with validation and field descriptions.
- **Important public interfaces:**
  - `QueryRequest` — query (required), top_k.
  - `RetrievedNode` — text, score, node_id, metadata.
  - `QueryResponse` — answer, sources, metadata, abstained.
  - `IngestionStats` — document_count, chunk_count, total_chars, avg_chunk_size, paths, index_version, embedding_model, preprocessing_config, chunk_size, chunk_overlap.
  - `ArxivResult` — entry_id, title, summary, pdf_url, local_path.

---

## Re-exports (top-level)

### `embedder`

- **Purpose:** Re-export embedder from retriever for backward compatibility (docstring).
- **Key responsibilities:** Expose embedder helpers from `periscope.retriever.embedder`.
- **Important public interfaces:**
  - `get_embed_model(model_name=None) -> HuggingFaceEmbedding`
  - `set_global_embed_model(model_name=None) -> None`

---

### `vector_store`

- **Purpose:** Re-export vector store builder and persistence for `periscope.vector_store` (docstring).
- **Key responsibilities:** Expose vector-store and BM25 persistence from `periscope.retriever.vector_store`.
- **Important public interfaces:**
  - `build_index_from_nodes(nodes, persist_dir=None) -> VectorStoreIndex`
  - `load_index_from_chroma(persist_dir=None) -> VectorStoreIndex | None`
  - `persist_bm25_nodes(nodes, path=None) -> None`
  - `load_bm25_nodes(path=None) -> list[BaseNode] | None`

---

## `ingestion` package

- **Purpose (package):** Document loading, preprocessing, chunking, pipeline, and table extraction (from `ingestion/__init__.py` docstring).
- **Important public interfaces (from `__all__`):** `chunk_documents`, `get_header_aware_chunker`, `IngestionPipeline`, `IngestionResult`, `load_documents_from_directory`, `NoDocumentsError`, `preprocess_documents`, `PreprocessingConfig`, `read_pdf_path`, `run_ingestion`, `PdfTableExtractor`.

### `ingestion.ingestion_pipeline`

- **Purpose:** Run load → preprocess → chunk → index → persist → stats in sequence; index build stores embeddings in Chroma; BM25 nodes persisted to disk; stats written only after persist (docstring).
- **Key responsibilities:**
  - Load documents from DATA_DIR then ARXIV_DATA_DIR via `load_documents_from_directory`; preprocess with `PreprocessingConfig`; chunk with `chunk_documents`; call `set_global_embed_model`, then `build_index_from_nodes`, `persist_bm25_nodes`; compute and write ingestion stats via monitoring.
  - Raise `NoDocumentsError` when no documents found.
- **Important public interfaces:**
  - `NoDocumentsError` — ValueError when no documents in configured directories.
  - `IngestionResult` — dataclass: index (VectorStoreIndex), nodes, stats (IngestionStats).
  - `IngestionPipeline` — Constructor accepts optional overrides for data_dir, arxiv_data_dir, required_extensions, chunk_size, chunk_overlap, preprocessing_config, embedding_model, chroma_persist_dir, index_nodes_path, ingestion_stats_path; defaults from config. `run() -> IngestionResult`; raises `NoDocumentsError` if no docs.
  - `run_ingestion(data_dir=None, arxiv_data_dir=None, **kwargs) -> IngestionResult` — Convenience; builds pipeline and runs.

### `ingestion.document_reader`

- **Purpose:** Read and process PDF documents from the data directory using Docling; support PDF format (docstring).
- **Key responsibilities:**
  - Load documents from a directory as LlamaIndex Documents via Docling DocumentConverter; extract text (Markdown with layout and tables), section headers, and tables; add headers and tables to document metadata.
  - Single-file extraction via `read_pdf_path(path)` returns Markdown. `load_documents()` returns one Document per file with metadata: `file_path`, `headers` (section headers and titles), `tables` (table HTML).
- **Important public interfaces:**
  - `DocumentReader` — Constructor: directory, required_extensions (defaults from config). Methods: `read_pdf_path(path) -> str`, `load_documents() -> list[Document]`; static: `read_pdf_path_default(path)`, `load_documents_from_directory_default(directory, required_extensions)`.
  - `read_pdf_path(path) -> str` — Module-level.
  - `load_documents_from_directory(directory=None, required_extensions=None) -> list[Document]` — Module-level.

### `ingestion.preprocessor`

- **Purpose:** Remove noise from document text before chunking: tables, footnotes, inline citations, reference sections (docstring).
- **Key responsibilities:**
  - Provide configurable text cleaning via regex-based strip functions; return new Documents with cleaned text and preserved metadata.
- **Important public interfaces:**
  - `PreprocessingConfig` — dataclass: remove_tables, remove_footnotes, remove_inline_citations, remove_reference_section; `to_dict()` for serialization.
  - `clean_text(text, config: PreprocessingConfig) -> str` — Remove noise according to config.
  - `preprocess_documents(documents, config: PreprocessingConfig) -> list[Document]` — Run preprocessing on each document; return new list with cleaned text.

### `ingestion.chunker`

- **Purpose:** Header-aware chunking for research paper structure; use Llama-index (docstring).
- **Key responsibilities:**
  - Split documents into nodes using SentenceSplitter with configurable chunk_size and chunk_overlap (defaults from config), paragraph_separator `\n\n`.
- **Important public interfaces:**
  - `HeaderAwareChunker` — Constructor: chunk_size, chunk_overlap (defaults from config). Methods: `chunk_documents(documents) -> list[BaseNode]`; property `parser` (SentenceSplitter). Static: `get_header_aware_chunker(...) -> SentenceSplitter`, `chunk_documents_with_options(...) -> list[BaseNode]`.
  - `PARAGRAPH_SEPARATOR` — `"\n\n"`.
  - `get_header_aware_chunker(chunk_size=None, chunk_overlap=None) -> SentenceSplitter`
  - `chunk_documents(documents, chunk_size=None, chunk_overlap=None) -> list[BaseNode]`

### `ingestion.table_extractor`

- **Purpose:** Table extraction from PDFs using pdfplumber; attach tables to page-level sections (docstring).
- **Key responsibilities:**
  - Extract tables per page; build Documents per page with text and tables in metadata.
- **Important public interfaces:**
  - `TableData` — type alias: list of rows, row = list of cell strings.
  - `PdfTableExtractor` — Methods: `extract_tables_from_pdf(path) -> dict[int, list[TableData]]`, `documents_from_pdf_with_tables(path) -> list[Document]`.

---

## `retriever` package

### `retriever.retriever`

- **Purpose:** Hybrid retrieval: combine keyword search (BM25) and vector similarity search; TOP_K from config (docstring).
- **Key responsibilities:**
  - Run vector retriever (from VectorStoreIndex) and BM25 retriever over provided nodes; merge results with reciprocal rank fusion (RRF, k=60); deduplicate by node_id; return top_k NodeWithScore. Uses `set_global_embed_model` before retrieval.
- **Important public interfaces:**
  - `HybridRetriever` — Constructor: vector_index, bm25_nodes, top_k (default from config). Method: `retrieve(query) -> list[NodeWithScore]`. Static: `_resolve_top_k`, `_reciprocal_rank_fusion`, `get_vector_retriever`, `get_bm25_retriever`, `hybrid_retrieve(query, vector_index, bm25_nodes, top_k=None) -> list[NodeWithScore]`.
  - `hybrid_retrieve(...)` — Module-level.
  - `get_vector_retriever(index, top_k=None)`, `get_bm25_retriever(nodes, top_k=None)` — Module-level.

### `retriever.vector_store`

- **Purpose:** Vector database storage for embedded chunks; ChromaDB for PoC; index in Chroma, BM25 nodes at INDEX_NODES_PATH (docstring).
- **Key responsibilities:**
  - Create Chroma persistent client and collection (with index_version in metadata); build VectorStoreIndex from nodes (embed and store); load existing index from Chroma; persist/load BM25 nodes via pickle.
- **Important public interfaces:**
  - `ChromaIndexBuilder` — Constructor: persist_dir (default from config). Methods: `get_chroma_vector_store() -> ChromaVectorStore`, `build_index_from_nodes(nodes) -> VectorStoreIndex`; static defaults for both.
  - `build_index_from_nodes(nodes, persist_dir=None) -> VectorStoreIndex`
  - `load_index_from_chroma(persist_dir=None) -> VectorStoreIndex | None`
  - `persist_bm25_nodes(nodes, path=None) -> None`
  - `load_bm25_nodes(path=None) -> list[BaseNode] | None`
  - `get_chroma_vector_store(persist_dir=None) -> ChromaVectorStore`

### `retriever.embedder`

- **Purpose:** Generate embeddings for chunked documents; use Llama-index; low latency with small embedding model (docstring).
- **Key responsibilities:**
  - Provide HuggingFace embedding model (model id from config); set LlamaIndex global Settings.embed_model for pipelines.
- **Important public interfaces:**
  - `Embedder` — Constructor: model_name (default from config). Methods: `get_embed_model() -> HuggingFaceEmbedding`, `set_global_embed_model()`; static defaults for both.
  - `get_embed_model(model_name=None) -> HuggingFaceEmbedding`
  - `set_global_embed_model(model_name=None) -> None`

---

## `generation` package

### `generation.generator`

- **Purpose:** Generate responses using LLM based on retrieved context; use GENERATION_MODEL and GENERATION_PROMPT; Hugging Face Inference API (serverless); error handling for model calls (docstring).
- **Key responsibilities:**
  - Build context string from RetrievedNode list (including formatted tables from metadata); format prompt with context_str and query_str; call LLM complete; return answer text. Support injected LLM or create from config (model, token).
- **Important public interfaces:**
  - `format_tables_for_display(tables: TableData) -> str` — Format tables as markdown.
  - `AnswerGenerator` — Constructor: model, token, prompt_template, llm (defaults from config). Methods: `generate_answer(query, context_nodes) -> str`, `_get_llm()`. Static: `_build_context_str`, `get_llm(model=None, token=None) -> HuggingFaceInferenceAPI`, `generate_answer_with_options(query, context_nodes, prompt_template=None, llm=None) -> str`.
  - `get_llm(...)`, `generate_answer(...)` — Module-level wrappers.

---

## `monitoring` package

- **Important public interfaces (from `__all__`):** `compute_ingestion_stats`, `read_ingestion_stats`, `run_retrieval_experiment`, `should_abstain`, `write_ingestion_stats`.

### `monitoring.monitoring`

- **Purpose:** Ingestion statistics and observability; stats written only after index and BM25 persist so they reflect persisted data (docstring).
- **Key responsibilities:**
  - Compute IngestionStats from counts and config (including index_version from config); write/read JSON to INGESTION_STATS_PATH. Run retrieval experiment: subset nodes (max_nodes, num_questions_per_chunk from config), generate synthetic QA with AnswerGenerator LLM, run RetrieverEvaluator (hit_rate, mrr, precision, recall, ap, ndcg) over vector retriever, aggregate metrics, write JSON with index_version and timestamp.
- **Important public interfaces:**
  - `IngestionStatsWriter` — Constructor: output_path (default from config). Methods: `compute_ingestion_stats(...) -> IngestionStats`, `write_ingestion_stats(stats, output_path=None)`; static `compute_ingestion_stats_default`, `write_ingestion_stats_default`.
  - `compute_ingestion_stats(...) -> IngestionStats` — Module-level; supports index_version param (default from config).
  - `read_ingestion_stats(output_path=None) -> IngestionStats | None`
  - `write_ingestion_stats(stats, output_path=None)`
  - `RetrievalExperiment` — Constructor: output_path, num_questions_per_chunk, max_nodes. Method: `run(vector_index, bm25_nodes) -> Path`.
  - `run_retrieval_experiment(vector_index, bm25_nodes, output_path=None, num_questions_per_chunk=None, max_nodes=None) -> Path` — Uses config defaults when args omitted.

### `monitoring.guardrails`

- **Purpose:** Safety mechanisms for generation; abstain from generation if similarity threshold not met (docstring).
- **Key responsibilities:**
  - Decide whether to abstain based on retrieved nodes: abstain if no nodes or best score < threshold (from config).
- **Important public interfaces:**
  - `Guardrails` — Constructor: threshold (default from config). Method: `should_abstain(nodes: list[RetrievedNode]) -> bool`. Static: `should_abstain_with_options(nodes, threshold=None)`.
  - `should_abstain(nodes, threshold=None) -> bool` — Module-level.

---

## `scraper` package

### `scraper.arxiv_scraper`

- **Purpose:** Simple arXiv scraper using the public API: send query, parse Atom XML into ArxivPaper, optionally download PDFs (docstring).
- **Key responsibilities:**
  - Fetch papers via httpx GET to config ARXIV_API_BASE_URL with search_query, max_results, start; parse Atom feed into ArxivPaper (id, title, summary, authors, pdf_url); download PDFs to a directory (stream and write to disk); use config for timeout, user-agent, default query/max_results.
- **Important public interfaces:**
  - `ArxivPaper` — dataclass: id, title, summary, authors, pdf_url; property `display_id`, method `filename()`.
  - `ArxivScraper` — Constructor: client (optional httpx.Client). Methods: `fetch_papers(query, max_results=None, start=0) -> list[ArxivPaper]`, `download_pdfs(papers, download_dir=None) -> list[Path]`, `fetch_default_from_config() -> list[ArxivPaper]`.
