"""CLI entrypoint for monitoring: run retrieval evaluation and write results to disk.

Loads the persisted index (Chroma + BM25 nodes), or runs ingestion first if missing,
then runs RetrievalExperiment and writes monitoring/data/retrieval_evaluation.json.
"""

from __future__ import annotations

import sys

from periscope.config import CHROMA_PERSIST_DIR, INDEX_NODES_PATH
from periscope.ingestion import NoDocumentsError, run_ingestion
from periscope.monitoring import run_retrieval_experiment
from periscope.vector_store import load_bm25_nodes, load_index_from_chroma


def main() -> int:
    """Load or build index, run retrieval evaluation, write JSON. Returns 0 on success, 1 on error."""
    vector_index = load_index_from_chroma(CHROMA_PERSIST_DIR)
    bm25_nodes = load_bm25_nodes(INDEX_NODES_PATH)

    if vector_index is None or bm25_nodes is None or len(bm25_nodes) == 0:
        print("No index found. Running ingestion first…", file=sys.stderr)
        try:
            result = run_ingestion()
            vector_index = result.index
            bm25_nodes = result.nodes
        except NoDocumentsError as e:
            print(
                f"Ingestion failed: {e}. Add PDFs to data/ or data/arxiv/ "
                "(e.g. run `make run-scraper`), then run this again.",
                file=sys.stderr,
            )
            return 1
        except Exception as e:
            print(f"Ingestion failed: {e}", file=sys.stderr)
            return 1

    path = run_retrieval_experiment(
        vector_index=vector_index,
        bm25_nodes=bm25_nodes,
    )
    print(f"Wrote retrieval evaluation to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
