"""Chroma-backed vector store for QA run history.

Persists story + test plan + locators on successful runs and enables the
Analyst to surface similar prior results before generating a fresh plan.

Graceful degradation: every public function catches ImportError (chromadb not
installed) and any DB-level exception, logs the problem, and returns a safe
default so the run always continues normally without memory.

Persistence directory: backend/data/chroma (created on demand).

Embedding strategy
──────────────────
Production (embedding_function=None): documents/query texts are passed to
chromadb which calls its built-in DefaultEmbeddingFunction
(all-MiniLM-L6-v2 via onnxruntime — local, offline after first download).

Tests (embedding_function=<callable>): embeddings are computed externally by
the caller before being handed to chromadb as pre-computed vectors. This
bypasses chromadb's internal EF registry so tests run without any model
download and without hitting chromadb's 1.5+ "name required" protocol.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_DEFAULT_PERSIST_DIR = Path(__file__).parent.parent.parent / "data" / "chroma"
_COLLECTION_NAME = "qa_runs"
# Cosine distance threshold: return results with distance < this value.
# distance = 1 − cosine_similarity → 0.4 ≈ similarity > 0.6.
_DISTANCE_THRESHOLD = 0.4

# One PersistentClient per directory path, shared across calls.
# Creating multiple clients pointing at the same path on the same process
# causes HNSW background threads to hold stale file handles on Windows,
# making the second client query an incomplete index.
_client_cache: dict[str, Any] = {}


def _get_collection(persist_dir: Path | None = None):
    """Return (or create) the Chroma collection.

    No embedding function is attached to the collection object — callers
    either provide pre-computed embeddings (tests) or pass raw documents so
    chromadb invokes its built-in DefaultEmbeddingFunction (production).

    Raises ImportError when chromadb is not installed.
    All other exceptions propagate to callers which log and return safe defaults.
    """
    import chromadb  # late import — fails clearly when not installed

    db_path = persist_dir or _DEFAULT_PERSIST_DIR
    db_path.mkdir(parents=True, exist_ok=True)

    key = str(db_path)
    if key not in _client_cache:
        _client_cache[key] = chromadb.PersistentClient(path=key)
    client = _client_cache[key]

    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def persist_run(
    run_id: str,
    story: str,
    test_plan: list[dict],
    locators: dict,
    verdict: str,
    target_url: str = "",
    *,
    persist_dir: Path | None = None,
    embedding_function: Callable | None = None,
) -> bool:
    """Embed and persist a completed run to the vector store.

    The document is the story concatenated with the serialised test plan so
    similarity queries match on both intent and step structure.

    Args:
        embedding_function: When provided, called with ``[document]`` to
            produce pre-computed embeddings passed directly to chromadb.
            When None, chromadb embeds the document with its built-in model.

    Returns True on success, False on any failure (always safe to ignore).
    """
    if not story.strip():
        return False

    try:
        collection = _get_collection(persist_dir)
        # Embed only the story so retrieval queries (which are also just stories)
        # land in the same semantic space.  Test plan and locators go in metadata
        # for downstream use, not in the embedded text.
        metadata = {
            "run_id":         run_id,
            "verdict":        verdict,
            "target_url":     target_url,
            "steps":          len(test_plan),
            "test_plan_json": json.dumps(test_plan),
            "locators_json":  json.dumps(locators),
        }

        if embedding_function is not None:
            # Pre-compute so chromadb never needs to call an EF internally.
            embeddings = embedding_function([story])
            collection.upsert(
                ids=[run_id],
                embeddings=embeddings,
                documents=[story],
                metadatas=[metadata],
            )
        else:
            # Production path: let chromadb's built-in local model embed.
            collection.upsert(
                ids=[run_id],
                documents=[story],
                metadatas=[metadata],
            )

        logger.info(
            "VectorStore: persisted run %s (verdict=%s, steps=%d)",
            run_id, verdict, len(test_plan),
        )
        return True

    except ImportError:
        logger.warning("VectorStore: chromadb not installed — skipping persistence")
        return False
    except Exception as exc:
        logger.error("VectorStore: persist failed for run %s — %s", run_id, exc)
        return False


def retrieve_similar(
    story: str,
    n_results: int = 1,
    threshold: float = _DISTANCE_THRESHOLD,
    *,
    persist_dir: Path | None = None,
    embedding_function: Callable | None = None,
) -> list[dict]:
    """Return up to *n_results* past runs similar to *story*.

    Each result dict contains: run_id, verdict, target_url, steps, distance.
    Returns [] on any failure or when no results fall within *threshold*.

    Args:
        embedding_function: When provided, query is embedded externally and
            passed as ``query_embeddings``.  When None, chromadb uses its
            built-in model via ``query_texts``.
    """
    if not story.strip():
        return []

    try:
        collection = _get_collection(persist_dir)

        count = collection.count()
        if count == 0:
            return []

        actual_n = min(n_results, count)

        if embedding_function is not None:
            query_embeddings = embedding_function([story])
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=actual_n,
                include=["metadatas", "distances"],
            )
        else:
            results = collection.query(
                query_texts=[story],
                n_results=actual_n,
                include=["metadatas", "distances"],
            )

        hits: list[dict] = []
        for i, distance in enumerate(results["distances"][0]):
            if distance >= threshold:
                continue
            meta = results["metadatas"][0][i]
            hits.append({
                "run_id":     meta.get("run_id", ""),
                "verdict":    meta.get("verdict", ""),
                "target_url": meta.get("target_url", ""),
                "steps":      meta.get("steps", 0),
                "distance":   distance,
            })
        return hits

    except ImportError:
        logger.warning("VectorStore: chromadb not installed — skipping retrieval")
        return []
    except Exception as exc:
        logger.error("VectorStore: retrieve failed — %s", exc)
        return []
