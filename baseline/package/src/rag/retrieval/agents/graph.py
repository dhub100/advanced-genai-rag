import json
import pathlib
import pickle

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# ---------- Adjust these paths if your folder layout changes ----------
ROOT = pathlib.Path(r"{STORAGE}")
EMB_DIR = ROOT / "embeddings"
CHUNK_PKL = pathlib.Path(r"{DATA_DIR}/fixed_size_chunk/docs_fixed_norm.pkl")
LOADER_PATH = ROOT / "load_graphrag.py"
# ----------------------------------------------------------------------

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Lazy-loaded globals
_embedder: SentenceTransformer = SentenceTransformer(_MODEL_NAME)
_emb_cache: dict[int, np.ndarray] = {}
_cid_cache: dict[int, list] = {}
_chunk_by_id: dict | None = None
_chunk_vec_cache: dict = {}


def _load_embeddings(level: int) -> tuple[np.ndarray, list]:
    """Load and cache the embedding matrix and chunk-ID list for a given level."""
    if level not in _emb_cache:
        _emb_cache[level] = np.load(EMB_DIR / f"EMB_fixed_C{level}.npy")
        _cid_cache[level] = json.loads(
            (EMB_DIR / f"CID_fixed_C{level}.json").read_text()
        )
    return _emb_cache[level], _cid_cache[level]


def _restore_chunk(doc: Document) -> Document:
    """Return a Document whose page_content is the original (un-normalised) text."""
    raw = doc.metadata.get("original_text") or doc.page_content
    return Document(page_content=raw, metadata=doc.metadata)


def _load_chunks() -> dict[str, Document]:
    """Load, cache, and return a mapping of chunk_id → Document."""
    global _chunk_by_id
    if _chunk_by_id is None:
        with open(CHUNK_PKL, "rb") as f:
            docs_norm = pickle.load(f)
        docs = [_restore_chunk(d) for d in docs_norm]
        _chunk_by_id = {d.metadata["chunk_id"]: d for d in docs}
    return _chunk_by_id


def _chunk_vector(cid: str, chunks: dict[str, Document]) -> np.ndarray:
    """Return (and cache) the normalised embedding vector for a chunk."""
    if cid not in _chunk_vec_cache:
        _chunk_vec_cache[cid] = _embedder.encode(
            [chunks[cid].page_content], normalize_embeddings=True
        )[0]
    return _chunk_vec_cache[cid]


class GraphAgent:

    def retrieve(
        self,
        query: str,
        *,
        level: str = "C1",
        k_comms: int = 24,
        top_k: int = 100,
    ) -> list[Document]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query:   The search query string.
            level:   Community level to search (e.g. "C1", "C2").
            k_comms: Number of top communities to expand.
            top_k:   Maximum number of chunks to return.

        Returns:
            A list of Documents with `metadata["grag_score"]` in [0, 1].
        """
        level_int = int(level.lstrip("C"))
        emb_mat, cid_list = _load_embeddings(level_int)
        chunks = _load_chunks()
        comm2chunk: dict = json.loads((ROOT / "comm2chunk_fixed.json").read_text())

        q_vec = _embedder.encode([query], normalize_embeddings=True)[0]

        # Find the top-k_comms communities closest to the query
        top_comm_indices = (emb_mat @ q_vec).argsort()[::-1][:k_comms]

        # Expand communities to candidate chunk IDs
        candidate_ids = {
            cid for idx in top_comm_indices for cid in comm2chunk.get(cid_list[idx], [])
        }
        if not candidate_ids:
            return []

        # Score and rank candidate chunks
        scored = sorted(
            ((cid, float(_chunk_vector(cid, chunks) @ q_vec)) for cid in candidate_ids),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = []
        for cid, sim in scored:
            doc = chunks[cid]
            doc.metadata["grag_score"] = (sim + 1) / 2
            results.append(doc)
        return results
