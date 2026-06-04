"""
embedder.py — Sentence Transformer embedding + FAISS index build/load.

Embeds all item text_descriptions using 'all-MiniLM-L6-v2' and builds a
FAISS IndexFlatL2 vector store. Index and metadata are saved to disk.

Run standalone to (re)build the index:
    python src/embedder.py [--csv data/wardrobe.csv]
                           [--index embeddings/faiss_index.bin]
                           [--meta  embeddings/items.pkl]
"""

import argparse
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CSV   = "data/wardrobe.csv"
DEFAULT_INDEX = "embeddings/faiss_index.bin"
DEFAULT_META  = "embeddings/items.pkl"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(
    df: pd.DataFrame,
    index_path: str = DEFAULT_INDEX,
    meta_path: str  = DEFAULT_META,
    model_name: str = MODEL_NAME,
    batch_size: int = 64,
) -> tuple[faiss.Index, list[dict]]:
    """
    Embed item descriptions and build a FAISS IndexFlatL2 index.

    Parameters
    ----------
    df : pd.DataFrame
        Wardrobe DataFrame with a 'text_description' column.
    index_path : str
        Path to write the serialized FAISS index.
    meta_path : str
        Path to write the pickled item metadata list.
    model_name : str
        Sentence-Transformer model identifier.
    batch_size : int
        Encoding batch size.

    Returns
    -------
    (faiss.Index, list[dict])
        The in-memory index and the ordered list of item metadata dicts.
    """
    print(f"Loading sentence transformer: {model_name} ...")
    model = SentenceTransformer(model_name)

    descriptions = df["text_description"].tolist()
    print(f"Embedding {len(descriptions)} items (batch_size={batch_size}) ...")
    t0 = time.perf_counter()
    embeddings = model.encode(
        descriptions,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity via L2 on unit vectors
        convert_to_numpy=True,
    ).astype("float32")
    elapsed = time.perf_counter() - t0
    print(f"Embedding done in {elapsed:.1f}s  |  shape: {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"FAISS index built  |  ntotal={index.ntotal}  dim={dim}")

    # Persist
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"Index saved → {index_path}")

    metadata = df.to_dict(orient="records")
    # Attach the embedding vector to each metadata record for optional reranking
    for i, rec in enumerate(metadata):
        rec["_embedding"] = embeddings[i]

    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved → {meta_path}")

    return index, metadata


def load_index(
    index_path: str = DEFAULT_INDEX,
    meta_path: str  = DEFAULT_META,
) -> tuple[faiss.Index, list[dict]]:
    """
    Load a previously built FAISS index and metadata from disk.

    Parameters
    ----------
    index_path : str
        Path to the serialized FAISS index file.
    meta_path : str
        Path to the pickled metadata file.

    Returns
    -------
    (faiss.Index, list[dict])
        The in-memory index and item metadata list.

    Raises
    ------
    FileNotFoundError
        If either file does not exist.
    """
    if not Path(index_path).exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not Path(meta_path).exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    print(f"Loaded FAISS index  |  ntotal={index.ntotal}  dim={index.d}")
    return index, metadata


def get_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Return a cached SentenceTransformer instance."""
    return SentenceTransformer(model_name)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from wardrobe CSV")
    parser.add_argument("--csv",   default=DEFAULT_CSV)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--meta",  default=DEFAULT_META)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    from data_gen import generate_wardrobe, save_wardrobe  # noqa: PLC0415

    if not Path(args.csv).exists():
        print(f"{args.csv} not found — generating synthetic data first ...")
        df = generate_wardrobe()
        save_wardrobe(df, args.csv)
    else:
        df = pd.read_csv(args.csv)

    build_index(df, index_path=args.index, meta_path=args.meta, model_name=args.model)
