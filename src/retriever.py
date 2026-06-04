"""
retriever.py — Query embedding, FAISS search, and post-retrieval filtering.

Pipeline:
  1. Embed user query with the same Sentence Transformer used at index time.
  2. Search FAISS for top-k=10 nearest neighbors (L2 on unit vectors ≈ cosine).
  3. Apply post-retrieval filters:
       • Weather appropriateness  — item weather_tags must match query condition
       • Recency penalty          — deprioritize items worn in last 7 days
       • Category balance         — include ≥1 item per major category where possible
  4. Return top-5 filtered candidates with full metadata.

Run standalone for a quick smoke-test:
    python src/retriever.py "What should I wear for a rainy 55°F day?"
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# Allow running from project root or from src/
sys.path.insert(0, str(Path(__file__).parent))

from embedder import MODEL_NAME, get_model, load_index


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP_K_FAISS   = 10          # candidates retrieved from FAISS before filtering
TOP_K_RETURN  = 5           # items returned to caller
RECENCY_DAYS  = 7           # items worn within this window are deprioritized
RECENCY_SCORE = 0.5         # multiplier applied to L2 score of recent items

# Condition → weather_tags that qualify as a match
CONDITION_TAGS: dict[str, set[str]] = {
    "sunny":  {"sunny", "hot", "mild"},
    "hot":    {"hot", "sunny"},
    "mild":   {"mild", "sunny", "cloudy"},
    "cloudy": {"cloudy", "mild", "cold"},
    "rainy":  {"rainy", "cold", "cloudy"},
    "cold":   {"cold", "rainy", "cloudy", "snowy"},
    "snowy":  {"snowy", "cold"},
    "formal": set(),          # formal is handled by name/fabric heuristic
}

# Temperature (°F) → warmth levels considered appropriate
def _warmth_for_temp(temp_f: float) -> set[int]:
    """Map temperature to appropriate warmth levels."""
    if temp_f >= 80:
        return {1, 2}
    elif temp_f >= 65:
        return {2, 3}
    elif temp_f >= 50:
        return {3, 4}
    elif temp_f >= 35:
        return {4, 5}
    else:
        return {5}

PRIORITY_CATEGORIES = ["top", "bottom", "outerwear", "shoes"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_weather_from_query(query: str) -> tuple[float | None, str | None]:
    """
    Heuristically extract temperature (°F) and condition from a free-text query.

    Returns
    -------
    (temperature_f, condition)
        Either value is None if not detected.
    """
    import re

    temp = None
    condition = None

    # Match patterns like "55°F", "55 F", "55 degrees", "55F"
    m = re.search(r"(\d{2,3})\s*(?:°\s*)?[Ff](?:\b|degrees?)", query)
    if not m:
        m = re.search(r"(\d{2,3})\s*degrees?", query, re.IGNORECASE)
    if m:
        temp = float(m.group(1))

    q_lower = query.lower()
    for cond in ("snowy", "rainy", "cold", "hot", "sunny", "cloudy", "mild", "formal"):
        if cond in q_lower:
            condition = cond
            break

    return temp, condition


def _is_weather_appropriate(item: dict[str, Any], condition: str | None, temp_f: float | None) -> bool:
    """
    Return True if the item is appropriate for the given weather context.

    If neither condition nor temp is provided, all items pass.
    """
    if condition is None and temp_f is None:
        return True

    item_tags = set(str(item.get("weather_tags", "")).split(","))
    warmth = int(item.get("warmth_level", 3))

    # Formal condition: pass blazers, dress shirts, tailored items
    if condition == "formal":
        formal_keywords = {"blazer", "oxford", "trouser", "loafer", "blouse", "silk", "tailored"}
        name_lower = str(item.get("name", "")).lower()
        fabric_lower = str(item.get("fabric", "")).lower()
        return any(kw in name_lower or kw in fabric_lower for kw in formal_keywords)

    # Temperature-based warmth filter
    if temp_f is not None:
        ok_warmths = _warmth_for_temp(temp_f)
        if warmth not in ok_warmths:
            return False

    # Condition tag match
    if condition is not None:
        acceptable = CONDITION_TAGS.get(condition, set())
        if acceptable and item_tags.isdisjoint(acceptable):
            return False

    return True


def _recency_penalty(item: dict[str, Any]) -> float:
    """
    Return a score multiplier (1.0 = no penalty, >1.0 = penalised).

    Items worn in the last RECENCY_DAYS get their effective L2 distance
    inflated so they rank lower.
    """
    last_worn = str(item.get("last_worn_date", "")).strip()
    if not last_worn:
        return 1.0
    try:
        d = datetime.strptime(last_worn, "%Y-%m-%d")
        if (datetime.today() - d).days <= RECENCY_DAYS:
            return 1.0 / RECENCY_SCORE   # inflate distance → lower priority
    except ValueError:
        pass
    return 1.0


def _balance_categories(candidates: list[dict[str, Any]], k: int = TOP_K_RETURN) -> list[dict[str, Any]]:
    """
    Greedily select up to *k* items ensuring category diversity.

    Strategy:
      1. Ensure one item per priority category (top, bottom, outerwear, shoes).
      2. Fill remaining slots from the remaining candidates in score order.
    """
    selected: list[dict[str, Any]] = []
    seen_cats: set[str] = set()
    remainder: list[dict[str, Any]] = []

    # Pass 1: one per priority category
    for cat in PRIORITY_CATEGORIES:
        for item in candidates:
            if item["category"] == cat and cat not in seen_cats:
                selected.append(item)
                seen_cats.add(cat)
                break

    # Pass 2: fill remaining slots
    for item in candidates:
        if item not in selected:
            remainder.append(item)

    combined = selected + remainder
    return combined[:k]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    index,
    metadata: list[dict[str, Any]],
    model=None,
    top_k_faiss: int = TOP_K_FAISS,
    top_k_return: int = TOP_K_RETURN,
    condition_override: str | None = None,
    temp_override: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Retrieve and filter wardrobe items for a natural-language query.

    Parameters
    ----------
    query : str
        User's natural language query.
    index : faiss.Index
        Pre-built FAISS index.
    metadata : list[dict]
        Item metadata aligned with FAISS index rows.
    model : SentenceTransformer | None
        Model to embed the query. Loaded if None.
    top_k_faiss : int
        Number of candidates to fetch from FAISS.
    top_k_return : int
        Number of items to return after filtering.
    condition_override : str | None
        Explicit weather condition (overrides query parsing).
    temp_override : float | None
        Explicit temperature in °F (overrides query parsing).

    Returns
    -------
    (items, debug_info)
        items : list of filtered item dicts (length ≤ top_k_return)
        debug_info : dict with latency, condition, temp, n_faiss_hits
    """
    if model is None:
        model = get_model(MODEL_NAME)

    # Parse weather context
    temp_f, condition = _parse_weather_from_query(query)
    if condition_override is not None:
        condition = condition_override
    if temp_override is not None:
        temp_f = temp_override

    # Embed query
    t0 = time.perf_counter()
    q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, top_k_faiss)
    latency_ms = (time.perf_counter() - t0) * 1000

    raw_candidates: list[tuple[float, dict]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx].copy()
        # Apply recency penalty to L2 distance
        adjusted_dist = dist * _recency_penalty(item)
        raw_candidates.append((adjusted_dist, item))

    # Sort by adjusted distance (ascending = more similar)
    raw_candidates.sort(key=lambda x: x[0])

    # Filter for weather appropriateness
    weather_filtered = [
        item for _, item in raw_candidates
        if _is_weather_appropriate(item, condition, temp_f)
    ]

    # If filter is too aggressive, fall back to raw ranking
    if len(weather_filtered) < 3:
        weather_filtered = [item for _, item in raw_candidates]

    # Balance categories
    results = _balance_categories(weather_filtered, k=top_k_return)

    debug_info = {
        "latency_ms": round(latency_ms, 2),
        "condition": condition,
        "temp_f": temp_f,
        "n_faiss_hits": len(raw_candidates),
        "n_weather_filtered": len(weather_filtered),
    }

    return results, debug_info


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="What should I wear for a rainy 55°F day?")
    parser.add_argument("--index", default="embeddings/faiss_index.bin")
    parser.add_argument("--meta",  default="embeddings/items.pkl")
    args = parser.parse_args()

    faiss_index, meta = load_index(args.index, args.meta)
    model = get_model()

    items, debug = retrieve(args.query, faiss_index, meta, model=model)
    print(f"\nQuery: {args.query!r}")
    print(f"Debug: {debug}")
    print("\nTop results:")
    for i, item in enumerate(items, 1):
        print(
            f"  {i}. [{item['category']:10s}] {item['name']:<35s} "
            f"warmth={item['warmth_level']}  tags={item['weather_tags']}"
        )
