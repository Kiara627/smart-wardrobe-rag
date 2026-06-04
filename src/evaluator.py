"""
evaluator.py — Evaluation metrics for the Smart Wardrobe RAG pipeline.

Metrics:
    Precision@5       — fraction of top-5 retrieved items that are weather-appropriate
    Diversity score   — category coverage across the top-5 results (0.0–1.0)
    Recency penalty   — % of recently-worn items (last 7 days) excluded from top-5

Run standalone to evaluate all 10 test queries and print the report:
    python src/evaluator.py
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from retriever import (
    RECENCY_DAYS,
    TOP_K_RETURN,
    _is_weather_appropriate,
    retrieve,
)


# ---------------------------------------------------------------------------
# Test query set (10 queries covering diverse conditions)
# ---------------------------------------------------------------------------

TEST_QUERIES: list[dict[str, Any]] = [
    {
        "query":     "What should I wear for a rainy 50°F day?",
        "condition": "rainy",
        "temp_f":    50.0,
        "label":     "rainy_cold",
    },
    {
        "query":     "I need an outfit for a hot 90°F sunny beach day",
        "condition": "hot",
        "temp_f":    90.0,
        "label":     "hot_sunny",
    },
    {
        "query":     "Outfit for a cold 30°F snowy morning commute",
        "condition": "cold",
        "temp_f":    30.0,
        "label":     "cold_snowy",
    },
    {
        "query":     "Casual outfit for a mild 68°F cloudy weekend",
        "condition": "mild",
        "temp_f":    68.0,
        "label":     "mild_casual",
    },
    {
        "query":     "What to wear to a formal dinner tonight?",
        "condition": "formal",
        "temp_f":    None,
        "label":     "formal",
    },
    {
        "query":     "Sunny 75°F day at the park — what looks good?",
        "condition": "sunny",
        "temp_f":    75.0,
        "label":     "sunny_warm",
    },
    {
        "query":     "Rainy 60°F commute to the office",
        "condition": "rainy",
        "temp_f":    60.0,
        "label":     "rainy_mild",
    },
    {
        "query":     "Cold 40°F windy day, something warm but stylish",
        "condition": "cold",
        "temp_f":    40.0,
        "label":     "cold_windy",
    },
    {
        "query":     "Casual hot day 85°F running errands",
        "condition": "hot",
        "temp_f":    85.0,
        "label":     "hot_casual",
    },
    {
        "query":     "Mild 65°F evening dinner with friends",
        "condition": "mild",
        "temp_f":    65.0,
        "label":     "mild_evening",
    },
]

ALL_CATEGORIES = {"top", "bottom", "outerwear", "shoes", "accessory"}


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def precision_at_k(
    items: list[dict[str, Any]],
    condition: str | None,
    temp_f: float | None,
    k: int = TOP_K_RETURN,
) -> float:
    """
    Compute Precision@k: fraction of top-k items that are weather-appropriate.

    Parameters
    ----------
    items : list[dict]
        Retrieved items (already filtered, length ≤ k).
    condition : str | None
        Weather condition label used for relevance judgement.
    temp_f : float | None
        Temperature in °F used for relevance judgement.
    k : int
        Cutoff rank.

    Returns
    -------
    float
        Precision@k in [0.0, 1.0].
    """
    top_k = items[:k]
    if not top_k:
        return 0.0
    relevant = sum(
        1 for item in top_k
        if _is_weather_appropriate(item, condition, temp_f)
    )
    return relevant / len(top_k)


def diversity_score(items: list[dict[str, Any]]) -> float:
    """
    Compute category diversity: unique categories present / total categories.

    Parameters
    ----------
    items : list[dict]
        Retrieved items.

    Returns
    -------
    float
        Diversity in [0.0, 1.0].
    """
    if not items:
        return 0.0
    cats = {item.get("category", "") for item in items}
    return len(cats) / len(ALL_CATEGORIES)


def recency_exclusion_rate(
    all_candidates: list[dict[str, Any]],
    returned_items: list[dict[str, Any]],
) -> float:
    """
    Fraction of recently-worn items in the candidate pool that were excluded
    from the final top-k results.

    Parameters
    ----------
    all_candidates : list[dict]
        All items in the metadata pool.
    returned_items : list[dict]
        Items actually returned by the retriever.

    Returns
    -------
    float
        Exclusion rate in [0.0, 1.0]. Higher = recency penalty working well.
    """
    cutoff = datetime.today() - timedelta(days=RECENCY_DAYS)
    recently_worn_ids: set[str] = set()
    for item in all_candidates:
        last_worn = str(item.get("last_worn_date", "")).strip()
        if last_worn:
            try:
                if datetime.strptime(last_worn, "%Y-%m-%d") >= cutoff:
                    recently_worn_ids.add(item["item_id"])
            except ValueError:
                pass

    if not recently_worn_ids:
        return 1.0   # no recently-worn items → penalty trivially "100% effective"

    returned_ids = {item.get("item_id") for item in returned_items}
    excluded_recent = recently_worn_ids - returned_ids
    return len(excluded_recent) / len(recently_worn_ids)


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    index,
    metadata: list[dict[str, Any]],
    model,
    queries: list[dict[str, Any]] = TEST_QUERIES,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run all test queries and aggregate evaluation metrics.

    Parameters
    ----------
    index : faiss.Index
        Pre-built FAISS index.
    metadata : list[dict]
        Item metadata aligned with the FAISS index.
    model : SentenceTransformer
        Embedding model.
    queries : list[dict]
        Test query dicts (see TEST_QUERIES format).
    verbose : bool
        Print per-query results.

    Returns
    -------
    dict
        Aggregated metrics:
            precision_at_5   : float
            diversity_score  : float
            recency_exclusion: float
            avg_latency_ms   : float
            dataset_size     : int
            faiss_dim        : int
    """
    precisions, diversities, latencies = [], [], []
    all_returned: list[dict[str, Any]] = []

    if verbose:
        print(f"\n{'='*65}")
        print(f"{'SMART WARDROBE RAG — EVALUATION REPORT':^65}")
        print(f"{'='*65}")
        print(f"  Dataset size : {len(metadata)} items")
        print(f"  FAISS dim    : {index.d}")
        print(f"  Test queries : {len(queries)}")
        print(f"{'='*65}\n")

    for q in queries:
        items, debug = retrieve(
            q["query"],
            index,
            metadata,
            model=model,
            condition_override=q.get("condition"),
            temp_override=q.get("temp_f"),
        )

        p5  = precision_at_k(items, q.get("condition"), q.get("temp_f"))
        div = diversity_score(items)
        lat = debug["latency_ms"]

        precisions.append(p5)
        diversities.append(div)
        latencies.append(lat)
        all_returned.extend(items)

        if verbose:
            print(f"  [{q['label']:<16s}] P@5={p5:.2f}  div={div:.2f}  lat={lat:6.1f}ms")
            print(f"    Query: {q['query']}")
            for item in items:
                print(
                    f"      • {item['name']:<35s} [{item['category']:10s}] "
                    f"warmth={item['warmth_level']}  tags={item['weather_tags']}"
                )
            print()

    recency_rate = recency_exclusion_rate(metadata, all_returned)

    avg_p5  = float(np.mean(precisions))
    avg_div = float(np.mean(diversities))
    avg_lat = float(np.mean(latencies))

    if verbose:
        print(f"{'='*65}")
        print(f"{'SUMMARY':^65}")
        print(f"{'='*65}")
        print(f"  Dataset size             : {len(metadata):>8}")
        print(f"  FAISS index dimension    : {index.d:>8}")
        print(f"  Avg retrieval latency    : {avg_lat:>7.1f} ms")
        print(f"  Precision@5 (avg)        : {avg_p5:>8.3f}")
        print(f"  Category diversity (avg) : {avg_div:>8.3f}")
        print(f"  Recency exclusion rate   : {recency_rate:>8.3f}")
        print(f"{'='*65}\n")

    return {
        "precision_at_5":    avg_p5,
        "diversity_score":   avg_div,
        "recency_exclusion": recency_rate,
        "avg_latency_ms":    avg_lat,
        "dataset_size":      len(metadata),
        "faiss_dim":         index.d,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from embedder import load_index, get_model

    idx, meta = load_index()
    model = get_model()
    run_evaluation(idx, meta, model)
