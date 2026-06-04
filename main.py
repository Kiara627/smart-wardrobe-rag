"""
main.py — CLI end-to-end pipeline runner for the Smart Wardrobe RAG Assistant.

Usage:
    # Full pipeline (generate data → build index → run 3 sample queries → evaluate)
    python main.py

    # Skip data/index rebuild if already done
    python main.py --skip-build

    # Custom query
    python main.py --query "What should I wear to a job interview on a 60°F day?"
"""

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_gen import generate_wardrobe, save_wardrobe
from embedder import DEFAULT_INDEX, DEFAULT_META, build_index, get_model, load_index
from evaluator import run_evaluation
from generator import generate_recommendation
from retriever import retrieve

# ---------------------------------------------------------------------------
# Default sample queries for the demo run
# ---------------------------------------------------------------------------

SAMPLE_QUERIES = [
    {
        "query":     "What should I wear for a rainy 55°F day commuting to work?",
        "condition": "rainy",
        "temp_f":    55.0,
    },
    {
        "query":     "I need a casual outfit for a hot 88°F sunny day at the beach.",
        "condition": "hot",
        "temp_f":    88.0,
    },
    {
        "query":     "What's a good outfit for a formal dinner in cold 38°F weather?",
        "condition": "formal",
        "temp_f":    38.0,
    },
]


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def build_pipeline(csv_path: str, index_path: str, meta_path: str):
    """Generate data and build the FAISS index from scratch."""
    print("\n[1/2] Generating synthetic wardrobe dataset …")
    df = generate_wardrobe(n=500, seed=42)
    save_wardrobe(df, csv_path)

    print("\n[2/2] Building FAISS index …")
    index, metadata = build_index(df, index_path=index_path, meta_path=meta_path)
    return index, metadata


def load_pipeline(index_path: str, meta_path: str):
    """Load existing FAISS index and metadata."""
    return load_index(index_path, meta_path)


def run_queries(
    queries: list[dict],
    index,
    metadata: list[dict],
    model,
) -> None:
    """Run a list of sample queries and print results."""
    for i, q in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"  QUERY {i}: {q['query']}")
        print(f"{'='*70}")

        items, debug = retrieve(
            q["query"],
            index,
            metadata,
            model=model,
            condition_override=q.get("condition"),
            temp_override=q.get("temp_f"),
        )

        print(f"\n  Retrieval debug: {debug}")
        print(f"\n  Top {len(items)} retrieved items:")
        for j, item in enumerate(items, 1):
            print(
                f"    {j}. [{item['category']:10s}] {item['name']:<35s} "
                f"warmth={item['warmth_level']}  tags={item['weather_tags']}"
            )

        print("\n  Generating recommendation …")
        result = generate_recommendation(q["query"], items)
        print(f"  Backend: {result['backend']}")
        print("\n  --- Recommendation ---")
        print(textwrap.indent(textwrap.fill(result["recommendation"], width=66), "  "))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smart Wardrobe RAG pipeline runner")
    parser.add_argument("--skip-build",  action="store_true",
                        help="Load existing index instead of rebuilding")
    parser.add_argument("--query",       default=None,
                        help="Run a single custom query instead of the 3 sample queries")
    parser.add_argument("--csv",         default="data/wardrobe.csv")
    parser.add_argument("--index",       default=DEFAULT_INDEX)
    parser.add_argument("--meta",        default=DEFAULT_META)
    parser.add_argument("--no-eval",     action="store_true",
                        help="Skip the evaluation step")
    args = parser.parse_args()

    # Step 1: Data + index
    index_exists = Path(args.index).exists() and Path(args.meta).exists()
    if args.skip_build and index_exists:
        print("\n[PIPELINE] Loading existing index …")
        index, metadata = load_pipeline(args.index, args.meta)
    else:
        if args.skip_build and not index_exists:
            print("[WARNING] --skip-build set but index not found; rebuilding.")
        print("\n[PIPELINE] Building pipeline from scratch …")
        index, metadata = build_pipeline(args.csv, args.index, args.meta)

    # Step 2: Load embedding model
    print("\n[PIPELINE] Loading sentence transformer …")
    model = get_model()

    # Step 3: Run queries
    if args.query:
        queries = [{"query": args.query, "condition": None, "temp_f": None}]
    else:
        queries = SAMPLE_QUERIES

    print(f"\n[PIPELINE] Running {len(queries)} sample queries …")
    run_queries(queries, index, metadata, model)

    # Step 4: Evaluation
    if not args.no_eval:
        print("\n[PIPELINE] Running full evaluation …")
        metrics = run_evaluation(index, metadata, model)

        # Resume-ready summary block
        print("\n" + "="*65)
        print("  RESUME METRICS (copy these)")
        print("="*65)
        print(f"  Dataset size             : {metrics['dataset_size']} items indexed")
        print(f"  FAISS index dimension    : {metrics['faiss_dim']}-D")
        print(f"  Avg retrieval latency    : {metrics['avg_latency_ms']:.1f} ms")
        print(f"  Precision@5 (avg)        : {metrics['precision_at_5']:.3f}")
        print(f"  Category diversity score : {metrics['diversity_score']:.3f}")
        print(f"  Recency exclusion rate   : {metrics['recency_exclusion']:.3f}")
        print("="*65 + "\n")


if __name__ == "__main__":
    main()
