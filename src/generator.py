"""
generator.py — LLM outfit recommendation generator.

Tries OpenAI GPT-3.5-turbo first (requires OPENAI_API_KEY in .env).
Falls back to google/flan-t5-base via HuggingFace Transformers if no key
is present or if the OpenAI call fails.

Run standalone:
    python src/generator.py "What should I wear for a rainy 55°F day?"
"""

import os
import sys
import textwrap
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a personal stylist assistant. "
    "Given the user's weather context and a set of retrieved wardrobe items, "
    "recommend a complete, cohesive outfit. Be specific, practical, and explain "
    "why each piece works together and for the conditions described."
)

def _format_items(items: list[dict[str, Any]]) -> str:
    """Format retrieved item metadata into a readable context block."""
    lines = []
    for i, item in enumerate(items, 1):
        lines.append(
            f"{i}. {item['name']} "
            f"[{item['category']}] — "
            f"{item['fabric']}, warmth level {item['warmth_level']}/5, "
            f"suitable for: {item['weather_tags'].replace(',', ', ')}"
        )
    return "\n".join(lines)


def build_prompt(query: str, items: list[dict[str, Any]]) -> str:
    """
    Construct the full user prompt combining the query and retrieved items.

    Parameters
    ----------
    query : str
        Original user query.
    items : list[dict]
        Retrieved wardrobe items (from retriever.retrieve).

    Returns
    -------
    str
        Formatted prompt string ready for the LLM.
    """
    context = _format_items(items)
    return (
        f"User request: {query}\n\n"
        f"Available wardrobe items:\n{context}\n\n"
        "Please recommend a complete outfit from these items. "
        "For each piece you choose, briefly explain why it fits the occasion and conditions."
    )


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

def _generate_openai(query: str, items: list[dict[str, Any]], model: str = "gpt-3.5-turbo") -> str:
    """
    Generate an outfit recommendation via the OpenAI Chat Completions API.

    Parameters
    ----------
    query : str
        Original user query.
    items : list[dict]
        Retrieved wardrobe items.
    model : str
        OpenAI model identifier.

    Returns
    -------
    str
        Generated recommendation text.

    Raises
    ------
    ImportError
        If the openai package is not installed.
    Exception
        Propagates API errors for the caller to handle.
    """
    try:
        from openai import OpenAI  # openai >= 1.0
    except ImportError as exc:
        raise ImportError("openai package not installed. Run: pip install openai") from exc

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    user_prompt = build_prompt(query, items)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Flan-T5 fallback backend
# ---------------------------------------------------------------------------

_flan_pipeline = None   # module-level cache to avoid reloading the model


def _get_flan_pipeline():
    """Lazy-load flan-t5-base pipeline (cached after first call)."""
    global _flan_pipeline
    if _flan_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers package not installed. Run: pip install transformers"
            ) from exc

        print("Loading google/flan-t5-base (this may take a moment on first run) ...")
        _flan_pipeline = hf_pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=256,
            do_sample=False,
        )
        print("flan-t5-base loaded.")
    return _flan_pipeline


def _generate_flan(query: str, items: list[dict[str, Any]]) -> str:
    """
    Generate an outfit recommendation using flan-t5-base locally.

    The model has a 512-token input limit so the prompt is truncated if needed.

    Parameters
    ----------
    query : str
        Original user query.
    items : list[dict]
        Retrieved wardrobe items.

    Returns
    -------
    str
        Generated recommendation text (shorter than OpenAI output).
    """
    pipe = _get_flan_pipeline()

    # Flan-T5 works best with an explicit instruction prefix
    context = _format_items(items)
    prompt = (
        f"You are a personal stylist. "
        f"The user says: '{query}'. "
        f"Recommend an outfit from these items:\n{context}\n"
        f"Outfit recommendation:"
    )
    # Truncate to ~450 tokens worth of characters (rough heuristic)
    prompt = prompt[:1800]

    result = pipe(prompt)
    return result[0]["generated_text"].strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_recommendation(
    query: str,
    items: list[dict[str, Any]],
    force_fallback: bool = False,
) -> dict[str, Any]:
    """
    Generate a natural-language outfit recommendation.

    Tries OpenAI if OPENAI_API_KEY is set; otherwise falls back to flan-t5-base.

    Parameters
    ----------
    query : str
        Original user query.
    items : list[dict]
        Retrieved wardrobe items from retriever.retrieve.
    force_fallback : bool
        Skip OpenAI and use flan-t5-base directly (for testing).

    Returns
    -------
    dict with keys:
        recommendation : str   — generated text
        backend        : str   — "openai" | "flan-t5"
        item_ids       : list  — item_ids of the retrieved items passed as context
    """
    item_ids = [item.get("item_id", "") for item in items]
    backend = "unknown"
    recommendation = ""

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    use_openai = bool(api_key) and not force_fallback

    if use_openai:
        try:
            recommendation = _generate_openai(query, items)
            backend = "openai"
        except Exception as exc:
            print(f"[generator] OpenAI call failed ({exc}); falling back to flan-t5-base.")
            use_openai = False

    if not use_openai:
        recommendation = _generate_flan(query, items)
        backend = "flan-t5"

    return {
        "recommendation": recommendation,
        "backend": backend,
        "item_ids": item_ids,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from embedder import load_index, get_model
    from retriever import retrieve

    query = sys.argv[1] if len(sys.argv) > 1 else "What should I wear for a rainy 55°F day?"

    idx, meta = load_index()
    model = get_model()
    items, debug = retrieve(query, idx, meta, model=model)

    print(f"\nQuery : {query!r}")
    print(f"Debug : {debug}\n")

    result = generate_recommendation(query, items)
    print(f"Backend: {result['backend']}\n")
    print("--- Recommendation ---")
    print(textwrap.fill(result["recommendation"], width=80))
