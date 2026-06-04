"""
app.py — Streamlit UI for the Smart Wardrobe RAG Assistant.

Run:
    streamlit run app.py
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_gen import generate_wardrobe, save_wardrobe
from embedder import DEFAULT_INDEX, DEFAULT_META, build_index, get_model, load_index
from evaluator import run_evaluation
from generator import generate_recommendation
from retriever import retrieve

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Wardrobe Assistant",
    page_icon="👗",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _ensure_index():
    """Load or build FAISS index, cached in session state."""
    if "index" not in st.session_state or "metadata" not in st.session_state:
        try:
            idx, meta = load_index(DEFAULT_INDEX, DEFAULT_META)
        except FileNotFoundError:
            with st.spinner("First run: generating wardrobe data and building FAISS index …"):
                df = generate_wardrobe()
                save_wardrobe(df, "data/wardrobe.csv")
                idx, meta = build_index(df, DEFAULT_INDEX, DEFAULT_META)
        st.session_state["index"] = idx
        st.session_state["metadata"] = meta

    if "model" not in st.session_state:
        with st.spinner("Loading sentence transformer …"):
            st.session_state["model"] = get_model()


def _mark_worn(item_id: str):
    """Mark an item as worn today in metadata and re-persist."""
    meta = st.session_state["metadata"]
    today = date.today().isoformat()
    for item in meta:
        if item["item_id"] == item_id:
            item["last_worn_date"] = today
    # Persist to CSV if it exists
    csv_path = Path("data/wardrobe.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.loc[df["item_id"] == item_id, "last_worn_date"] = today
        df.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("Smart Wardrobe RAG Assistant")
st.caption("Describe your day and get AI-powered outfit recommendations from your wardrobe.")

_ensure_index()

# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Weather Context")
    temp_f = st.slider("Temperature (°F)", min_value=0, max_value=100, value=65, step=1)
    condition = st.selectbox(
        "Condition",
        options=["sunny", "mild", "cloudy", "rainy", "cold", "hot", "formal"],
        index=0,
    )
    st.divider()
    st.caption(f"Wardrobe: **{len(st.session_state['metadata'])}** items indexed")
    st.caption(f"FAISS dim: **{st.session_state['index'].d}**")

# ---- Main panel ------------------------------------------------------------
query = st.text_input(
    "Describe your day / occasion",
    placeholder="e.g. Rainy 55°F commute to the office",
)

run_btn = st.button("Get Outfit Recommendation", type="primary", use_container_width=True)

if run_btn and query.strip():
    idx   = st.session_state["index"]
    meta  = st.session_state["metadata"]
    model = st.session_state["model"]

    with st.spinner("Retrieving items …"):
        items, debug = retrieve(
            query,
            idx,
            meta,
            model=model,
            condition_override=condition,
            temp_override=float(temp_f),
        )

    # ---- Retrieved item cards ----------------------------------------------
    st.subheader("Retrieved Wardrobe Items")
    cols = st.columns(min(len(items), 5))
    worn_today: list[str] = []

    for col, item in zip(cols, items):
        with col:
            warmth_bar = "🔥" * item["warmth_level"] + "·" * (5 - item["warmth_level"])
            st.markdown(
                f"""
**{item['name']}**

`{item['category'].upper()}`

- Fabric: {item['fabric']}
- Color: {item['color']}
- Warmth: {warmth_bar}
- Tags: `{item['weather_tags']}`
"""
            )
            if st.checkbox("Mark as worn today", key=f"worn_{item['item_id']}"):
                worn_today.append(item["item_id"])

    if worn_today:
        for iid in worn_today:
            _mark_worn(iid)
        st.success(f"Marked {len(worn_today)} item(s) as worn today.")

    # ---- LLM recommendation ------------------------------------------------
    st.subheader("AI Outfit Recommendation")
    with st.spinner("Generating recommendation …"):
        result = generate_recommendation(query, items)

    backend_badge = "🟢 OpenAI GPT-3.5" if result["backend"] == "openai" else "🟡 flan-t5-base (local)"
    st.caption(f"Powered by: {backend_badge}  |  Retrieval latency: {debug['latency_ms']:.1f} ms")
    st.markdown(result["recommendation"])

    # ---- Evaluation metrics expander ---------------------------------------
    with st.expander("Evaluation Metrics (this query)"):
        from evaluator import precision_at_k, diversity_score, recency_exclusion_rate

        p5  = precision_at_k(items, condition, float(temp_f))
        div = diversity_score(items)
        rec = recency_exclusion_rate(meta, items)

        m1, m2, m3 = st.columns(3)
        m1.metric("Precision@5",       f"{p5:.2f}")
        m2.metric("Diversity Score",   f"{div:.2f}")
        m3.metric("Recency Exclusion", f"{rec:.2f}")

elif run_btn:
    st.warning("Please enter a query above.")

# ---- Full eval expander ----------------------------------------------------
with st.expander("Run full evaluation on 10 test queries"):
    if st.button("Run Evaluation", key="run_eval"):
        with st.spinner("Evaluating …"):
            metrics = run_evaluation(
                st.session_state["index"],
                st.session_state["metadata"],
                st.session_state["model"],
                verbose=False,
            )
        st.json(metrics)
