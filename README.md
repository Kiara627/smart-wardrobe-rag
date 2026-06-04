# Smart Wardrobe RAG Assistant

A retrieval-augmented outfit recommendation system that answers natural language
queries like *"What should I wear for a rainy 55°F day?"* by searching a personal
wardrobe vector store and generating a styled recommendation via an LLM.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  RETRIEVAL LAYER                                                │
│                                                                 │
│  Query ──► SentenceTransformer ──► Query Embedding (384-D)     │
│            (all-MiniLM-L6-v2)                                  │
│                      │                                         │
│                      ▼                                         │
│            FAISS IndexFlatL2 ──► top-10 candidates            │
│            (500 wardrobe items)                                │
│                      │                                         │
│                      ▼                                         │
│            Post-retrieval Filters                               │
│            ├── Weather appropriateness (temp + condition)      │
│            ├── Recency penalty (deprioritize last 7 days)      │
│            └── Category balance (top/bottom/outerwear/shoes)   │
│                      │                                         │
│                      ▼                                         │
│            Top-5 filtered items + metadata                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  GENERATION LAYER                                               │
│                                                                 │
│  System prompt + User query + Retrieved context                │
│                      │                                         │
│            ┌─────────┴──────────┐                              │
│            ▼                    ▼                              │
│     OpenAI GPT-3.5        flan-t5-base                         │
│     (if API key set)      (local fallback)                     │
│            └─────────┬──────────┘                              │
│                      ▼                                         │
│            Natural language outfit recommendation              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
              Streamlit UI / CLI output
```

---

## File Structure

```
smart-wardrobe-rag/
├── data/
│   └── wardrobe.csv          # 500-item synthetic wardrobe (auto-generated)
├── embeddings/
│   ├── faiss_index.bin        # FAISS index (gitignored, built at runtime)
│   └── items.pkl              # item metadata (gitignored, built at runtime)
├── src/
│   ├── data_gen.py            # synthetic wardrobe dataset generator
│   ├── embedder.py            # sentence transformer embedding + FAISS build/load
│   ├── retriever.py           # query → FAISS search → post-filter → top-5
│   ├── generator.py           # LLM prompt + OpenAI / flan-t5 fallback
│   └── evaluator.py           # precision@5, diversity score, recency exclusion
├── app.py                     # Streamlit UI
├── main.py                    # CLI end-to-end pipeline runner
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Kiara627/smart-wardrobe-rag
cd smart-wardrobe-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key (optional — flan-t5 works without it)
```

`.env` format:
```
OPENAI_API_KEY=sk-...
```

### 3. Run the CLI pipeline

```bash
# Full build + 3 sample queries + evaluation report
python main.py

# Skip rebuild if index already exists
python main.py --skip-build

# Custom single query
python main.py --skip-build --query "What to wear for a hot 85°F sunny day?"
```

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

---

## Example Queries and Output Format

### Query 1 — Rainy cold commute
```
Input : "What should I wear for a rainy 55°F day commuting to work?"

Retrieved items:
  1. [outerwear ] Navy Nylon Rain Jacket          warmth=4  tags=cold,rainy,cloudy
  2. [top       ] Charcoal Wool Blend Turtleneck  warmth=4  tags=cold,rainy,cloudy
  3. [bottom    ] Navy Cotton Twill Slim Chinos   warmth=3  tags=mild,cloudy
  4. [shoes     ] Black Leather Chelsea Boots     warmth=4  tags=cold,rainy,cloudy
  5. [accessory ] Charcoal Wool Scarf             warmth=4  tags=cold,rainy,cloudy

Recommendation:
  "For a rainy 55°F commute, start with the Charcoal Turtleneck as your base
   layer — the wool blend keeps you warm without bulk. Layer the Navy Rain
   Jacket over it; the nylon shell will keep you dry through the commute.
   The Slim Chinos in navy are weather-resistant enough for light rain while
   staying office-appropriate. Finish with the Chelsea Boots (waterproof
   leather) and the Wool Scarf for neck warmth. This outfit transitions
   seamlessly from commute to desk."
```

### Query 2 — Hot sunny beach day
```
Input : "I need a casual outfit for a hot 88°F sunny day at the beach."

Retrieved items:
  1. [top       ] White Cotton Tank Top           warmth=1  tags=hot,sunny
  2. [bottom    ] Cobalt Blue Denim Shorts        warmth=1  tags=hot,sunny
  3. [shoes     ] Beige Canvas Slip-on Sneakers   warmth=2  tags=sunny,mild
  4. [accessory ] Beige Straw Hat                 warmth=2  tags=sunny,mild
  5. [top       ] Olive Linen Sleeveless Top      warmth=1  tags=hot,sunny
```

### Query 3 — Formal dinner, cold weather
```
Input : "What's a good outfit for a formal dinner in cold 38°F weather?"

Retrieved items:
  1. [outerwear ] Camel Wool Overcoat             warmth=5  tags=cold,rainy,snowy
  2. [top       ] Cream Silk Blouse               warmth=2  tags=sunny,mild
  3. [bottom    ] Black Wool Blend Wide-leg Trousers warmth=4  tags=cold,cloudy
  4. [shoes     ] Black Leather Oxford Shoes      warmth=3  tags=mild,cloudy
  5. [accessory ] Black Leather Belt              warmth=3  tags=mild,cloudy
```

---

## Running Evaluation

```bash
# Standalone evaluation on 10 test queries
python src/evaluator.py

# Or via main.py (included by default)
python main.py --skip-build
```

Output format:
```
Dataset size             :      500
FAISS index dimension    :      384
Avg retrieval latency    :   462.4 ms
Precision@5 (avg)        :    0.900
Category diversity score :    0.440
Recency exclusion rate   :    0.974
```
