"""
data_gen.py — Synthetic wardrobe dataset generator.

Generates a realistic 500-item wardrobe CSV with the schema:
    item_id, name, category, color, fabric, warmth_level,
    weather_tags, last_worn_date, compatible_with

Run standalone:
    python src/data_gen.py [--output data/wardrobe.csv] [--n 500] [--seed 42]
"""

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Vocabulary tables
# ---------------------------------------------------------------------------

CATEGORIES = {
    "top": [
        "Oxford Shirt", "Linen Shirt", "Flannel Shirt", "Turtleneck", "Crew-neck Sweater",
        "V-neck Sweater", "Cardigan", "Hoodie", "Graphic Tee", "Plain Tee",
        "Tank Top", "Blouse", "Silk Blouse", "Denim Shirt", "Henley",
        "Quarter-zip Pullover", "Polo Shirt", "Sleeveless Top", "Crop Top", "Cami",
    ],
    "bottom": [
        "Slim Chinos", "Wide-leg Trousers", "Straight Jeans", "Skinny Jeans",
        "Cargo Pants", "Joggers", "Linen Trousers", "Corduroy Pants",
        "Midi Skirt", "Mini Skirt", "Pleated Skirt", "Denim Shorts",
        "Tailored Shorts", "Bermuda Shorts", "Flared Jeans", "Mom Jeans",
    ],
    "outerwear": [
        "Wool Overcoat", "Trench Coat", "Puffer Jacket", "Rain Jacket",
        "Denim Jacket", "Leather Jacket", "Fleece Jacket", "Blazer",
        "Windbreaker", "Down Vest", "Peacoat", "Bomber Jacket",
    ],
    "shoes": [
        "White Sneakers", "Running Shoes", "Chelsea Boots", "Ankle Boots",
        "Loafers", "Oxford Shoes", "Sandals", "Flip Flops",
        "Hiking Boots", "Rain Boots", "Slip-on Sneakers", "Platform Sneakers",
        "Ballet Flats", "Wedge Heels", "Mules", "Brogues",
    ],
    "accessory": [
        "Wool Scarf", "Beanie Hat", "Baseball Cap", "Silk Scarf",
        "Leather Belt", "Canvas Belt", "Sunglasses", "Knit Gloves",
        "Bucket Hat", "Wool Beret", "Straw Hat", "Headband",
    ],
}

COLORS = [
    "white", "black", "navy", "grey", "beige", "olive",
    "camel", "burgundy", "forest green", "light blue", "cream",
    "charcoal", "rust", "blush pink", "cobalt blue", "mustard yellow",
    "terracotta", "slate blue", "off-white", "dark brown",
]

FABRICS = {
    "top":       ["cotton", "linen", "silk", "wool blend", "polyester", "jersey", "flannel", "cashmere blend"],
    "bottom":    ["denim", "cotton twill", "linen", "corduroy", "polyester blend", "wool blend", "jersey"],
    "outerwear": ["wool", "nylon", "polyester", "leather", "cotton canvas", "fleece", "down fill"],
    "shoes":     ["leather", "suede", "canvas", "rubber", "synthetic mesh", "nubuck"],
    "accessory": ["wool", "silk", "cotton", "leather", "acrylic", "polyester"],
}

# warmth_level → typical weather tags
WARMTH_WEATHER = {
    1: ["hot", "sunny"],
    2: ["sunny", "mild"],
    3: ["mild", "cloudy"],
    4: ["cold", "rainy", "cloudy"],
    5: ["cold", "snowy", "rainy"],
}

# category → warmth range
WARMTH_RANGE = {
    "top":       (1, 5),
    "bottom":    (1, 4),
    "outerwear": (2, 5),
    "shoes":     (1, 5),
    "accessory": (2, 5),
}

# Polyvore-style compatibility: category → list of compatible categories
COMPAT_MAP = {
    "top":       ["bottom", "outerwear", "shoes", "accessory"],
    "bottom":    ["top", "outerwear", "shoes"],
    "outerwear": ["top", "bottom", "shoes"],
    "shoes":     ["top", "bottom", "outerwear"],
    "accessory": ["top", "outerwear"],
}


def _weather_tags(warmth: int, category: str) -> str:
    """Return comma-separated weather tags appropriate for warmth level and category."""
    tags = set(WARMTH_WEATHER[warmth])
    # outerwear and accessories are always viable in rain/wind if warm enough
    if category in ("outerwear", "accessory") and warmth >= 3:
        tags.add("rainy")
    if warmth == 3:
        tags.add("mild")
    return ",".join(sorted(tags))


def _last_worn_date(rng: random.Random) -> str:
    """Return a random ISO date in the past 90 days (or never worn = empty string)."""
    if rng.random() < 0.15:          # 15 % never worn
        return ""
    days_ago = rng.randint(0, 90)
    d = datetime.today() - timedelta(days=days_ago)
    return d.strftime("%Y-%m-%d")


def generate_wardrobe(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic wardrobe DataFrame with *n* items.

    Parameters
    ----------
    n : int
        Number of wardrobe items to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        item_id, name, category, color, fabric, warmth_level,
        weather_tags, last_worn_date, compatible_with, text_description
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # Build a flat pool of (category, name) pairs, cycling if n > pool size
    pool: list[tuple[str, str]] = []
    for cat, names in CATEGORIES.items():
        for name in names:
            pool.append((cat, name))

    rows = []
    for i in range(n):
        cat, base_name = pool[i % len(pool)]

        color = rng.choice(COLORS)
        fabric = rng.choice(FABRICS[cat])

        lo, hi = WARMTH_RANGE[cat]
        warmth = rng.randint(lo, hi)

        w_tags = _weather_tags(warmth, cat)
        last_worn = _last_worn_date(rng)

        # compatible_with: 2–4 item_ids assigned later (placeholder for now)
        compatible_with = ""   # filled in post-generation pass

        name = f"{color.title()} {base_name}"

        text_description = (
            f"{color} {fabric} {base_name} suitable for "
            f"{w_tags.replace(',', ', ')} weather, warmth level {warmth}"
        )

        rows.append({
            "item_id": f"ITEM_{i:04d}",
            "name": name,
            "category": cat,
            "color": color,
            "fabric": fabric,
            "warmth_level": warmth,
            "weather_tags": w_tags,
            "last_worn_date": last_worn,
            "compatible_with": compatible_with,
            "text_description": text_description,
        })

    df = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Compatibility pass: assign 2–4 compatible item IDs per item
    # -----------------------------------------------------------------------
    id_by_cat: dict[str, list[str]] = {
        cat: df.loc[df["category"] == cat, "item_id"].tolist()
        for cat in CATEGORIES
    }

    compat_lists = []
    for _, row in df.iterrows():
        compat_cats = COMPAT_MAP.get(row["category"], [])
        picks: list[str] = []
        for c in compat_cats:
            pool_c = [x for x in id_by_cat.get(c, []) if x != row["item_id"]]
            if pool_c:
                picks.append(rng.choice(pool_c))
        compat_lists.append(",".join(picks[:4]))

    df["compatible_with"] = compat_lists

    return df


def save_wardrobe(df: pd.DataFrame, output_path: str = "data/wardrobe.csv") -> None:
    """Save wardrobe DataFrame to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} items to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic wardrobe dataset")
    parser.add_argument("--output", default="data/wardrobe.csv")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_wardrobe(n=args.n, seed=args.seed)
    save_wardrobe(df, args.output)
    print(df.head(5).to_string(index=False))
