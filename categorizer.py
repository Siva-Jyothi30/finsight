"""
categorizer.py - FinSight Personal Finance Analytics Agent
===========================================================
Maps the 14 raw bank transaction categories from the dataset
into 7 clean, human-readable FinSight spending categories.

This module is a pure transformation layer.
It takes a DataFrame (from data_loader.py) and returns an enriched
DataFrame with a new ``spending_category`` column. No I/O, no side
effects, fully unit-testable.

Raw category → FinSight category mapping
-----------------------------------------
food_dining, grocery_pos, grocery_net  → Food & Groceries
shopping_net, shopping_pos             → Shopping
gas_transport, travel                  → Transport & Travel
entertainment                          → Entertainment
health_fitness, personal_care          → Health & Wellness
home, kids_pets                        → Home & Family
misc_net, misc_pos                     → Miscellaneous
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Category mapping: all label translations live here
# ---------------------------------------------------------------------------

# Maps every raw bank category to a clean FinSight display label.
CATEGORY_MAP: dict[str, str] = {
    # Food
    "food_dining":    "Food & Groceries",
    "grocery_pos":    "Food & Groceries",
    "grocery_net":    "Food & Groceries",
    # Shopping
    "shopping_pos":   "Shopping",
    "shopping_net":   "Shopping",
    # Transport & Travel
    "gas_transport":  "Transport & Travel",
    "travel":         "Transport & Travel",
    # Entertainment
    "entertainment":  "Entertainment",
    # Health
    "health_fitness": "Health & Wellness",
    "personal_care":  "Health & Wellness",
    # Home & Family
    "home":           "Home & Family",
    "kids_pets":      "Home & Family",
    # Miscellaneous (catch-all)
    "misc_pos":       "Miscellaneous",
    "misc_net":       "Miscellaneous",
}

# Ordered list of categories (used by visualizations for consistent color mapping).
CATEGORY_ORDER: list[str] = [
    "Food & Groceries",
    "Shopping",
    "Transport & Travel",
    "Entertainment",
    "Health & Wellness",
    "Home & Family",
    "Miscellaneous",
]

# Color palette: one hex color per category, in the same order as CATEGORY_ORDER.
# Used across all Plotly charts for visual consistency.
CATEGORY_COLORS: dict[str, str] = {
    "Food & Groceries":  "#2ecc71",   # green
    "Shopping":          "#3498db",   # blue
    "Transport & Travel":"#e67e22",   # orange
    "Entertainment":     "#9b59b6",   # purple
    "Health & Wellness": "#e74c3c",   # red
    "Home & Family":     "#1abc9c",   # teal
    "Miscellaneous":     "#95a5a6",   # grey
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def categorize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``spending_category`` column to the DataFrame by mapping
    each raw ``category`` value through ``CATEGORY_MAP``.

    Parameters
    ----------
    df : pd.DataFrame
        A cleaned DataFrame returned by ``data_loader.load_data()``.
        Must contain a ``category`` column.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with one new column:
        - ``spending_category`` (str): human-readable category label.

    Notes
    -----
    Unknown raw categories are mapped to ``"Miscellaneous"`` so the
    pipeline never breaks on unexpected data.
    """
    _validate_input(df)

    # Map raw → clean; fall back to "Miscellaneous" for any unknown values.
    df = df.copy()  # avoid mutating the caller's DataFrame
    df["spending_category"] = (
        df["category"]
        .map(CATEGORY_MAP)
        .fillna("Miscellaneous")
    )

    return df


def get_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary table of total spend and transaction count per category.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``spending_category`` and ``amount`` columns
        (i.e. after calling ``categorize()``).

    Returns
    -------
    pd.DataFrame
        Columns: spending_category | total_spent | transaction_count | avg_transaction
        Sorted by total_spent descending.
    """
    summary = (
        df.groupby("spending_category", observed=True)
        .agg(
            total_spent=("amount", "sum"),
            transaction_count=("amount", "count"),
            avg_transaction=("amount", "mean"),
        )
        .reset_index()
        .sort_values("total_spent", ascending=False)
        .reset_index(drop=True)
    )

    # Round monetary values to 2 decimal places for display cleanliness
    summary["total_spent"]     = summary["total_spent"].round(2)
    summary["avg_transaction"] = summary["avg_transaction"].round(2)

    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_input(df: pd.DataFrame) -> None:
    """Raise ValueError if the required column is missing."""
    if "category" not in df.columns:
        raise ValueError(
            "Input DataFrame must contain a 'category' column. "
            "Did you forget to call data_loader.load_data() first?"
        )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import load_data

    df = load_data(sample_size=10_000)
    df = categorize(df)

    print("=== New column added ===")
    print(df[["category", "spending_category"]].drop_duplicates().sort_values("category").to_string())
    print()

    print("=== Spending summary ===")
    summary = get_summary(df)
    print(summary.to_string(index=False))
    print()

    print("=== Unmapped categories (should be empty) ===")
    unmapped = df[df["spending_category"] == "Miscellaneous"]["category"].unique()
    raw_misc  = {"misc_pos", "misc_net"}
    truly_unknown = [c for c in unmapped if c not in raw_misc]
    print(truly_unknown if truly_unknown else "None - all raw categories mapped correctly.")
