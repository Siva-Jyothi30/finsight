"""
data_loader.py - FinSight Personal Finance Analytics Agent
===========================================================
Responsible for loading the raw transaction CSV, performing
all cleaning and type-casting steps, and returning a tidy
DataFrame that every other module can rely on.

Dataset: Credit Card Transactions (Kaggle)
Shape  : ~1.3 M rows × 24 columns
Period : January 2019 – June 2020
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default path to the raw data file, relative to the project root.
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "credit_card_transactions.csv")

# Columns we actually need for financial analysis.
# Dropping PII (street, dob), geo coordinates, and internal IDs.
COLUMNS_TO_KEEP = [
    "trans_date_trans_time",  # full transaction timestamp
    "merchant",               # merchant name (will be cleaned)
    "category",               # raw spending category from the bank
    "amt",                    # transaction amount in USD
    "city",                   # cardholder city
    "state",                  # cardholder state
    "is_fraud",               # fraud flag (0 = legitimate, 1 = fraud)
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(filepath: str = DATA_PATH, sample_size: int | None = None) -> pd.DataFrame:
    """
    Load, clean, and return the transaction dataset as a tidy DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file. Defaults to ``data/credit_card_transactions.csv``.
    sample_size : int or None
        If provided, randomly sample this many rows (reproducible via seed=42).
        Useful for fast prototyping; set to None to load the full ~1.3 M rows.

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame with standardized column names, parsed dates,
        and derived time columns ready for analysis and visualization.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    """
    _validate_filepath(filepath)

    # ------------------------------------------------------------------
    # 1. Read raw CSV
    # ------------------------------------------------------------------
    print(f"[data_loader] Loading data from: {filepath}")
    df = pd.read_csv(filepath, usecols=COLUMNS_TO_KEEP, low_memory=False)
    print(f"[data_loader] Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ------------------------------------------------------------------
    # 2. Optional random sample (for fast dev iteration)
    # ------------------------------------------------------------------
    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
        print(f"[data_loader] Sampled down to: {len(df):,} rows")

    # ------------------------------------------------------------------
    # 3. Parse & enrich the timestamp column
    # ------------------------------------------------------------------
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    # Extract granular time fields used by the visualizations layer.
    df["date"]       = df["trans_date_trans_time"].dt.date          # calendar date
    df["year"]       = df["trans_date_trans_time"].dt.year          # 4-digit year
    df["month"]      = df["trans_date_trans_time"].dt.month         # 1–12
    df["month_name"] = df["trans_date_trans_time"].dt.strftime("%b %Y")  # e.g. "Jan 2019"
    df["day_of_week"]= df["trans_date_trans_time"].dt.day_name()   # e.g. "Monday"
    df["hour"]       = df["trans_date_trans_time"].dt.hour          # 0–23

    # ------------------------------------------------------------------
    # 4. Clean merchant names
    #    The raw dataset prefixes every merchant with "fraud_" as an
    #    artifact of the fraud-detection labelling process. Strip it.
    # ------------------------------------------------------------------
    df["merchant"] = df["merchant"].str.removeprefix("fraud_").str.strip()

    # ------------------------------------------------------------------
    # 5. Standardize column names & data types
    # ------------------------------------------------------------------
    df.rename(columns={
        "trans_date_trans_time": "datetime",
        "amt": "amount",
        "is_fraud": "is_fraud",   # kept as int (0/1) intentionally
    }, inplace=True)

    # Ensure amount is float (should already be, but be explicit)
    df["amount"] = df["amount"].astype(float)

    # city / state as clean strings
    df["city"]  = df["city"].str.strip().str.title()
    df["state"] = df["state"].str.strip().str.upper()

    # ------------------------------------------------------------------
    # 6. Drop any remaining null rows (only merch_zipcode had nulls;
    #    we dropped that column, so this is a safety net)
    # ------------------------------------------------------------------
    before = len(df)
    df.dropna(subset=["datetime", "amount", "merchant", "category"], inplace=True)
    after = len(df)
    if before != after:
        print(f"[data_loader] Dropped {before - after:,} rows with nulls in key columns.")

    # ------------------------------------------------------------------
    # 7. Reset index so downstream code can rely on a clean 0-based index
    # ------------------------------------------------------------------
    df.reset_index(drop=True, inplace=True)

    print(f"[data_loader] Clean shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"[data_loader] Date range  : {df['datetime'].min().date()} → {df['datetime'].max().date()}")
    print(f"[data_loader] Amount range: ${df['amount'].min():.2f} – ${df['amount'].max():,.2f}")
    print(f"[data_loader] Done.\n")

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_filepath(filepath: str) -> None:
    """Raise FileNotFoundError with a helpful message if the CSV is missing."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            f"Make sure 'credit_card_transactions.csv' is inside the data/ folder."
        )


# ---------------------------------------------------------------------------
# Quick self-test: run this file directly to verify the loader works
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data(sample_size=10_000)   # use 10 K rows for a fast smoke test

    print("=== Column overview ===")
    print(df.dtypes)
    print()
    print("=== First 5 rows ===")
    print(df.head().to_string())
    print()
    print("=== Unique raw categories ===")
    print(sorted(df["category"].unique()))
