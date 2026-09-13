"""
Phase 2: Inspect the raw Customer Support on Twitter dataset and
help select a good brand to build the agent for.

Usage:
    python src/ingest.py inspect
    python src/ingest.py filter --brand AppleSupport --limit 5000
"""

import argparse #built-in CLI arg passing
import os #os path helpers
import pandas as pd #dataframe parsing

RAW_PATH = os.path.join("data", "raw", "twcs.csv")
PROCESSED_DIR = os.path.join("data", "processed") #destination

# Columns per Kaggle dataset card:
# tweet_id, author_id, inbound, created_at, text,
# response_tweet_id, in_response_to_tweet_id
DTYPES = {
    "tweet_id": "int64",
    "author_id": "str",
    "inbound": "bool",
    "text": "str",
    "response_tweet_id": "str",
    "in_response_to_tweet_id": "str",
}


def inspect_brands(chunksize: int = 200_000, top_n: int = 25):
    """
    Stream through the raw CSV in chunks and count how many tweets
    each BRAND account (inbound == False, i.e. the company replying)
    sent. Prints the top_n most active brand accounts by volume.
    """
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError( #verify if twcs.csv exists
            f"Could not find {RAW_PATH}. Did Phase 1 download complete?"
        )

    brand_counts = pd.Series(dtype="int64")
    total_rows = 0

    reader = pd.read_csv(
        RAW_PATH,
        chunksize=chunksize,
        dtype=DTYPES,
        on_bad_lines="skip",
    )

    for i, chunk in enumerate(reader): #loops over 200,000 row batch
        total_rows += len(chunk)
        brand_chunk = chunk[chunk["inbound"] == False]  # noqa: E712 (filters company tweets/outbound)
        counts = brand_chunk["author_id"].value_counts()
        brand_counts = brand_counts.add(counts, fill_value=0)
        print(f"  processed chunk {i + 1} ({total_rows:,} rows so far)...")

    print(f"\nTotal rows scanned: {total_rows:,}")
    print(f"\nTop {top_n} brand accounts by outbound tweet volume:\n")
    print(brand_counts.sort_values(ascending=False).head(top_n)) #top 25


def filter_brand(brand: str, limit: int | None = None):
    """
    Pull all rows involving a single brand account (as sender OR as the
    account a customer is replying to), save to data/processed/<brand>.csv
    """
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Could not find {RAW_PATH}. Did Phase 1 download complete?"
        )
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    matched_rows = []
    reader = pd.read_csv(
        RAW_PATH,
        chunksize=200_000,
        dtype=DTYPES,
        on_bad_lines="skip",
    )

    for i, chunk in enumerate(reader):
        mask = chunk["author_id"] == brand
        matched = chunk[mask]
        if len(matched):
            matched_rows.append(matched) #saves rows from current chunk into a list
        print(f"  scanned chunk {i + 1}, matched so far: "
              f"{sum(len(m) for m in matched_rows):,}")
        if limit and sum(len(m) for m in matched_rows) >= limit:
            break #stop if limit reaches

    if not matched_rows:
        print(f"No rows found for brand '{brand}'. Check the exact author_id "
              f"from `inspect` output (case-sensitive).")
        return

    result = pd.concat(matched_rows, ignore_index=True) #export filtered records to processed
    if limit:
        result = result.head(limit)

    out_path = os.path.join(PROCESSED_DIR, f"{brand}_outbound.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved {len(result):,} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser() #creates CLI frameowkr
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inspect") 

    filter_parser = sub.add_parser("filter")
    filter_parser.add_argument("--brand", required=True,
                                help="Exact author_id, e.g. AppleSupport")
    filter_parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    if args.command == "inspect":
        inspect_brands()
    elif args.command == "filter":
        filter_brand(args.brand, args.limit)