"""
Phase 3a: Reconstruct (customer_message -> brand_reply) pairs for one brand.

Two-pass approach over the raw CSV (kept memory-safe via chunking):
  Pass 1: find all outbound tweets from `brand` that are replies to
          something (in_response_to_tweet_id is not null), collect the
          set of tweet_ids they are replying to.
  Pass 2: find the text of those referenced tweets (the customer's
          original message).
Then join the two in pandas.

Usage:
    python src/threads.py --brand AmazonHelp --limit 3000
"""

import argparse
import os
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "twcs.csv")
PROCESSED_DIR = os.path.join("data", "processed")

READ_KWARGS = dict(
    dtype={
        "tweet_id": "str", #forcing IDs to strings
        "author_id": "str",
        "text": "str",
        "response_tweet_id": "str",
        "in_response_to_tweet_id": "str",
    },
    on_bad_lines="skip", #skipping corrupted lines
)


def norm_id(x):
    """Normalize an id that may show up as '123', '123.0', or NaN."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def build_pairs(brand: str, limit: int | None = None, chunksize: int = 200_000):
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Missing {RAW_PATH}. Run Phase 1 download first.")

    # ---- Pass 1: collect the brand's outbound replies ----
    outbound_rows = []
    print("Pass 1/2: scanning for brand outbound replies...")
    for i, chunk in enumerate(pd.read_csv(RAW_PATH, chunksize=chunksize, **READ_KWARGS)):
        chunk["tweet_id"] = chunk["tweet_id"].map(norm_id)
        chunk["in_response_to_tweet_id"] = chunk["in_response_to_tweet_id"].map(norm_id)

        mask = (chunk["author_id"] == brand) & chunk["in_response_to_tweet_id"].notna() #inresponse to tweet shouldnt be empty
        matched = chunk.loc[mask, ["tweet_id", "in_response_to_tweet_id", "text", "created_at"]]
        if len(matched):
            outbound_rows.append(matched)
        print(f"  chunk {i + 1}: matched so far = "
              f"{sum(len(m) for m in outbound_rows):,}")

    if not outbound_rows:
        print(f"No outbound replies found for brand '{brand}'. Check the exact author_id.")
        return

    outbound_df = pd.concat(outbound_rows, ignore_index=True)
    outbound_df = outbound_df.rename(columns={
        "tweet_id": "brand_tweet_id",
        "text": "brand_reply",
        "created_at": "brand_reply_time",
    })
    if limit:
        outbound_df = outbound_df.head(limit) #caps them at your specified --limit, eg:3000

    needed_ids = set(outbound_df["in_response_to_tweet_id"].dropna().unique())
    print(f"\nNeed to find {len(needed_ids):,} original customer tweets.")

    # ---- Pass 2: find the customer's original messages ----
    #if ID is present in neededIDs thenextract the tweet's text and saves to dictionary
    inbound_lookup = {}
    print("Pass 2/2: scanning for referenced customer tweets...")
    for i, chunk in enumerate(pd.read_csv(RAW_PATH, chunksize=chunksize, **READ_KWARGS)):
        chunk["tweet_id"] = chunk["tweet_id"].map(norm_id)
        matched = chunk[chunk["tweet_id"].isin(needed_ids)]
        for _, row in matched.iterrows():
            inbound_lookup[row["tweet_id"]] = row["text"]
        print(f"  chunk {i + 1}: found {len(inbound_lookup):,}/{len(needed_ids):,} so far")
        if len(inbound_lookup) >= len(needed_ids):
            break

    outbound_df["customer_message"] = outbound_df["in_response_to_tweet_id"].map(inbound_lookup)
    pairs = outbound_df.dropna(subset=["customer_message"]).reset_index(drop=True) #drop any unmatched rows
    pairs = pairs.rename(columns={"in_response_to_tweet_id": "customer_tweet_id"})
    pairs = pairs[["customer_tweet_id", "customer_message",
                    "brand_tweet_id", "brand_reply", "brand_reply_time"]]

    #export:save final dataset to pata/processed/<brand>_pairs.csv

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, f"{brand}_pairs.csv") 
    pairs.to_csv(out_path, index=False)
    print(f"\nSaved {len(pairs):,} customer<->brand pairs to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", required=True)
    parser.add_argument("--limit", type=int, default=3000,
                         help="Cap on brand replies to pair (keeps runtime sane)")
    args = parser.parse_args()
    build_pairs(args.brand, args.limit)