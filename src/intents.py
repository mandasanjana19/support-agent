"""
Phase 3b: Discover candidate intent clusters from real customer messages,
then hold the hand-finalized taxonomy used everywhere downstream.

Usage:
    python src/intents.py discover --brand AmazonHelp --k 8
"""

import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

PROCESSED_DIR = os.path.join("data", "processed")

# --------------------------------------------------------------------------
# FINAL TAXONOMY — fill this in AFTER running `discover` and reading the
# output. This dict is imported by every other module (baselines, agent,
# eval), so it is the single source of truth for intent labels.
# --------------------------------------------------------------------------
TAXONOMY = {
    "order_delivery_issue": "Missing, delayed, damaged, or misdelivered order/package.",
    "refund_or_billing": "Refund requests, incorrect charges, payment/subscription billing problems.",
    "account_access": "Login, password reset, account locked/hacked, security concerns.",
    "product_technical_issue": "App/device/product not working, bug, error message.",
    "return_or_cancellation": "Wants to return an item or cancel an order/subscription.",
    "general_inquiry": "How-to question, status check, or general information request.",
    "complaint_no_specific_ask": "Venting frustration/dissatisfaction without a clear actionable request.",
    "other": "Doesn't fit cleanly into the above; catch-all for edge cases.",
}
# --------------------------------------------------------------------------


def discover(brand: str, k: int = 8, samples_per_cluster: int = 6):
    path = os.path.join(PROCESSED_DIR, f"{brand}_pairs.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run Phase 3a (threads.py) first.")

    df = pd.read_csv(path)
    texts = df["customer_message"].fillna("").astype(str)

    vec = TfidfVectorizer(max_features=3000, stop_words="english", min_df=3)
    X = vec.fit_transform(texts)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    df["cluster"] = labels

    terms = vec.get_feature_names_out()
    print(f"\n{'=' * 70}\nDiscovered {k} clusters from {len(df):,} messages\n{'=' * 70}")

    for c in range(k):
        center = km.cluster_centers_[c]
        top_idx = center.argsort()[::-1][:10]
        top_terms = [terms[i] for i in top_idx]
        cluster_df = df[df["cluster"] == c]
        print(f"\n--- Cluster {c} (n={len(cluster_df)}) ---")
        print(f"Top terms: {', '.join(top_terms)}")
        print("Sample messages:")
        for msg in cluster_df["customer_message"].head(samples_per_cluster):
            snippet = str(msg).replace("\n", " ")[:140]
            print(f"  - {snippet}")

    print(f"\n{'=' * 70}")
    print("Now hand-edit TAXONOMY in src/intents.py based on what you saw above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    disc = sub.add_parser("discover")
    disc.add_argument("--brand", required=True)
    disc.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    if args.command == "discover":
        discover(args.brand, args.k)