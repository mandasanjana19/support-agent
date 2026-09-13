"""
Phase 4: Sample examples for hand-labeling.

Sampling strategy (documented — this goes in the report):
  - Stratify by TF-IDF/KMeans cluster (from Phase 3b) so all discovered
    intents get representation, not just the most common one.
  - Within each cluster, sample randomly (not cherry-picked) to avoid
    labeler bias toward "clean" examples.
  - Deduplicate near-identical boilerplate messages (common in this
    dataset, e.g. "@Brand my order is late") so the golden set isn't
    padded with copies.
  - Target: ~200 total, roughly evenly split across clusters, with a
    floor of at least 8 examples per cluster so rare intents aren't lost.

Usage:
    python eval/build_golden_set.py --brand AmazonHelp --k 8 --total 200
"""

import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

PROCESSED_DIR = os.path.join("data", "processed")
GOLDEN_DIR = os.path.join("data", "golden")


def build(brand: str, k: int, total: int, seed: int = 42):
    path = os.path.join(PROCESSED_DIR, f"{brand}_pairs.csv")
    df = pd.read_csv(path)

    # Drop near-duplicate customer messages (common boilerplate)
    df["_norm"] = df["customer_message"].fillna("").str.lower().str.strip()
    df = df.drop_duplicates(subset="_norm").drop(columns="_norm").reset_index(drop=True)

    texts = df["customer_message"].fillna("").astype(str)
    vec = TfidfVectorizer(max_features=3000, stop_words="english", min_df=3)
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    df["cluster"] = km.fit_predict(X)

    per_cluster = max(total // k, 8)
    sampled_parts = []
    for c in range(k):
        cluster_df = df[df["cluster"] == c]
        n = min(per_cluster, len(cluster_df))
        sampled_parts.append(cluster_df.sample(n=n, random_state=seed))

    sample = pd.concat(sampled_parts, ignore_index=True)
    # Top up / trim to exact total if rounding left us short/over
    if len(sample) > total:
        sample = sample.sample(n=total, random_state=seed)
    elif len(sample) < total:
        remaining = df.drop(sample.index, errors="ignore")
        extra = remaining.sample(n=min(total - len(sample), len(remaining)), random_state=seed)
        sample = pd.concat([sample, extra], ignore_index=True)

    sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    sample.insert(0, "golden_id", [f"g{i:04d}" for i in range(len(sample))])

    # Empty columns the human will fill in via label_tool.py
    sample["gold_intent"] = ""
    sample["gold_escalate"] = ""       # "yes" / "no"
    sample["gold_escalate_reason"] = ""
    sample["labeler_notes"] = ""

    os.makedirs(GOLDEN_DIR, exist_ok=True)
    out_path = os.path.join(GOLDEN_DIR, "golden_set.csv")
    cols = ["golden_id", "customer_message", "brand_reply", "cluster",
            "gold_intent", "gold_escalate", "gold_escalate_reason", "labeler_notes"]
    sample[cols].to_csv(out_path, index=False)
    print(f"Saved {len(sample)} examples to {out_path}")
    print("Cluster distribution:\n", sample["cluster"].value_counts().sort_index())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--total", type=int, default=200)
    args = parser.parse_args()
    build(args.brand, args.k, args.total)