"""
Phase 3b: Discover candidate intent clusters from real customer messages,
then hold the hand-finalized taxonomy used everywhere downstream.

Usage:
    python src/intents.py discover --brand AmazonHelp --k 8
"""

import argparse
import os #windows:(\) | macOS,Linux(/)
import pandas as pd #laods csv intro structured DF table
from sklearn.feature_extraction.text import TfidfVectorizer #term-freq inverse doc freq
from sklearn.cluster import KMeans

PROCESSED_DIR = os.path.join("data", "processed")

# --------------------------------------------------------------------------
# FINAL TAXONOMY — fill this in AFTER running `discover` and reading the
# output. This dict is imported by every other module (baselines, agent,
# eval), so it is the single source of truth for intent labels.
# --------------------------------------------------------------------------

#py dictionary mapping standardized intent keys to clear human description
#INITIAL TAXONOMY of general ecommerce support system

# --------------------------------------------------------------------------
# FINAL TAXONOMY — Tailored for AppleSupport based on Phase 3b discovery.
# --------------------------------------------------------------------------
TAXONOMY = {
    "system_update_issue": "Issues related to iOS/watchOS software updates, update installation failures, or version checks.",
    "battery_performance": "Battery drain, short battery life, or power management issues after updates or usage.",
    "software_bug_glitch": "UI/keyboard glitches, typing bugs (e.g., 'I' symbol bug), app crashes, or unexpected system behavior.",
    "apple_id_account": "Apple ID login, region change, security settings, or account access configuration.",
    "app_store_media": "Issues with Apple Music, App Store downloads, media syncing, or control center playback.",
    "device_hardware_connectivity": "Hardware defects, screen display issues, Wi-Fi/Bluetooth connections, or Watch responsiveness.",
    "complaint_no_specific_ask": "Venting frustration or dissatisfaction regarding updates/products without a specific actionable request.",
    "other": "Doesn't fit cleanly into the above; catch-all for edge cases or ambiguous messages."
}
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

#clusters cust txt auto without requiring manual rading
def discover(brand: str, k: int = 8, samples_per_cluster: int = 6):
    path = os.path.join(PROCESSED_DIR, f"{brand}_pairs.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run Phase 3a (threads.py) first.")

    df = pd.read_csv(path)
    texts = df["customer_message"].fillna("").astype(str)

    vec = TfidfVectorizer(max_features=3000, stop_words="english", min_df=3)
    X = vec.fit_transform(texts) #ignores stopwords

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    df["cluster"] = labels

    terms = vec.get_feature_names_out()
    print(f"\n{'=' * 70}\nDiscovered {k} clusters from {len(df):,} messages\n{'=' * 70}")

    for c in range(k):
        center = km.cluster_centers_[c]
        top_idx = center.argsort()[::-1][:10] #fr each clstr find 10 most influential vocab keywrds based on clstr centroid wgts
        top_terms = [terms[i] for i in top_idx]
        cluster_df = df[df["cluster"] == c]
        print(f"\n--- Cluster {c} (n={len(cluster_df)}) ---")
        print(f"Top terms: {', '.join(top_terms)}")
        print("Sample messages:") #print sample tweets
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