"""
Phase 5: Two baselines to compare the real agent against.

  1. Trivial baseline:
       - Intent: always predict the single most common intent (majority class).
       - Routing: always the same decision (whatever is most common in gold labels).
     This is the "a broken clock is right twice a day" floor — if the real
     agent can't beat this, it's not worth building.

  2. Simple baseline:
       - Intent: TF-IDF + Logistic Regression, trained on a train split of
         the golden set (our only labeled data — documented limitation).
       - Routing: keyword/rule-based (no ML) — escalate if certain risk
         keywords appear (fraud/hacked/legal/lawsuit/cancel account/refund
         dispute), else auto-handle.

Usage:
    python src/baselines.py evaluate --golden data/golden/golden_set.csv
"""

import argparse
import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

RESULTS_DIR = os.path.join("eval", "results")

ESCALATE_KEYWORDS = [
    "fraud", "hacked", "hack", "lawyer", "lawsuit", "legal action",
    "sue", "scam", "unauthorized", "stolen", "police", "identity theft",
    "cancel my account", "never buying", "disgusting", "worst company",
]


def load_labeled(golden_path: str) -> pd.DataFrame:
    df = pd.read_csv(golden_path, keep_default_na=False)
    df = df[df["gold_intent"] != ""].reset_index(drop=True)
    if len(df) < 20:
        raise ValueError(
            f"Only {len(df)} labeled rows found. Label at least ~20-30 "
            f"rows in golden_set.csv before running baselines."
        )
    return df


def trivial_baseline(train_df, test_df):
    majority_intent = train_df["gold_intent"].mode()[0]
    majority_escalate = train_df["gold_escalate"].mode()[0]

    pred_intent = [majority_intent] * len(test_df)
    pred_escalate = [majority_escalate] * len(test_df)
    return pred_intent, pred_escalate


def simple_baseline_intent(train_df, test_df):
    vec = TfidfVectorizer(max_features=2000, stop_words="english", min_df=1)
    X_train = vec.fit_transform(train_df["customer_message"].astype(str))
    X_test = vec.transform(test_df["customer_message"].astype(str))

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, train_df["gold_intent"])
    pred = clf.predict(X_test)
    return list(pred)


def simple_baseline_routing(test_df):
    preds = []
    for msg in test_df["customer_message"].astype(str):
        lower = msg.lower()
        hit = any(kw in lower for kw in ESCALATE_KEYWORDS)
        preds.append("yes" if hit else "no")
    return preds


def evaluate(golden_path: str, seed: int = 42):
    df = load_labeled(golden_path)
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=seed,
        stratify=df["gold_intent"] if df["gold_intent"].nunique() > 1 else None,
    )
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    results = {"n_train": len(train_df), "n_test": len(test_df)}

    # --- Trivial baseline ---
    triv_intent, triv_escalate = trivial_baseline(train_df, test_df)
    results["trivial_intent_accuracy"] = accuracy_score(test_df["gold_intent"], triv_intent)
    results["trivial_intent_f1_macro"] = f1_score(
        test_df["gold_intent"], triv_intent, average="macro", zero_division=0
    )
    results["trivial_escalate_accuracy"] = accuracy_score(test_df["gold_escalate"], triv_escalate)

    # --- Simple baseline ---
    simp_intent = simple_baseline_intent(train_df, test_df)
    results["simple_intent_accuracy"] = accuracy_score(test_df["gold_intent"], simp_intent)
    results["simple_intent_f1_macro"] = f1_score(
        test_df["gold_intent"], simp_intent, average="macro", zero_division=0
    )

    simp_escalate = simple_baseline_routing(test_df)
    results["simple_escalate_accuracy"] = accuracy_score(test_df["gold_escalate"], simp_escalate)

    print("\n=== Baseline Results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Save the actual train/test split + predictions for later comparison
    test_df = test_df.copy()
    test_df["trivial_pred_intent"] = triv_intent
    test_df["trivial_pred_escalate"] = triv_escalate
    test_df["simple_pred_intent"] = simp_intent
    test_df["simple_pred_escalate"] = simp_escalate
    test_df.to_csv(os.path.join(RESULTS_DIR, "baseline_test_predictions.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--golden", default=os.path.join("data", "golden", "golden_set.csv"))
    args = parser.parse_args()

    if args.command == "evaluate":
        evaluate(args.golden)