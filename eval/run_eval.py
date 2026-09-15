"""
Phase 7a: Run the real agent on the same held-out test split baselines
used, score with automated metrics + LLM judge, save everything to CSV
so a human can review / re-score a subset afterward.

Usage:
    python eval/run_eval.py --brand AppleSupport
"""

import os
import sys
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agent import run_agent          # noqa: E402
from retrieval import HistoricalRetriever  # noqa: E402
from judge import judge_reply        # noqa: E402

RESULTS_DIR = os.path.join("eval", "results")
GOLDEN_PATH = os.path.join("data", "golden", "golden_set.csv")


def main(brand: str, seed: int = 42):
    df = pd.read_csv(GOLDEN_PATH, keep_default_na=False)
    df = df[df["gold_intent"] != ""].reset_index(drop=True)
    if len(df) < 10:
        raise ValueError(f"Only {len(df)} labeled rows. Label more before running eval.")

    # SAME split logic/seed as baselines.py -> apples-to-apples test set
    _, test_df = train_test_split(
        df, test_size=0.3, random_state=seed,
        stratify=df["gold_intent"] if df["gold_intent"].nunique() > 1 else None,
    )
    print(f"Running agent on {len(test_df)} held-out test examples...")

    retriever = HistoricalRetriever(brand)  # build TF-IDF index once, reuse

    rows = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        print(f"  [{i + 1}/{len(test_df)}] {row['golden_id']}")
        result = run_agent(
            brand,
            row["customer_message"],
            exclude_customer_message=row["customer_message"],  # avoid self-retrieval leak
            retriever=retriever,
        )
        judge_scores = judge_reply(
            row["customer_message"], result["retrieved_examples"], result["draft_reply"]
        )

        rows.append({
            "golden_id": row["golden_id"],
            "customer_message": row["customer_message"],
            "gold_intent": row["gold_intent"],
            "pred_intent": result["intent"],
            "gold_escalate": row["gold_escalate"],
            "pred_escalate": "yes" if result["escalate"] else "no",
            "pred_escalate_reason": result["escalate_reason"],
            "n_retrieved": len(result["retrieved_examples"]),
            "top_similarity": (result["retrieved_examples"][0]["similarity"]
                                if result["retrieved_examples"] else 0.0),
            "draft_reply": result["draft_reply"],
            "judge_groundedness": judge_scores.get("groundedness"),
            "judge_tone_fit": judge_scores.get("tone_fit"),
            "judge_helpfulness": judge_scores.get("helpfulness"),
            "judge_groundedness_rationale": judge_scores.get("groundedness_rationale"),
            "judge_tone_fit_rationale": judge_scores.get("tone_fit_rationale"),
            "judge_helpfulness_rationale": judge_scores.get("helpfulness_rationale"),
        })

    out_df = pd.DataFrame(rows)

    metrics = {
        "n_test": len(out_df),
        "agent_intent_accuracy": accuracy_score(out_df["gold_intent"], out_df["pred_intent"]),
        "agent_intent_f1_macro": f1_score(
            out_df["gold_intent"], out_df["pred_intent"], average="macro", zero_division=0
        ),
        "agent_escalate_accuracy": accuracy_score(out_df["gold_escalate"], out_df["pred_escalate"]),
        "judge_groundedness_mean": out_df["judge_groundedness"].mean(),
        "judge_tone_fit_mean": out_df["judge_tone_fit"].mean(),
        "judge_helpfulness_mean": out_df["judge_helpfulness"].mean(),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_df.to_csv(os.path.join(RESULTS_DIR, "agent_eval_full.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "agent_eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Agent Eval Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nSaved full results to {RESULTS_DIR}\\agent_eval_full.csv")

    baseline_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        print("\n=== Comparison to Baselines ===")
        print(f"  intent accuracy   -> trivial: {baseline['trivial_intent_accuracy']:.2f} | "
              f"simple: {baseline['simple_intent_accuracy']:.2f} | "
              f"agent: {metrics['agent_intent_accuracy']:.2f}")
        print(f"  escalate accuracy -> trivial: {baseline['trivial_escalate_accuracy']:.2f} | "
              f"simple: {baseline['simple_escalate_accuracy']:.2f} | "
              f"agent: {metrics['agent_escalate_accuracy']:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", required=True)
    args = parser.parse_args()
    main(args.brand)