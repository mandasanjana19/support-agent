"""
Phase 7c: Human-agreement check for the LLM judge.

You hand-score a random sample of rows from agent_eval_full.csv on the
same 3 dimensions the judge used (WITHOUT seeing the judge's scores --
this file deliberately hides them while you label). Then we compute
agreement between you and the judge.

Usage:
    python eval/human_agreement.py label --n 20
    python eval/human_agreement.py score
"""

import argparse
import os
import pandas as pd
import numpy as np

RESULTS_DIR = os.path.join("eval", "results")
FULL_PATH = os.path.join(RESULTS_DIR, "agent_eval_full.csv")
HUMAN_PATH = os.path.join(RESULTS_DIR, "human_agreement_labels.csv")


def label(n: int, seed: int = 7):
    df = pd.read_csv(FULL_PATH)

    if os.path.exists(HUMAN_PATH):
        human_df = pd.read_csv(HUMAN_PATH, keep_default_na=False)
    else:
        sample = df.sample(n=min(n, len(df)), random_state=seed).copy()
        human_df = sample[["golden_id", "customer_message", "draft_reply"]].copy()
        human_df["human_groundedness"] = ""
        human_df["human_tone_fit"] = ""
        human_df["human_helpfulness"] = ""
        human_df.to_csv(HUMAN_PATH, index=False)
        human_df = pd.read_csv(HUMAN_PATH, keep_default_na=False)

    print("Scoring blind -- judge's scores are hidden from you during labeling.")
    print("Scale: 0 (terrible) to 5 (excellent) for each dimension.\n")

    for idx, row in human_df.iterrows():
        if row["human_groundedness"] != "":
            continue
        print("=" * 70)
        print(f"[{row['golden_id']}] CUSTOMER: {row['customer_message']}")
        print(f"DRAFT REPLY: {row['draft_reply']}")
        g = input("groundedness (0-5): ").strip()
        t = input("tone_fit (0-5): ").strip()
        h = input("helpfulness (0-5): ").strip()
        human_df.at[idx, "human_groundedness"] = g
        human_df.at[idx, "human_tone_fit"] = t
        human_df.at[idx, "human_helpfulness"] = h
        human_df.to_csv(HUMAN_PATH, index=False)
        print("saved.\n")

    print("Done labeling. Run `python eval/human_agreement.py score` next.")


def score():
    human_df = pd.read_csv(HUMAN_PATH)
    full_df = pd.read_csv(FULL_PATH)
    merged = human_df.merge(
        full_df[["golden_id", "judge_groundedness", "judge_tone_fit", "judge_helpfulness"]],
        on="golden_id",
    )

    print(f"\nAgreement analysis over {len(merged)} human-labeled examples:\n")
    results = {}
    for dim in ["groundedness", "tone_fit", "helpfulness"]:
        h = merged[f"human_{dim}"].astype(float)
        j = merged[f"judge_{dim}"].astype(float)

        exact_match_rate = (h == j).mean()
        within_1_rate = (abs(h - j) <= 1).mean()
        corr = np.corrcoef(h, j)[0, 1] if h.std() > 0 and j.std() > 0 else float("nan")
        mae = abs(h - j).mean()

        results[dim] = {
            "exact_match_rate": exact_match_rate,
            "within_1_point_rate": within_1_rate,
            "pearson_corr": corr,
            "mean_abs_error": mae,
        }
        print(f"{dim}: exact_match={exact_match_rate:.2f}  within_1={within_1_rate:.2f}  "
              f"corr={corr:.2f}  MAE={mae:.2f}")

    import json
    with open(os.path.join(RESULTS_DIR, "judge_human_agreement.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}\\judge_human_agreement.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    lbl = sub.add_parser("label")
    lbl.add_argument("--n", type=int, default=20)
    sub.add_parser("score")
    args = parser.parse_args()

    if args.command == "label":
        label(args.n)
    elif args.command == "score":
        score()