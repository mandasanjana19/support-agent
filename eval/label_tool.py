"""
Phase 4: A tiny CLI to hand-label the golden set.
Resumable — skips rows already labeled. Saves after every entry so
you can Ctrl+C any time without losing work.

Usage:
    python eval/label_tool.py
"""

import os
import pandas as pd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from intents import TAXONOMY  # noqa: E402

GOLDEN_PATH = os.path.join("data", "golden", "golden_set.csv")


def main():
    df = pd.read_csv(GOLDEN_PATH, keep_default_na=False)
    intent_names = list(TAXONOMY.keys())

    print("Intent taxonomy:")
    for i, name in enumerate(intent_names):
        print(f"  [{i}] {name} — {TAXONOMY[name]}")
    print("\nType the number for intent, 'y'/'n' for escalate, then a short reason.")
    print("Ctrl+C any time to stop; progress is saved after each row.\n")

    for idx, row in df.iterrows():
        if row["gold_intent"]:  # already labeled, skip
            continue

        print("=" * 70)
        print(f"[{row['golden_id']}]  CUSTOMER: {row['customer_message']}")
        print(f"(actual brand reply, for context only): {row['brand_reply']}")

        while True:
            raw = input(f"Intent (0-{len(intent_names) - 1}): ").strip()
            if raw.isdigit() and 0 <= int(raw) < len(intent_names):
                intent = intent_names[int(raw)]
                break
            print("Invalid — enter a number from the list above.")

        while True:
            esc = input("Should this be escalated to a human? (y/n): ").strip().lower()
            if esc in ("y", "n"):
                break
        reason = input("Why (short reason for the escalate/auto decision): ").strip()
        notes = input("Any other notes (optional, Enter to skip): ").strip()

        df.at[idx, "gold_intent"] = intent
        df.at[idx, "gold_escalate"] = "yes" if esc == "y" else "no"
        df.at[idx, "gold_escalate_reason"] = reason
        df.at[idx, "labeler_notes"] = notes

        df.to_csv(GOLDEN_PATH, index=False)  # save progress every row
        print(f"Saved. ({df['gold_intent'].astype(bool).sum()}/{len(df)} labeled)\n")

    print("\nAll rows labeled!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped — progress saved. Run again to resume.")