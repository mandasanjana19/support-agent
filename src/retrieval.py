"""
Phase 6a: Retrieval over historical (customer_message -> brand_reply) pairs,
using TF-IDF cosine similarity (see decision_log.md for why not embeddings).

Given a new customer message, finds the k most similar past customer
messages and returns their (message, brand_reply) pairs as grounding
context for the reply-drafting LLM call.
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED_DIR = os.path.join("data", "processed")


class HistoricalRetriever:
    def __init__(self, brand: str):
        path = os.path.join(PROCESSED_DIR, f"{brand}_pairs.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run src/threads.py first.")

        self.df = pd.read_csv(path).dropna(subset=["customer_message", "brand_reply"])
        self.df = self.df.reset_index(drop=True)

        self.vec = TfidfVectorizer(max_features=3000, stop_words="english", min_df=2)
        self.matrix = self.vec.fit_transform(self.df["customer_message"].astype(str))

    def retrieve(self, query: str, k: int = 3, exclude_customer_message: str | None = None):
        """Return top-k most similar historical pairs, as a list of dicts."""
        q_vec = self.vec.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]

        top_idx = sims.argsort()[::-1]
        results = []
        for idx in top_idx:
            if len(results) >= k:
                break
            row = self.df.iloc[idx]
            # Don't let a message retrieve itself when evaluating on
            # historical data (would trivially "leak" the gold answer).
            if exclude_customer_message and row["customer_message"] == exclude_customer_message:
                continue
            if sims[idx] <= 0:
                continue
            results.append({
                "similarity": float(sims[idx]),
                "customer_message": row["customer_message"],
                "brand_reply": row["brand_reply"],
            })
        return results