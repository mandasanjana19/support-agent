"""
Phase 6c: The actual agent. Orchestrates:
  1. Classify intent (LLM, using TAXONOMY definitions)
  2. Retrieve grounding examples (TF-IDF similarity over historical pairs)
  3. Draft a reply (LLM, grounded in retrieved examples)
  4. Decide auto-handle vs escalate, with reason (router.py)

Usage:
    python src/agent.py --brand AmazonHelp --message "Where is my order?!"
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, os.path.dirname(__file__))
from intents import TAXONOMY          # noqa: E402
from retrieval import HistoricalRetriever  # noqa: E402
from router import decide_routing     # noqa: E402

load_dotenv()

LLM_MODEL = "llama-3.3-70b-versatile"

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env file.")
        _client = Groq(api_key=api_key)
    return _client
def classify_intent(message: str) -> str:
    taxonomy_desc = "\n".join(f"- {name}: {desc}" for name, desc in TAXONOMY.items())
    prompt = f"""You are classifying a customer support tweet into exactly one intent.

Intents:
{taxonomy_desc}

Customer message: "{message}"

Respond with ONLY the intent name (one of the keys above), nothing else."""

    resp = get_client().chat.completions.create(
        model=LLM_MODEL,
        max_tokens=30,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content.strip().lower().replace(" ", "_")
    return raw if raw in TAXONOMY else "other"


def draft_reply(message: str, intent: str, retrieved: list[dict]) -> str:
    if retrieved:
        examples_block = "\n\n".join(
            f'Past customer message: "{r["customer_message"]}"\n'
            f'Past brand reply: "{r["brand_reply"]}"'
            for r in retrieved
        )
    else:
        examples_block = "(no similar historical examples found)"

    prompt = f"""You are drafting a reply as this brand's support account, in their voice
and style, based on how they have actually resolved similar issues before.

Customer's message (intent: {intent}): "{message}"

Similar past resolutions from this brand's real history:
{examples_block}

Write a short, on-brand reply (1-3 sentences, Twitter-appropriate). Match the
tone and typical structure of the past replies above. Do not invent policy
details (refund amounts, timelines) that aren't supported by the examples —
if unsure, ask the customer to DM order/account details instead."""

    resp = get_client().chat.completions.create(
        model=LLM_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

def run_agent(brand: str, message: str, exclude_customer_message: str | None = None,
              retriever: HistoricalRetriever | None = None) -> dict:
    if retriever is None:
        retriever = HistoricalRetriever(brand)

    intent = classify_intent(message)
    retrieved = retriever.retrieve(message, k=3, exclude_customer_message=exclude_customer_message)
    reply = draft_reply(message, intent, retrieved)
    routing = decide_routing(message, intent, retrieved)

    return {
        "customer_message": message,
        "intent": intent,
        "retrieved_examples": retrieved,
        "draft_reply": reply,
        "escalate": routing["escalate"],
        "escalate_reason": routing["reason"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", required=True)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()

    result = run_agent(args.brand, args.message)
    print(json.dumps(result, indent=2))