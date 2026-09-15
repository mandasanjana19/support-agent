"""
Phase 7b: LLM-as-judge for reply quality. (Groq version)
"""

import json
import os
import re
import sys

from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, os.path.dirname(__file__))

load_dotenv()

JUDGE_MODEL = "openai/gpt-oss-120b"
_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set.")
        api_key = api_key.strip("'\" \t\r\n")
        _client = Groq(api_key=api_key)
    return _client


JUDGE_PROMPT_TEMPLATE = """You are an impartial evaluator scoring an AI-drafted customer support
reply. Be strict and skeptical -- do not give high scores by default.

CUSTOMER MESSAGE:
"{message}"

HISTORICAL EXAMPLES THE REPLY WAS SUPPOSED TO BE GROUNDED IN:
{examples_block}

AI-DRAFTED REPLY TO EVALUATE:
"{reply}"

Score the reply on three dimensions, each 0-5 (0=terrible, 5=excellent):

1. groundedness: Does the reply avoid inventing specific facts, policies,
   timelines, or promises that are NOT supported by the historical examples
   above? A reply that makes a vague, safe statement scores HIGH. A reply
   that invents a specific refund amount/timeline not seen in examples
   scores LOW, even if it sounds plausible.

2. tone_fit: Does the reply's tone, length, and structure match the style
   of the historical examples (if any were provided)?

3. helpfulness: Does the reply give the customer a clear, actionable next
   step or acknowledgment, rather than being vague/generic filler?

Respond with ONLY valid JSON in this exact shape, no other text, no markdown fences:
{{
  "groundedness": <int 0-5>,
  "groundedness_rationale": "<one short sentence>",
  "tone_fit": <int 0-5>,
  "tone_fit_rationale": "<one short sentence>",
  "helpfulness": <int 0-5>,
  "helpfulness_rationale": "<one short sentence>"
}}"""


def judge_reply(message: str, retrieved: list[dict], reply: str) -> dict:
    if retrieved:
        examples_block = "\n\n".join(
            f'- Past customer: "{r["customer_message"]}"\n  Past brand reply: "{r["brand_reply"]}"'
            for r in retrieved
        )
    else:
        examples_block = "(none provided to the drafting model)"

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        message=message, examples_block=examples_block, reply=reply
    )

    resp = get_client().chat.completions.create(
        model=JUDGE_MODEL,
        max_tokens=400,
        temperature=0,
        reasoning_effort="low",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = (resp.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "groundedness": None, "groundedness_rationale": "PARSE_ERROR",
            "tone_fit": None, "tone_fit_rationale": "PARSE_ERROR",
            "helpfulness": None, "helpfulness_rationale": "PARSE_ERROR",
            "_raw_response": raw,
        }
    return parsed