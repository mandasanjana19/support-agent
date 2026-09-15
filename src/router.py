"""
Phase 6b: Decide auto-handle vs escalate, with a stated reason.

Combines:
  - Rule-based risk signals (fast, deterministic, auditable)
  - Retrieval confidence (low similarity to any historical resolution
    means we have no good precedent to ground a reply in -> escalate)
  - Intent-based policy (some intents are policy-escalated regardless
    of confidence, e.g. account security)
"""

RISK_KEYWORDS = [
    "fraud", "hacked", "hack", "lawyer", "lawsuit", "legal action",
    "sue", "scam", "unauthorized", "stolen", "police", "identity theft",
]

# Intents that should always route to a human regardless of confidence —
# these are judgment calls tied to your brand/risk tolerance, stated here
# explicitly so they can be debated/changed, not buried in prompt text.
ALWAYS_ESCALATE_INTENTS = {"apple_id_account"}

LOW_CONFIDENCE_SIMILARITY_THRESHOLD = 0.12


def decide_routing(message: str, intent: str, retrieved: list[dict]) -> dict:
    lower = message.lower()

    hit_keywords = [kw for kw in RISK_KEYWORDS if kw in lower]
    if hit_keywords:
        return {
            "escalate": True,
            "reason": f"Risk keyword(s) detected: {', '.join(hit_keywords)}. "
                      f"Routed to human out of caution regardless of intent/confidence.",
        }

    if intent in ALWAYS_ESCALATE_INTENTS:
        return {
            "escalate": True,
            "reason": f"Intent '{intent}' is policy-escalated (security/account access) "
                      f"regardless of retrieval confidence.",
        }

    best_similarity = retrieved[0]["similarity"] if retrieved else 0.0
    if best_similarity < LOW_CONFIDENCE_SIMILARITY_THRESHOLD:
        return {
            "escalate": True,
            "reason": f"No close historical precedent found (best similarity "
                      f"{best_similarity:.2f} < threshold {LOW_CONFIDENCE_SIMILARITY_THRESHOLD}). "
                      f"Not confident the draft reply is grounded in a real resolution pattern.",
        }

    return {
        "escalate": False,
        "reason": f"Intent '{intent}' is not policy-escalated, no risk keywords found, "
                  f"and a similar historical resolution exists "
                  f"(similarity {best_similarity:.2f}).",
    }