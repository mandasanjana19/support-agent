"""
Phase 8: Clean demo interface for the support agent.
Run: streamlit run app/streamlit_app.py
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agent import run_agent            # noqa: E402
from retrieval import HistoricalRetriever  # noqa: E402
from intents import TAXONOMY           # noqa: E402

st.set_page_config(page_title="AI Support Agent", page_icon="🎧", layout="centered")

BRAND = "AppleSupport"

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .escalate-badge {
        display: inline-block; padding: 4px 14px; border-radius: 999px;
        font-weight: 600; font-size: 0.85rem;
    }
    .escalate-yes { background-color: #4a1414; color: #ff8080; border: 1px solid #ff4d4d; }
    .escalate-no { background-color: #143a1e; color: #7cffa0; border: 1px solid #2ecc71; }
    .intent-badge {
        display: inline-block; padding: 4px 14px; border-radius: 999px;
        background-color: #1e2a4a; color: #8fb3ff; border: 1px solid #3d5ba0;
        font-weight: 600; font-size: 0.85rem;
    }
    .reply-box {
    background-color: #e6f4ea; border-left: 4px solid #2e7d32;
    padding: 16px; border-radius: 8px; margin-top: 8px;
    color: #1b5e20;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(f"🎧 AI Support Agent — {BRAND}")
st.caption("Classify → retrieve grounding → draft reply → route decision")


@st.cache_resource
def get_retriever():
    return HistoricalRetriever(BRAND)


with st.expander("ℹ️ Intent taxonomy this agent uses"):
    for name, desc in TAXONOMY.items():
        st.markdown(f"**{name}** — {desc}")

message = st.text_area(
    "Customer message",
    placeholder="e.g. My battery has been draining so fast ever since the last update!",
    height=100,
)

if st.button("Run agent", type="primary", disabled=not message.strip()):
    with st.spinner("Classifying, retrieving grounding, drafting reply..."):
        retriever = get_retriever()
        result = run_agent(BRAND, message, retriever=retriever)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<span class="intent-badge">🏷️ {result["intent"]}</span>',
                    unsafe_allow_html=True)
    with col2:
        badge_class = "escalate-yes" if result["escalate"] else "escalate-no"
        label = "🚨 ESCALATE" if result["escalate"] else "✅ AUTO-HANDLE"
        st.markdown(f'<span class="escalate-badge {badge_class}">{label}</span>',
                    unsafe_allow_html=True)

    st.markdown("**Routing reason:**")
    st.info(result["escalate_reason"])

    st.markdown("**Drafted reply:**")
    st.markdown(f'<div class="reply-box">{result["draft_reply"]}</div>',
                unsafe_allow_html=True)

    with st.expander(f"📚 Grounding evidence ({len(result['retrieved_examples'])} examples used)"):
        if result["retrieved_examples"]:
            for i, ex in enumerate(result["retrieved_examples"], 1):
                st.markdown(f"**Match {i}** (similarity: {ex['similarity']:.2f})")
                st.markdown(f"> Customer: {ex['customer_message']}")
                st.markdown(f"> Brand replied: {ex['brand_reply']}")
                st.divider()
        else:
            st.warning("No similar historical examples found — this is part of why "
                       "the routing decision may lean toward escalation.")
else:
    st.caption("Enter a message above and click **Run agent** to see the full pipeline output.")