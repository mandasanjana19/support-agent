# Decision Log

A plain list of the non-obvious decisions made while building this system,
and why. Ordered roughly by when they came up.

1. **TF-IDF retrieval over embeddings / a vector DB.** Standardized on
   scikit-learn `TfidfVectorizer` + cosine similarity instead of
   `sentence-transformers`/`torch`-based embeddings. The dev machine runs
   Python 3.14.6, and wheel availability for compiled libraries like `torch`
   on such a new interpreter version was uncertain — not worth risking the
   whole pipeline on. TF-IDF is deterministic, dependency-light, and
   sufficient at our data scale (thousands of rows, not millions).

2. **Groq API for the LLM engine.** Chosen over Anthropic/OpenAI for this
   build specifically because of its free tier, low latency, and
   OpenAI-compatible API surface, making it a drop-in replacement once the
   original Anthropic-based plan changed.

3. **Brand selection: AppleSupport over AmazonHelp.** AmazonHelp's replies
   are dominated by generic "please DM us your order number" deflections,
   giving a retrieval-grounded agent very little real resolution text to
   learn from. AppleSupport's threads contain concrete troubleshooting
   language and resolution steps stated directly in-thread, making grounded
   drafting meaningfully testable.

4. **Retrieval self-exclusion during evaluation.** `HistoricalRetriever.retrieve()`
   accepts an `exclude_customer_message` parameter so that, when evaluating
   on a golden-set example that also exists in the historical pairs corpus,
   the retriever cannot trivially "ground" a reply in its own gold answer —
   avoiding an easy, invisible evaluation-leakage bug.

5. **Router uses deterministic rules, not LLM judgment, for escalation.**
   A real design choice, not a shortcut: a keyword/policy/confidence-based
   rule engine is auditable and cannot be talked out of escalating a risky
   case by clever phrasing, unlike an LLM asked to "decide" on the spot.
   Tradeoff, stated explicitly: it will miss paraphrased risk that doesn't
   match a keyword or policy-escalated intent (see failure mode #3 in the
   report).

6. **Word-boundary regex for risk keywords, not substring matching.**
   Discovered during evaluation that the keyword `"sue"` matched inside the
   word `"issues"` via naive `kw in text` substring checks, causing false
   escalations. Fixed with `\b`-anchored regex matching. Verified
   improvement: escalation accuracy on the same test split rose from 0.90 to
   0.93 immediately after the fix.

7. **Blanket policy-escalation for the `apple_id_account` intent.**
   Deliberately escalates the *entire* intent category regardless of
   retrieval confidence, even though this causes false positives on routine
   account-management questions (see failure mode #2 in the report). Kept
   as-is rather than refined further, given the higher cost of a human
   missing a genuine account-hijack case versus the lower cost of
   unnecessary review on a benign one.

8. **Switched the intent taxonomy mid-build, and relabeled the full golden
   set.** Originally labeled 200 examples under a generic, domain-agnostic
   e-commerce taxonomy. A trivial majority-class baseline scored 95%
   accuracy under it — evidence the taxonomy didn't meaningfully
   differentiate AppleSupport's real traffic. Discarded that labeling pass
   and rebuilt a brand-specific 8-intent taxonomy from actual KMeans cluster
   inspection, then relabeled all 200 examples against it. The original
   labeled set is preserved in the repo
   (`golden_set_v1_generic_taxonomy_ARCHIVED.csv`) as a record of the
   iteration.

9. **Golden set sampling: stratified by KMeans cluster, not pure random.**
   Pure random sampling from a corpus this skewed risks under-representing
   rare intents entirely. Stratifying by TF-IDF/KMeans cluster (with a
   per-cluster floor) ensures the golden set contains examples from every
   discovered pattern, not just the dominant one.

10. **Report agreement metrics as MAE / exact-match / within-1-point rate,
    not Pearson correlation, for the judge↔human agreement check.** With
    n=20 and both judge and human scores clustered near the ceiling of the
    0–5 scale, correlation is numerically unstable (and undefined when a
    dimension has zero variance, as happened for `tone_fit`). MAE and
    exact-match rate remain meaningful under low score variance and are
    reported as the primary evidence instead.

11. **`reasoning_effort="low"` set explicitly for Groq's `gpt-oss` models.**
    Discovered that these reasoning-capable models, at default settings,
    could consume the entire `max_tokens` budget on internal reasoning and
    return an empty final answer. Fixed by explicitly requesting low
    reasoning effort and raising `max_tokens` as a safety margin, applied
    consistently to both the drafting and judging calls.

12. **Subsampled ~3,000 customer↔brand pairs, not the full 106,000+
    AppleSupport tweets.** Keeps the retrieval index construction and full
    pipeline reproducible in well under 15 minutes on a normal laptop, in
    line with the assignment's own guidance that a subsample is expected.

13. **Streamlit for the demo interface, not a custom frontend.** Fastest
    path to a clean, functional, presentable UI without introducing
    additional infrastructure or frontend build tooling for what is
    fundamentally a demo/evaluation artifact, not a production product.

14. **Added an overwrite guard to `build_golden_set.py` after an incident.**
    An earlier re-run of the golden-set generation script silently
    regenerated `golden_set.csv` from scratch, wiping roughly 70 rows of
    already-completed hand-labeling with no warning. Added a check that
    refuses to overwrite a file that already contains labeled rows unless
    it is explicitly deleted or renamed first — a process safeguard learned
    the hard way, not a modeling decision, but one worth recording.

15. **Enforced stratified train/test splitting in baseline evaluation.**
    Given the golden set's real class imbalance (one intent makes up
    ~38.5% of examples), an unstratified split risks dropping a whole rare
    class into either train or test by chance, producing misleadingly
    volatile results. `train_test_split(..., stratify=gold_intent)` keeps
    class proportions consistent across the split.
