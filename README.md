# AI Support Agent for AppleSupport (Twitter)

An evaluation-first AI support agent built on the Kaggle *Customer Support on
Twitter* dataset. Given an incoming customer tweet, the agent (1) classifies
it into one of eight hand-defined intents, (2) drafts a reply grounded in how
AppleSupport has historically resolved similar issues, and (3) decides
whether the message can be auto-handled or should be escalated to a human,
with a stated reason.

Full write-up: `project_report.pdf`.
Non-obvious decisions: `decision_log.md`.

---

## Headline Results

| Metric | Trivial baseline | Simple baseline | **Agent** |
|---|---|---|---|
| Intent accuracy | 0.38 | 0.45 | **0.67** |
| Intent F1 (macro) | 0.07 | 0.34 | **0.69** |
| Escalation accuracy | 0.98 | 0.95 | 0.93 |

(Escalation accuracy is misleading in isolation due to severe label
imbalance — see `project_report.pdf` Section 5, "What is misleading about my
headline number?")

LLM-judge reply quality (0–5): groundedness 4.87, tone fit 4.70,
helpfulness 4.72. Judge↔human agreement (n=20): groundedness MAE 0.20,
tone fit MAE 0.25, helpfulness MAE 0.90 — see `project_report.pdf` for full table and
interpretation.

---

## System Architecture

1. **Language & Runtime:** Python 3.14.6
2. **LLM Engine:** Groq API (`llama-3.1-8b-instant` for classification/drafting,
   `openai/gpt-oss-120b` with `reasoning_effort="low"` for judging) — chosen for
   free-tier availability, low latency, and an OpenAI-compatible API.
3. **Grounding Engine:** scikit-learn TF-IDF + cosine similarity over historical
   brand resolutions (lightweight, pure-Python, no compiled binaries — see
   `decision_log.md` item 1).
4. **Routing Engine:** deterministic, rule-based auto-handle vs. escalate logic
   with an explicit stated reason per decision (not LLM judgment — auditability
   by design, see `decision_log.md` item 5).
5. **Evaluation Engine:** automated intent/escalation metrics + LLM-as-judge
   reply-quality rubric + a blind human-agreement check against the judge.

---

## Setup

```bash
git clone <this-repo-url>
cd support-agent
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env       # then fill in GROQ_API_KEY
```

Get a free Groq API key at https://console.groq.com.

---

## Reproducing the Headline Results (~15 minutes)

The full raw dataset (~3M tweets) is **not required** to reproduce the
headline numbers — the repo ships the processed subsample and the completed
200-row golden set already committed under `data/processed/` and
`data/golden/`. If you only want to reproduce results, skip straight to
**Step 6**.

**Full pipeline from raw data** (only needed to rebuild from scratch):

```bash
# 1. Download raw dataset (Kaggle account + API token required)
#    https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data\raw --unzip

# 2. Inspect brand volume, then filter to AppleSupport
python src\ingest.py inspect
python src\ingest.py filter --brand AppleSupport --limit 20000

# 3. Reconstruct (customer_message, brand_reply) pairs
python src\threads.py --brand AppleSupport --limit 3000

# 4. (Optional) Re-run taxonomy discovery — taxonomy itself is already
#    finalized in src/intents.py
python src\intents.py discover --brand AppleSupport --k 8

# 5. Regenerate golden set sample (WARNING: will refuse to overwrite an
#    already-labeled file — see decision_log.md item 14)
python eval\build_golden_set.py --brand AppleSupport --k 8 --total 200
python eval\label_tool.py   # manual hand-labeling — not reproducible by design
```

**Reproduce headline results** (uses the committed golden set):

```bash
# 6. Two baselines
python src\baselines.py evaluate

# 7. Full agent evaluation: classify + retrieve + draft + route + LLM-judge
#    scoring, on the same held-out test split as the baselines
python eval\run_eval.py --brand AppleSupport

# 8. (Optional) Judge/human agreement check — requires manual blind scoring,
#    already completed and committed at eval/results/judge_human_agreement.json
python eval\human_agreement.py score
```

Steps 6–7 alone (no raw data download required) reproduce every number in
the Results table above in well under 15 minutes.

**Try the agent directly:**

```bash
python src\agent.py --brand AppleSupport --message "My battery has been draining so fast ever since the last update!"
```

**Launch the demo interface:**

```bash
streamlit run app\streamlit_app.py
```

**demo**: [▶ Open Demo](https://drive.google.com/file/d/1HF4s3fBZubDoeRRUJqDuSg-FYRXyIXV2/view?usp=sharing)

---

## Folder Structure

```
support-agent/
├── data/
│   ├── raw/            # gitignored — full Kaggle CSV (not committed)
│   ├── processed/       # AppleSupport_pairs.csv — committed subsample
│   └── golden/          # golden_set.csv — 200 hand-labeled examples, committed
├── src/
│   ├── ingest.py         # raw data inspection + brand filtering
│   ├── threads.py        # conversation pair reconstruction
│   ├── intents.py         # intent taxonomy + discovery clustering
│   ├── retrieval.py       # TF-IDF grounding retrieval
│   ├── router.py          # escalation decision logic
│   ├── agent.py            # orchestrates classify -> retrieve -> draft -> route
│   ├── judge.py             # LLM-as-judge reply scoring
│   └── baselines.py         # trivial + simple baselines
├── eval/
│   ├── build_golden_set.py  # golden set sampling (has an overwrite guard)
│   ├── label_tool.py         # CLI hand-labeling tool
│   ├── run_eval.py            # full agent eval harness
│   ├── human_agreement.py      # judge <-> human agreement check
│   └── results/                 # all output metrics/CSVs, committed
├── app/
│   └── streamlit_app.py          # demo interface
├── project_report.pdf         # full report
├── decision_log.md
└── requirements.txt
```

---

## Golden Evaluation Set

200 hand-labeled examples, sampled via TF-IDF + KMeans (k=8) stratified
clustering over deduplicated AppleSupport customer messages (ensures rare
intents aren't excluded by chance), then randomly sampled within each
cluster. Each example was manually labeled for `gold_intent`,
`gold_escalate` (yes/no), and a short `gold_escalate_reason` using a custom
resumable CLI tool (`eval/label_tool.py`). Full methodology and label
distribution in `project_report.pdf` Section 2.2.

---

## Known Limitations

- Golden set (n=200) has only 4 true `escalate=yes` examples — escalation
  recall cannot be reliably estimated. See `project_report.pdf` Section 5.
- Retrieval uses TF-IDF, not semantic embeddings (Python 3.14 dependency
  compatibility tradeoff — see `decision_log.md` item 1).
- Single-turn grounding only; full multi-turn thread context is not used.
- See `project_report.pdf` Section 6 for the full "what we'd do next" list.

---

## Citations / Borrowed Material

- Dataset: Kaggle *Customer Support on Twitter*
  (`thoughtvector/customer-support-on-twitter`).
- LLM inference: Groq API (Llama 3.1 8B Instant, gpt-oss-120b).
- No external code repositories were copied; all pipeline code was
  written for this project (with AI coding assistance, per assignment rules).