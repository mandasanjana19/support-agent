# Twitter Customer Support AI Agent

An evaluation-first AI support agent built on the Kaggle Customer Support on Twitter dataset.

## System Architecture

1. **Language & Runtime:** Python 3.14.6
2. **LLM Engine:** Groq API (`llama-3.1-8b-instant`) for intent classification, response drafting, and judge evaluation
3. **Grounding Engine:** Scikit-Learn TF-IDF + Cosine Similarity over historical brand resolutions (lightweight, pure-Python)
4. **Routing Engine:** Rule-based and model-assisted auto-reply vs. human escalation logic with explicit reasoning
5. **Evaluation Engine:** Automated LLM-as-a-judge rubric + human-alignment agreement verification

## Setup
1. Clone the repository.
2. Create virtual environment and install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and supply `GROQ_API_KEY`.
4. Run pipeline: `python -m src.ingest`
5. python src/ingest.py inspect
6. python src/ingest.py filter --brand AppleSupport --limit 20000 #save 20,000 rows to data\processed\AppleSupport_outbound.csv
7. python src\threads.py --brand AppleSupport--limit 3000
#Saved 2,998 customer<->brand pairs to data\processed\AppleSupport_pairs.csv
8. initial taxonomy (#present in intents.py)
9. python src/intents.py discover --brand AppleSupport --k 8
#get samples and update the final taxonomy
10. python eval\build_golden_set.py --brand AppleSupport --k 8 --total 200
This creates data\golden\golden_set.csv — 200 rows, gold_intent/gold_escalate/etc. columns empty, waiting for you.
11. Labeling tool (so you can label fast without editing raw CSV by hand)
Create eval/label_tool.py:
12. python eval\label_tool.py. hand-lable real judgement calls