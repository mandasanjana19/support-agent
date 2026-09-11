# Decision Log

1. **TF-IDF Retrieval over Vector DBs / Embeddings**: Standardized on Scikit-Learn TF-IDF + Cosine Similarity to maintain light, pure-Python dependencies compatible with Python 3.14.6 without compiled torch binaries.
2. **Google Gemini API Free Tier**: Selected Gemini 2.5 Flash as the primary LLM engine for cost efficiency, generous rate limits, and structured evaluation.