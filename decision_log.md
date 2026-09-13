# Decision Log
#facing choices and deciding to take which option

1. **TF-IDF Retrieval over Vector DBs / Embeddings**: Standardized on Scikit-Learn TF-IDF + Cosine Similarity to maintain light, pure-Python dependencies compatible with Python 3.14.6 without compiled torch binaries.
from sklearn.feature_extraction.text import TfidfVectorizer
What it is: A text-processing class from Scikit-Learn that stands for Term Frequency-Inverse Document Frequency.
Why it’s used: Machine learning algorithms (K-Means) cannot process raw text strings directly—they require numbers. TfidfVectorizer converts tweet messages into numerical matrices by calculating word frequencies while penalizing common filler words (giving higher mathematical weight to distinct terms like "battery", "update", or "iPhone").
2. **Groq API for LLM Engine**: Selected Groq API (using Llama 3.1 8B / Qwen models) due to its generous free tier (14,400 requests/day), sub-second latency, and standard OpenAI-compatible API format.
3. **Brand Selection Decision**: We will select AppleSupport (106,860 tweets) over AmazonHelp.
In-thread resolution vs. DM deflection: AmazonHelp relies almost exclusively on automatic "Please DM us your order number" replies.
Rich troubleshooting data: AppleSupport contains clear technical issue statements (iOS updates, battery drain, device sync, Apple ID locks) and explicit resolution steps (reset settings, force restart, toggle specific settings) directly in tweet threads. This gives our scikit-learn grounding retriever concrete text patterns to learn from.