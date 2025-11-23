# Thailand Brain 🧠🇹🇭

Personal RAG (Retrieval-Augmented Generation) system for planning a Thailand trip.

You:
- Save content (videos, notes, etc.) into a vector database.
- Ask natural language questions.
- Get answers grounded in your own saved content.

This project currently includes:
- Qdrant Cloud as vector DB
- Local embeddings via `sentence-transformers`
- LLM via Groq's OpenAI-compatible API
- FastAPI HTTP API (`/chat`) you can later connect to Telegram

---

## 1. Architecture (High Level)

**Components:**

1. **Qdrant Cloud**  
   - Stores embeddings + metadata for your Thailand content.  
   - Collection: `thailand_content`.

2. **Embeddings (local)**  
   - Model: `sentence-transformers/all-mpnet-base-v2`  
   - Converts text → 768-dim vectors.

3. **LLM (Groq)**  
   - You call Groq's OpenAI-compatible API (e.g. `llama-3.1-8b-instant`)  
   - Used only at query time to turn:
     - your question + retrieved chunks  
     into a nice natural-language answer.

4. **FastAPI backend**  
   - `/chat` endpoint:
     - takes `{"question": "..."}`  
     - does vector search in Qdrant  
     - calls LLM with context  
     - returns answer + sources.

At the moment, content is **seeded manually** via `init_qdrant.py` (3 sample Thailand texts). Later you’ll replace that with a real ingestion pipeline (YouTube transcripts, notes, etc.).

---

## 2. Prerequisites

- Python 3.10+ recommended
- A Qdrant Cloud account + free cluster
- A Groq API key

### 2.1 Qdrant Cloud

1. Create an account at Qdrant Cloud.
2. Create a **free cluster**.
3. Note:
   - Cluster **URL** (e.g. `https://XXXXXXXXXX.us-east-0-0.aws.cloud.qdrant.io`)
   - **API key** with access to that cluster.

These go into your `.env` file as `API_ENDPOINT` and `API_KEY`.

### 2.2 Groq API key

1. Create an account at [console.groq.com](https://console.groq.com/).
2. Generate an API key.
3. Put it into your `.env` as `GROQ_API_KEY`.

---

## 3. Setup

Clone or create the project directory:

```bash
mkdir thailand-brain
cd thailand-brain
