# core_rag.py
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# --------- CONFIG ---------
QDRANT_URL = os.getenv("API_ENDPOINT")
QDRANT_API_KEY = os.getenv("API_KEY")
COLLECTION_NAME = "thailand_content"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --------- SETUP CLIENTS (lazy-ish singletons) ---------
print("Loading embedding model (core_rag)...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Connecting to Qdrant (core_rag)...")
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

print("Setting up Groq client (core_rag)...")
llm = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# --------- HELPERS ---------
def embed(texts):
    return embedder.encode(texts, show_progress_bar=False)

def retrieve_context(query: str, top_k: int = 5):
    query_vec = embed([query])[0].tolist()

    result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    ).points

    contexts = []
    for point in result:
        contexts.append(
            {
                "score": point.score,
                "text": point.payload.get("text"),
                "url": point.payload.get("url"),
                "city": point.payload.get("city"),
                "tags": point.payload.get("tags"),
            }
        )
    return contexts

def build_context_block(contexts):
    lines = []
    for i, c in enumerate(contexts, start=1):
        lines.append(
            f"[{i}] (city={c['city']}, tags={c['tags']})\n{c['text']}\n"
        )
    return "\n".join(lines)

def answer_question(question: str, top_k: int = 5):
    contexts = retrieve_context(question, top_k=top_k)
    if not contexts:
        return "I don't have any saved content that matches this yet.", contexts

    context_block = build_context_block(contexts)

    system_prompt = (
        "You are my personal Thailand trip planner. "
        "You can ONLY use the context I provide from my saved content. "
        "If the context does not contain the answer, explicitly say you don't know. "
        "Do NOT invent specific hotel, bar, restaurant or tour names if they are not present in the context. "
        "Prefer concrete itineraries, places, and practical tips that clearly come from the context."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Here is context from my saved videos/articles:\n\n"
        f"{context_block}\n\n"
        "Using ONLY this context, answer the question. "
        "If the context is not enough, say 'I don't have this in your saved content yet.' "
        "At the end, list which [numbers] you used as 'Sources: [1, 3, ...]'."
    )

    completion = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.25,
    )

    answer = completion.choices[0].message.content
    return answer, contexts
