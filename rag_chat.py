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

# Groq via OpenAI client (OpenAI-compatible API)
# Docs: https://console.groq.com/docs/openai 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --------- SETUP CLIENTS ---------
print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Connecting to Qdrant...")
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

print("Setting up Groq client...")
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
    """Format retrieved chunks into a numbered block for the LLM."""
    lines = []
    for i, c in enumerate(contexts, start=1):
        lines.append(
            f"[{i}] (city={c['city']}, tags={c['tags']})\n{c['text']}\n"
        )
    return "\n".join(lines)

def answer_question(question: str, top_k: int = 5):
    contexts = retrieve_context(question, top_k=top_k)
    if not contexts:
        return "I don't have any saved content that matches this yet.", []

    context_block = build_context_block(contexts)

    system_prompt = (
        "You are my personal Thailand trip planner. "
        "You can ONLY use the context I provide from my saved content. "
        "If the context does not contain the answer, say you don't know. "
        "Prefer concrete itineraries, places, and practical tips."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Here is context from my saved videos/articles:\n\n"
        f"{context_block}\n\n"
        "Using ONLY this context, answer the question. "
        "At the end, list which [numbers] you used as 'Sources: [1, 3, ...]'."
    )

    # Pick a Groq model; e.g. Llama 3.1 8B instant or 3.3 70B versatile.
    completion = llm.chat.completions.create(
        model="llama-3.1-8b-instant",   # or "llama-3.3-70b-versatile" if you want heavier
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.25,
    )

    answer = completion.choices[0].message.content
    return answer, contexts

if __name__ == "__main__":
    while True:
        try:
            q = input("\nAsk about Thailand (or 'exit'): ")
        except EOFError:
            break

        if not q or q.lower().strip() in {"exit", "quit"}:
            break

        answer, ctx = answer_question(q)
        print("\n=== ANSWER ===")
        print(answer)
        print("\n=== RAW CONTEXT (debug) ===")
        for c in ctx:
            print("----")
            print("City:", c["city"], "| Tags:", c["tags"], "| Score:", c["score"])
            print(c["text"])
