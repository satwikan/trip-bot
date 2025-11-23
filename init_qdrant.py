from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

# --------- CONFIG ---------
QDRANT_URL = os.getenv("API_ENDPOINT")
QDRANT_API_KEY = os.getenv("API_KEY")
COLLECTION_NAME = "thailand_content"

# We'll use a 768-dim embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# --------- SETUP CLIENTS ---------
print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Connecting to Qdrant...")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# --------- STEP 1: CREATE COLLECTION IF NOT EXISTS ---------
def create_collection_if_needed():
    collections = client.get_collections().collections
    existing = {c.name for c in collections}
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists, skipping creation.")
        return

    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest.VectorParams(
            size=768,             # dimension of all-MiniLM-L6-v2 embeddings
            distance=rest.Distance.COSINE,
        ),
    )
    print("Collection created.")

# --------- STEP 2: SAMPLE DOCUMENTS ---------
SAMPLE_TEXTS = [
    {
        "id": 1,
        "text": "Bangkok has amazing rooftop bars around Sukhumvit and Silom. "
                "Sunset at a rooftop bar is a great way to start a night out.",
        "url": "https://example.com/bangkok-rooftop",
        "city": "Bangkok",
        "tags": ["nightlife", "rooftop", "city"],
    },
    {
        "id": 2,
        "text": "In Krabi, an island-hopping tour to Railay Beach and the nearby islands "
                "is one of the best day trips. Clear water and limestone cliffs.",
        "url": "https://example.com/krabi-islands",
        "city": "Krabi",
        "tags": ["beach", "islands", "daytrip"],
    },
    {
        "id": 3,
        "text": "Chiang Mai is known for its temples and cafes. "
                "You can spend a day visiting Wat Phra Singh and exploring Nimmanhaemin cafes.",
        "url": "https://example.com/chiangmai-temples-cafes",
        "city": "Chiang Mai",
        "tags": ["temples", "cafes", "culture"],
    },
]

def embed(texts):
    # sentence-transformers expects a list of strings
    vectors = embedder.encode(texts, show_progress_bar=False)
    return vectors

# --------- STEP 3: INSERT SAMPLE DATA ---------
def insert_samples():
    print("Embedding sample texts...")
    texts = [item["text"] for item in SAMPLE_TEXTS]
    vectors = embed(texts)

    print("Upserting points into Qdrant...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            rest.PointStruct(
                id=item["id"],
                vector=vec.tolist(),
                payload={
                    "text": item["text"],
                    "url": item["url"],
                    "city": item["city"],
                    "tags": item["tags"],
                },
            )
            for item, vec in zip(SAMPLE_TEXTS, vectors)
        ],
    )
    print("Sample data inserted.")

# --------- STEP 4: TEST SEARCH ---------
def test_search(query: str, top_k: int = 3):
    print(f"\nRunning search for query: {query!r}")
    query_vec = embed([query])[0].tolist()

    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    ).points

    for i, point in enumerate(search_result, start=1):
        print(f"\nResult #{i}")
        print(f"  Score: {point.score:.4f}")
        print(f"  City: {point.payload.get('city')}")
        print(f"  Tags: {point.payload.get('tags')}")
        print(f"  Text: {point.payload.get('text')}")
        print(f"  URL:  {point.payload.get('url')}")

def retrieve_context(query: str, top_k: int = 5):
    """Return top_k chunks from Qdrant as a list of dicts."""
    query_vec = embed([query])[0].tolist()

    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    ).points

    contexts = []
    for point in search_result:
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


if __name__ == "__main__":
    create_collection_if_needed()
    insert_samples()

    query = "I want beaches and island hopping near Krabi"
    ctx = retrieve_context(query)
    print("Top contexts for:", query)
    for c in ctx:
        print("----")
        print("Score:", c["score"])
        print("City:", c["city"])
        print("Text:", c["text"])
