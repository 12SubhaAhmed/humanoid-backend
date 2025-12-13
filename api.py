import os
import cohere
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# -------------------------------------
# CONFIG
# -------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

BOOK_DIR = r"E:\AI Native\Hackathon-1\humanoid-robotics\website\docs" 
COLLECTION_NAME = "humanoid_robotics"
EMBED_MODEL = "embed-english-v3.0"

# -------------------------------------
# INIT CLIENTS
# -------------------------------------
cohere_client = cohere.Client(COHERE_API_KEY)

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)
BATCH_SIZE = 50 
# FUNCTIONS
# -------------------------------------

# Chunk text into smaller pieces
def chunk_text(text, max_chars=1200):
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + max_chars
        if end >= text_len:
            chunks.append(text[start:])
            break

        # Find last period within the chunk for clean split
        split_pos = text.rfind(". ", start, end)
        if split_pos == -1:
            split_pos = end

        chunks.append(text[start:split_pos + 1].strip())
        start = split_pos + 1

    return chunks

# Batch embed chunks using Cohere
def embed_batch(chunks_batch):
    response = cohere_client.embed(
        model=EMBED_MODEL,
        input_type="search_query",
        texts=chunks_batch
    )
    return response.embeddings

# Create or reset Qdrant collection
def create_collection():
    print("\nCreating Qdrant collection...")
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1024,  # Cohere embed dimension
            distance=Distance.COSINE
        )
    )

# Save a batch of chunks to Qdrant
def save_batch_to_qdrant(chunks, embeddings, start_id, sources):
    points = []
    for i, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=start_id + i,
                vector=embeddings[i],
                payload={
                    "source": sources[i],
                    "text": chunk,
                    "chunk_id": start_id + i
                }
            )
        )
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

# Ingest Markdown files with batching
def ingest_from_markdown():
    create_collection()
    global_id = 1

    all_chunks = []
    all_sources = []

    # Step 1: Read and chunk all files
    for root, _, files in os.walk(BOOK_DIR):
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()

                chunks = chunk_text(text)
                all_chunks.extend(chunks)
                all_sources.extend([path] * len(chunks))

    print(f"\nTotal chunks to embed: {len(all_chunks)}")

    # Step 2: Embed in batches
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_chunks = all_chunks[i:i + BATCH_SIZE]
        batch_sources = all_sources[i:i + BATCH_SIZE]

        embeddings = embed_batch(batch_chunks)
        save_batch_to_qdrant(batch_chunks, embeddings, global_id, batch_sources)
        print(f"Saved chunks {global_id} to {global_id + len(batch_chunks) - 1}")
        global_id += len(batch_chunks)

    print("\n✔️ Ingestion completed! Total chunks stored:", global_id - 1)

# -------------------------------------
# MAIN
# -------------------------------------
if __name__ == "__main__":
    ingest_from_markdown()
