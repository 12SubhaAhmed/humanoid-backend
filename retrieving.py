import cohere
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import os


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

cohere_client = cohere.Client(COHERE_API_KEY)

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

#--------------
#FUNCTION
#--------------
def get_embedding(text):
    response = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_query",
        texts=[text]
    )
    return response.embeddings[0]


def retrieve(query):
    embedding = get_embedding(query)
    
    result = qdrant.query_points(
        collection_name="humanoid_robotics",
        query=embedding,
        limit=5,
        with_vectors=False
    )

    chunks = []
    for point in result.points:
        if point.score < 0.35:
            continue
        
        chunks.append(point.payload["text"])

    if not chunks:
        return ["No relevant content found in book."]

    return chunks[:3]

    